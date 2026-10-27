"""Closed-wave load generator: N clients send one streamed chat request each at the same time, repeated for K waves.

Prompts are unique random syllables calibrated to P prompt tokens, or with --text windows of a text file from random
offsets (natural text: on a MoE model, expert routing can differ from random syllables); ignore_eos makes every request
generate exactly G tokens.
Wave 0 runs on a fresh server (first allocations, CUDA graph captures): it is marked cold, and callers report it
apart from the warm waves.
Every token is timed twice:
  client: arrival of its streamed chunk at this process. A delivery metric: it adds the HTTP thread, the socket and
          this client to the server's own timing.
  server: timings.predicted_ms of its chunk (timings_per_token), stamped on the inference thread when the token was
          sampled, in ms since the request's first token.
Gaps between tokens are reported for both: itl_* between chunks at the client, server_itl_* per generated token.
They differ where a chunk carries several tokens: a character split over 2-3 byte tokens (emoji, non-Latin scripts)
is sent once it is complete, which shows as one long gap at the client (merged_tokens_pct). Each request's raw
per-chunk times and token counts are saved. Per wave, the prefill phase runs until the last request's first token and
the decode phase from there; gen_tok_s_decode_phase counts the tokens generated in the decode phase (requests that got
their first token early generate during the prefill phase too). A warning goes to stderr when the server's prompt or output token counts are not what was asked.
usage: gen.py --port 8091 --n 4 --prompt-tokens 4096 --gen-tokens 256 --waves 3 --pause 3 --out res.json
"""
import argparse
import http.client
import itertools
import json
import random
import sys
import threading
import time

import runinfo

SYL = ["ka", "lo", "mi", "ne", "ru", "ta", "vo", "se", "di", "pa", "zu", "ge", "fo", "hi", "ba", "ny", "ce", "wo", "xi", "qu",
       "an", "el", "is", "or", "um", "tra", "ple", "stro", "gin", "dex", "mar", "tol", "ven", "sor", "lik", "pon", "rad", "fen"]


def post(port, path, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = json.loads(c.getresponse().read())
    c.close()
    return r


def n_tokens(port, text):
    return len(post(port, "/tokenize", {"content": text, "add_special": False})["tokens"])


class Text:
    # natural text for prompts: each prompt is a window of the file's words from a random offset no prompt started at
    def __init__(self, path):
        self.words = open(path, errors="replace").read().split()
        self.used = set()

    def words_from(self, rng):
        start = rng.randrange(len(self.words))
        while start in self.used:
            start = rng.randrange(len(self.words))
        self.used.add(start)
        return (self.words[(start + i) % len(self.words)] for i in itertools.count())


def make_prompt(port, rng, target, text=None):
    # random syllables, or a window of text (a Text); the syllables come from rng in the same order as before --text
    if text:
        source = text.words_from(rng)
    else:
        source = ("".join(rng.choice(SYL) for _ in range(rng.randint(1, 3))) for _ in itertools.count())
    words = []
    # grow in chunks, then trim to the target with a few tokenize calls
    while True:
        words += itertools.islice(source, max(16, target // 4))
        if n_tokens(port, " ".join(words)) >= target:
            break
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if n_tokens(port, " ".join(words[:mid])) <= target:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


class Req(threading.Thread):
    # one streamed chat request; messages (a whole conversation) replace the single user message with text; seed fixes
    # the sampling, which is random by default
    def __init__(self, port, text, gen, barrier, messages=None, seed=None):
        super().__init__(daemon=True)
        self.port, self.gen, self.barrier, self.seed = port, gen, barrier, seed
        self.messages = messages or [{"role": "user", "content": text}]
        self.t_send, self.times, self.server_ms, self.server_n, self.timings, self.error = None, [], [], [], None, None
        self.text = []

    def run(self):
        body = {"messages": self.messages, "stream": True, "max_tokens": self.gen, "ignore_eos": True, "timings_per_token": True}
        if self.seed is not None:
            body["seed"] = self.seed
        try:
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3600)
            self.barrier.wait()
            self.t_send = time.time()
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            if r.status != 200:
                self.error = f"HTTP {r.status}: {r.read()[:300]!r}"
                return
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                now = time.time()
                ev = json.loads(line[6:])
                tm = ev.get("timings")
                if tm:
                    self.timings = tm
                content = False
                for ch in ev.get("choices", []):
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        content = True
                        self.text.append(d.get("content") or "")
                if content:
                    self.times.append(now)
                    # the server attaches the timings to the chunk of the token they describe; a chunk carries several
                    # tokens when a character spans tokens (emoji, non-Latin scripts): the text waits until it is complete
                    self.server_ms.append(tm["predicted_ms"] if tm else None)
                    self.server_n.append(tm["predicted_n"] if tm else None)
            c.close()
        except Exception as e:
            self.error = repr(e)

    def raw(self):
        # per-token times: client arrival in s since t_send, server sampling time in ms since the first token
        return {"t_send": self.t_send, "client_s": [round(t - self.t_send, 5) for t in self.times],
                "server_ms": [None if x is None else round(x, 3) for x in self.server_ms], "server_n": self.server_n,
                "timings": self.timings, "error": self.error}


def tokens_by(r, t):
    # tokens generated for r up to its last chunk that arrived by time t: the server's count, or one per chunk without it
    n = 0
    for tt, k in zip(r.times, r.server_n):
        if tt > t:
            break
        n = k if k is not None else n + 1
    return n


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def server_gaps(r):
    # per token: a chunk of several tokens spreads its time over them, so a character split over tokens is not a slow step
    pts = list(zip(r.server_ms, r.server_n))
    out = []
    for (ma, na), (mb, nb) in zip(pts, pts[1:]):
        if None in (ma, mb, na, nb) or nb <= na:
            continue
        out += [(mb - ma) / (nb - na)] * (nb - na)
    return out


def req_stats(reqs):
    # first-token time, gaps between tokens on both clocks, generation time per response (first to last token)
    # itl_*: between chunks as the client receives them; server_itl_*: per generated token, on the server's clock
    ok = [r for r in reqs if r.times]
    ttft = [r.times[0] - r.t_send for r in ok]
    c_gap = [1e3 * (b - a) for r in ok for a, b in zip(r.times, r.times[1:])]
    s_gap = [g for r in ok for g in server_gaps(r)]
    merged = sum(r.timings["predicted_n"] - len(r.times) for r in ok if r.timings)
    gen_s = [r.timings["predicted_ms"] / 1e3 for r in ok if r.timings]
    # time before the slot started on the prompt (queue, HTTP, tokenizing) plus delivery of the first chunk
    wait_s = [r.times[0] - r.t_send - r.timings["prompt_ms"] / 1e3 for r in ok if r.timings]
    return {
        "ttft_p50": pct(ttft, 0.5), "ttft_p90": pct(ttft, 0.9), "ttft_max": max(ttft, default=None),
        "pre_prompt_s_p50": pct(wait_s, 0.5), "pre_prompt_s_max": max(wait_s, default=None),
        "itl_p50_ms": pct(c_gap, 0.5), "itl_p99_ms": pct(c_gap, 0.99), "itl_max_ms": max(c_gap, default=None),
        "server_itl_p50_ms": pct(s_gap, 0.5), "server_itl_p99_ms": pct(s_gap, 0.99), "server_itl_max_ms": max(s_gap, default=None),
        "gen_s_p50": pct(gen_s, 0.5), "gen_s_p90": pct(gen_s, 0.9),
        # tokens that reached the client inside another token's chunk (emoji, non-Latin characters spanning tokens)
        "merged_tokens_pct": 100 * merged / max(1, sum(r.timings["predicted_n"] for r in ok if r.timings)),
    }


def check_lengths(reqs_targets, slack=64):
    # (request, prompt target or None, output target): the chat template adds a few tokens to the calibrated prompt
    off_p, off_g = [], []
    for r, p, g in reqs_targets:
        if not r.timings:
            continue
        total = r.timings["cache_n"] + r.timings["prompt_n"]
        if p is not None and abs(total - p) > max(slack, p // 50):
            off_p.append((p, total))
        if r.timings["predicted_n"] != g:
            off_g.append((g, r.timings["predicted_n"]))
    if off_p or off_g:
        print(f"warning: {len(off_p)} requests with prompt tokens off target (target, actual) {off_p[:3]}, "
              f"{len(off_g)} with output tokens off target {off_g[:3]}", file=sys.stderr, flush=True)
    return {"prompt_off_target": len(off_p), "output_off_target": len(off_g)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--prompt-tokens", type=int, required=True)
    ap.add_argument("--gen-tokens", type=int, required=True)
    ap.add_argument("--waves", type=int, default=3)
    ap.add_argument("--pause", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--text", help="a text file: prompts are windows of it, not random syllables")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runinfo.pin_client()

    rng = random.Random(a.seed)
    text = Text(a.text) if a.text else None
    prompts = [[make_prompt(a.port, rng, a.prompt_tokens, text) for _ in range(a.n)] for _ in range(a.waves)]
    waves = []
    failed = False
    for w in range(a.waves):
        barrier = threading.Barrier(a.n)
        reqs = [Req(a.port, prompts[w][i], a.gen_tokens, barrier) for i in range(a.n)]
        for r in reqs:
            r.start()
        for r in reqs:
            r.join()
        errs = [r.error for r in reqs if r.error]
        ok = [r for r in reqs if r.times]
        if not ok:
            waves.append({"wave": w, "cold": w == 0, "errors": errs or ["no tokens received"]})
            failed = True
            break
        failed |= bool(errs)
        t0 = min(r.t_send for r in reqs if r.t_send)
        t1 = max(r.times[-1] for r in ok)
        t_all_first = max(r.times[0] for r in ok)
        n_decode = sum(r.timings["predicted_n"] - tokens_by(r, t_all_first) for r in ok if r.timings)
        waves.append({
            "wave": w, "cold": w == 0, "t0": t0, "t1": t1, "wall_s": t1 - t0, "errors": errs,
            **req_stats(reqs),
            "prompt_n": [r.timings["prompt_n"] for r in reqs if r.timings], "cache_n": [r.timings["cache_n"] for r in reqs if r.timings],
            "predicted_n": [r.timings["predicted_n"] for r in reqs if r.timings],
            "prefill_phase_s": t_all_first - t0, "decode_phase_s": t1 - t_all_first,
            "gen_tok_s_decode_phase": n_decode / max(1e-9, t1 - t_all_first),
            "lengths": check_lengths([(r, a.prompt_tokens, a.gen_tokens) for r in reqs]),
            "requests": [r.raw() for r in reqs],
        })
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in waves[-1].items()
                          if not isinstance(v, (list, dict)) or k == "errors"}), flush=True)
        if w + 1 < a.waves:
            time.sleep(a.pause)
    json.dump({"args": vars(a), "waves": waves}, open(a.out, "w"), indent=1)
    if failed:
        sys.exit("gen.py: some requests failed, see errors in " + a.out)


if __name__ == "__main__":
    main()
