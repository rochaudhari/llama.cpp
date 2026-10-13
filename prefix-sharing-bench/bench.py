"""Prefix-sharing benchmarks with Hermes-like prompts and Hermes's managed-server flags.

  s1   concurrent sessions that differ only by session ID (end of the system prompt)      configs: off, attach
  s2   sequential sessions on different days (date line differs), slots cleared between  configs: off, prefix
  s3   3 concurrent sessions, same prefix, private history of H tokens each               configs: off, attach
  s4a  correctness: one fixed continuation scored at every position, cold vs reuse vs attach
  s4b  correctness: donor generates while other sessions attach and leave, vs donor alone
  s5   like s3 without history, with the shared prefix grown by N tokens of project context configs: off, attach

Usage: bench.py <scenario> --model <gguf> --out <jsonl> [--reps N] [--history H ...] [--extra N ...] [--kv-debug]
"""

import argparse
import http.client
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HOST, PORT = "127.0.0.1", 8091
HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = "/localhome/local-rochaudhari/roshan/llama.cpp/build/bin/llama-server"

# Hermes's managed server (hermes_cli/local_runtime/context_policy.py): q8_0 KV, 2048 batch, default slots (4, unified)
HERMES_FLAGS = "-ngl 99 -fa on -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0"
CONFIGS = {"off": "", "prefix": "--cache-prefix", "attach": "--cache-prefix --slot-attach"}

P = json.load(open(os.path.join(HERE, "hermes_prompts.json")))
WORDS = "the a session server cache tool agent model token prompt result file test build user note value".split()


def system_prompt(session_id="0000-0000", day=0, memory=0, extra=""):
    date = ["Monday, October 12, 2026", "Tuesday, October 13, 2026", "Wednesday, October 14, 2026"][day]
    volatile = P["volatile_template"].format(skills=P["skills"], memory=P["memory"][memory], date=date, session_id=session_id)
    context = f"{P['context']}\n{extra}" if extra else P["context"]
    return f"{P['stable']}\n\n{context}\n\n{volatile}"


def padding(n_tokens, seed=7):
    """More project context of about n_tokens tokens, counted with the running server's tokenizer."""
    rng = random.Random(seed)

    def line():
        return "- " + " ".join(rng.choice(WORDS) for _ in range(18)) + "."

    sample = [line() for _ in range(200)]
    per_line = len(post_json("/tokenize", {"content": "\n".join(sample)})["tokens"]) / len(sample)
    n_lines = round(n_tokens / per_line)
    lines = sample + [line() for _ in range(max(0, n_lines - len(sample)))]
    return "\n".join(lines[:n_lines])


def history(n_tokens, seed):
    """Unique private conversation of about n_tokens tokens (about 1.3 tokens per word here)."""
    rng, msgs, n = random.Random(seed), [], 0
    while n < n_tokens:
        k = 300
        msgs.append({"role": "user", "content": " ".join(rng.choice(WORDS) for _ in range(k))})
        msgs.append({"role": "assistant", "content": " ".join(rng.choice(WORDS) for _ in range(k))})
        n += int(2 * k * 1.3)
    return msgs


def post_json(path, body, timeout=600):
    req = urllib.request.Request(f"http://{HOST}:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_json(path):
    with urllib.request.urlopen(f"http://{HOST}:{PORT}{path}", timeout=30) as r:
        return json.loads(r.read())


def chat_body(system, messages, max_tokens, stream=True, slot=-1, **extra):
    body = {"messages": [{"role": "system", "content": system}] + messages, "tools": P["tools"],
            "stream": stream, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True, "cache_prompt": True,
            "id_slot": slot, "chat_template_kwargs": {"enable_thinking": False}}
    body.update(extra)
    return body


class Stream(threading.Thread):
    """Streams one chat request; records the arrival time, text and log-prob of every token."""

    def __init__(self, body):
        super().__init__(daemon=True)
        self.body, self.tok_times, self.pieces, self.logprobs = body, [], [], []
        self.t_send, self.stop, self.conn, self.error = None, False, None, None

    def run(self):
        try:
            self.conn = http.client.HTTPConnection(HOST, PORT, timeout=1800)
            self.t_send = time.perf_counter()
            self.conn.request("POST", "/v1/chat/completions", json.dumps(self.body), {"Content-Type": "application/json"})
            resp = self.conn.getresponse()
            while not self.stop:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {})
                    piece = d.get("content") or d.get("reasoning_content") or (json.dumps(d["tool_calls"]) if d.get("tool_calls") else None)
                    if piece:
                        self.tok_times.append(time.perf_counter())
                        self.pieces.append(piece)
                        lp = (ch.get("logprobs") or {}).get("content") or []
                        self.logprobs.extend(x.get("logprob") for x in lp)
        except Exception as e:  # cancelled streams end with a reset
            if not self.stop:
                self.error = repr(e)
        finally:
            if self.conn:
                self.conn.close()

    def wait_tokens(self, n):
        while len(self.tok_times) < n and self.is_alive():
            time.sleep(0.02)
        if len(self.tok_times) < n:
            raise RuntimeError(f"stream ended early: {self.error}")
        return self.tok_times[0] - self.t_send

    def cancel(self):
        self.stop = True
        try:
            self.conn.sock.shutdown(2)
        except Exception:
            pass


def start_server(model, flags, log_path, ctx, kv_debug=False):
    slots_dir = log_path + ".slots"
    os.makedirs(slots_dir, exist_ok=True)
    cmd = [SERVER, "-m", model, "-c", str(ctx), "--host", HOST, "--port", str(PORT), "--slot-save-path", slots_dir]
    cmd += HERMES_FLAGS.split() + flags.split() + (["-v"] if kv_debug else [])
    env = dict(os.environ, **({"LLAMA_KV_CACHE_DEBUG": "1"} if kv_debug else {}))
    srv = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env)
    for _ in range(900):
        if srv.poll() is not None:
            raise RuntimeError(f"server exited with {srv.returncode}, see {log_path}")
        try:
            if get_json("/health").get("status") == "ok":
                return srv
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not start")


def stop_server(srv):
    srv.terminate()
    srv.wait(timeout=60)


def rate(times, t0, t1):
    return sum(1 for t in times if t0 <= t < t1) / (t1 - t0)


def events(log_path):
    t = open(log_path, errors="replace").read()
    kib = re.search(r"llama_kv_cache: size =\s*([\d.]+) MiB \(\s*(\d+) cells", t)
    return {"attached": [int(x) for x in re.findall(r"attached (\d+) prompt tokens from slot", t)],
            "prefix_saved": [int(x) for x in re.findall(r"saved prompt prefix to the prompt cache, n_tokens = (\d+)", t)],
            "prefix_restored": [int(x) for x in re.findall(r"restored prompt prefix from the prompt cache, n_tokens = \d+, n_reuse = (\d+)", t)],
            "kib_per_cell": float(kib.group(1)) * 1024 / int(kib.group(2)) if kib else None,
            "cache_reuse_refused": len(re.findall(r"cache reuse is not supported - ignoring", t))}


def kv_max(log_path):
    used = [(int(n), int(u)) for n, u in re.findall(r"find_slot: stream\[0\], n = +(\d+), used = +(\d+)", open(log_path, errors="replace").read())]
    return {"kv_cells_used_max": max(u for _, u in used) if used else None, "kv_scan_width_max": max(n for n, _ in used) if used else None}


def task_msg(i, tag):
    # plain prose, so that tokens stream at once (a tool call is held back by the server until its name is parsed)
    return [{"role": "user", "content": f"[{tag}-{i}] Do not call any tools. Reply in plain prose with a long, detailed story about a team that maintains a server."}]


# ---- scenarios -------------------------------------------------------------------------------------------------------

def s1(a, cfg, log_path):
    """Concurrent: A, then B once A decodes, then C once B decodes; sessions differ only by session ID."""
    srv = start_server(a.model, CONFIGS[cfg], log_path, a.ctx)
    try:
        st = [Stream(chat_body(system_prompt(session_id=f"{i}{i}{i}{i}-sess"), task_msg(i, "s1"), 4000)) for i in range(3)]
        ttft = []
        for s in st:
            s.start()
            ttft.append(s.wait_tokens(16 if s is not st[-1] else 1))
        t0 = st[-1].tok_times[0]
        time.sleep(a.window)
        dec3 = sum(rate(s.tok_times, t0, t0 + a.window) for s in st)
        for s in st:
            s.cancel()
        return {"ttft_s": ttft, "decode3_tps": dec3}
    finally:
        stop_server(srv)


def s2(a, cfg, log_path):
    """Sequential: day 1, day 2, day 3; all slots erased after each, so only the RAM cache can help."""
    srv = start_server(a.model, CONFIGS[cfg], log_path, a.ctx)
    try:
        ttft = []
        for day in range(3):
            s = Stream(chat_body(system_prompt(day=day), task_msg(day, "s2"), 8))
            s.start()
            s.join()
            ttft.append(s.tok_times[0] - s.t_send)
            for sid in range(4):
                post_json(f"/slots/{sid}?action=erase", {})
        return {"ttft_s": ttft}
    finally:
        stop_server(srv)


def s3(a, cfg, log_path, hist, extra_tokens=0):
    """3 concurrent sessions with the same prefix and hist tokens of private history each.

    extra_tokens grows the shared prefix with more project context (s5).
    """
    srv = start_server(a.model, CONFIGS[cfg], log_path, a.ctx, a.kv_debug)
    try:
        system = system_prompt(extra=padding(extra_tokens) if extra_tokens else "")
        st = [Stream(chat_body(system, history(hist, seed=i) + task_msg(i, "s3"), 4000)) for i in range(3)]
        ttft = []
        for s in st:
            s.start()
            ttft.append(s.wait_tokens(16 if s is not st[-1] else 1))
        t0 = st[-1].tok_times[0]
        time.sleep(a.window)
        dec3 = sum(rate(s.tok_times, t0, t0 + a.window) for s in st)
        for s in st:
            s.cancel()
        return {"ttft_s": ttft, "decode3_tps": dec3, "history": hist, "extra_tokens": extra_tokens}
    finally:
        stop_server(srv)


def completion_probs(prompt, slot, n_probs=10):
    r = post_json("/completion", {"prompt": prompt, "n_predict": 1, "n_probs": n_probs, "temperature": 0.0,
                                  "cache_prompt": True, "id_slot": slot, "return_tokens": True})
    p = r["completion_probabilities"][0]
    return {t["id"]: t["logprob"] for t in p["top_logprobs"]}, r["tokens"][0]


def s4a(a, path, log_path):
    """Teacher-forced: the same reference continuation is scored at every position through one path."""
    flags = "--cache-prefix --slot-attach --no-cache-idle-slots"
    srv = start_server(a.model, flags, log_path, a.ctx)
    try:
        out = []
        for k in range(a.prompts):
            system = system_prompt(session_id=f"corr-{k}")
            user = [{"role": "user", "content": f"Question {k}: explain how a prompt cache saves work, in detail."}]
            tmpl = post_json("/apply-template", {"messages": [{"role": "system", "content": system}] + user,
                                                 "tools": P["tools"], "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
            ref = json.load(open(a.ref))[str(k)] if path != "cold" else None
            slot = 1
            if path in ("reuse", "attach"):
                post_json("/v1/chat/completions", chat_body(system, [{"role": "user", "content": "[donor] ok"}], 1, stream=False, slot=0))
                slot = 0 if path == "reuse" else 1
            if ref is None:  # the cold path makes the reference continuation
                r = post_json("/completion", {"prompt": tmpl, "n_predict": a.n_tokens, "temperature": 0.0, "cache_prompt": True,
                                              "id_slot": slot, "return_tokens": True, "ignore_eos": True})
                ref = r["tokens"]
            rows = []
            for i in range(len(ref)):
                probs, top1 = completion_probs([tmpl] + ref[:i], slot)
                rows.append({"top1": top1, "ref": ref[i], "lp_ref": probs.get(ref[i])})
            out.append({"prompt": k, "ref": ref, "rows": rows})
            for sid in range(4):
                post_json(f"/slots/{sid}?action=erase", {})
        return {"path": path, "results": out}
    finally:
        stop_server(srv)


def s4b(a, cfg, log_path, with_others, t_wait):
    """The donor generates n tokens; optionally others attach to it, leave mid-generation, and one asks for cache reuse.

    The server rejects logprobs for streamed requests with tools, so the donor is not streamed: the others start
    t_wait seconds after the donor, which is just after its prefill (measured in the run without others).
    """
    srv = start_server(a.model, CONFIGS[cfg], log_path, a.ctx)
    try:
        system = system_prompt(session_id="donor-0000")
        res = {}

        def donor():
            res["r"] = post_json("/v1/chat/completions", chat_body(system, task_msg(0, "donor"), a.n_tokens, stream=False,
                                                                   logprobs=True, top_logprobs=1))

        th = threading.Thread(target=donor)
        th.start()
        others = []
        if with_others:
            time.sleep(t_wait)
            # "shared": the others share the donor prefix and attach to it; "unrelated": short prompts that share
            # nothing but also start decoding next to the donor at once, so the batches look the same in both runs
            if with_others == "shared":
                bodies = [chat_body(system, task_msg(i, "other"), n) for i, n in ((1, 30), (2, 2000), (3, 20))]
            else:
                bodies = [{"messages": [{"role": "user", "content": f"[{i}] Write a long story about servers."}], "stream": True,
                           "max_tokens": n, "temperature": 0.0, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
                          for i, n in ((1, 30), (2, 2000), (3, 20))]
            bodies[2]["n_cache_reuse"] = 256
            others = [Stream(b) for b in bodies]
            for s in others:
                s.start()
            others[1].wait_tokens(10)
            others[1].cancel()  # leaves mid-generation
        th.join()
        for s in others:
            s.cancel()
        r = res["r"]
        lp = r["choices"][0]["logprobs"]["content"]
        return {"donor_tokens": [x["token"] for x in lp], "donor_logprobs": [x["logprob"] for x in lp],
                "prompt_ms": r["timings"]["prompt_ms"], "with_others": with_others}
    finally:
        stop_server(srv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["s1", "s2", "s3", "s4a", "s4b", "s5"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=196608)
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--history", type=int, nargs="+", default=[0, 16384, 32768])
    ap.add_argument("--extra", type=int, nargs="+", default=[10000, 18000], help="s5: tokens of project context added to the prefix")
    ap.add_argument("--kv-debug", action="store_true")
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--n-tokens", type=int, default=128)
    ap.add_argument("--ref", default=None, help="s4a: reference continuations from the cold path (json)")
    a = ap.parse_args()

    name = os.path.basename(a.model).removesuffix(".gguf")
    t_wait = 0.0
    jobs = []
    if a.scenario in ("s1", "s3", "s5"):
        cfgs = ["off", "attach"]
    elif a.scenario == "s2":
        cfgs = ["off", "prefix"]
    if a.scenario in ("s1", "s2"):
        jobs = [(cfg, rep, None) for rep in range(a.reps) for cfg in cfgs]
    elif a.scenario == "s3":
        jobs = [(cfg, rep, h) for h in a.history for rep in range(a.reps) for cfg in cfgs]
    elif a.scenario == "s5":
        jobs = [(cfg, rep, n) for n in a.extra for rep in range(a.reps) for cfg in cfgs]
    elif a.scenario == "s4a":
        jobs = [(path, 0, None) for path in ("cold", "reuse", "attach")]
    elif a.scenario == "s4b":
        jobs = [("attach", 0, None), ("attach", 0, "unrelated"), ("attach", 0, "shared")]

    for cfg, rep, extra in jobs:
        log_path = f"{a.out}.{a.scenario}.{cfg}.{rep}.{extra}.server.log"
        if a.scenario == "s1":
            r = s1(a, cfg, log_path)
        elif a.scenario == "s2":
            r = s2(a, cfg, log_path)
        elif a.scenario == "s3":
            r = s3(a, cfg, log_path, extra)
        elif a.scenario == "s5":
            r = s3(a, cfg, log_path, 0, extra)
        elif a.scenario == "s4a":
            if cfg == "cold":
                r = s4a(a, cfg, log_path)
                a.ref = a.out + ".ref.json"
                json.dump({str(x["prompt"]): x["ref"] for x in r["results"]}, open(a.ref, "w"))
            else:
                r = s4a(a, cfg, log_path)
        else:
            r = s4b(a, cfg, log_path, extra, t_wait)
            if not extra:
                t_wait = r["prompt_ms"] / 1000.0 + 0.5
        r.update({"model": name, "scenario": a.scenario, "config": cfg, "rep": rep, "kv_debug": a.kv_debug}, **events(log_path))
        if a.kv_debug:
            r.update(kv_max(log_path))
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps({k: v for k, v in r.items() if k not in ("results", "donor_tokens", "donor_logprobs")}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
