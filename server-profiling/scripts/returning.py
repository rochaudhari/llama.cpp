"""Multi-turn sessions that come back: more sessions than slots, fewer clients than slots.

Each session starts with a unique prompt of P tokens. Every later turn sends the whole conversation so far (user and
assistant messages) plus a new user message of U tokens, so the previous state can be reused from the slot or the
RAM prompt cache. Each client owns sessions/clients sessions and takes turns between them.
The replies are sampled with a fixed seed per session and turn. Batching still changes a token in most replies (the
logits differ slightly with the batch), so a later turn's prompt usually differs between builds: the same build run
twice matched 2 to 33 of 72 turns. Each result keeps a hash of its reply, and ab.py counts the turns whose prompt
matched the first variant's; the reused and recomputed token totals stay within 0.5%.
Reports, for the returning turns, prompt tokens reused (timings.cache_n) vs recomputed (timings.prompt_n), first-token
time and wall time; token timing on both clocks as in gen.py.
usage: returning.py --port 8091 --sessions 32 --clients 8 --turns 4 --prompt-tokens 4096 --gen-tokens 128 --out res.json
"""
import argparse
import hashlib
import json
import random
import sys
import threading
import time

import runinfo
from gen import Req, Text, make_prompt, pct, req_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--sessions", type=int, default=32)
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--user-tokens", type=int, default=64)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--text", help="a text file: prompts are windows of it, not random syllables")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runinfo.pin_client()

    rng = random.Random(a.seed)
    text = Text(a.text) if a.text else None
    first = [make_prompt(a.port, rng, a.prompt_tokens, text) for _ in range(a.sessions)]
    follow = [[make_prompt(a.port, rng, a.user_tokens, text) for _ in range(a.turns - 1)] for _ in range(a.sessions)]
    history = [[{"role": "user", "content": first[s]}] for s in range(a.sessions)]
    results = []
    lock = threading.Lock()

    def client(c):
        own = list(range(c, a.sessions, a.clients))
        for turn in range(a.turns):
            for s in own:
                r = Req(a.port, None, a.gen_tokens, threading.Barrier(1), messages=list(history[s]), seed=a.seed * 1000000 + s * 1000 + turn)
                r.run()
                reply = "".join(r.text)
                r.reply = hashlib.sha1(reply.encode()).hexdigest()[:12]
                history[s].append({"role": "assistant", "content": reply})
                if turn + 1 < a.turns:
                    history[s].append({"role": "user", "content": follow[s][turn]})
                r.session, r.turn = s, turn
                with lock:
                    results.append(r)

    t0 = time.time()
    threads = [threading.Thread(target=client, args=(c,), daemon=True) for c in range(a.clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    ret = [r for r in results if r.turn > 0 and r.timings]
    ttft = [r.times[0] - r.t_send for r in ret if r.times]
    errors = [r.error for r in results if r.error]
    out = {
        "args": vars(a), "wall_s": wall, "requests": len(results),
        "returning": len(ret),
        "returning_prompt_tokens_reused": sum(r.timings["cache_n"] for r in ret),
        "returning_prompt_tokens_recomputed": sum(r.timings["prompt_n"] for r in ret),
        "returning_full_recompute": sum(1 for r in ret if r.timings["cache_n"] < a.prompt_tokens // 2),
        "returning_ttft_p50": pct(ttft, 0.5), "returning_ttft_p90": pct(ttft, 0.9), "returning_ttft_max": max(ttft, default=None),
        "first_ttft_p50": pct([r.times[0] - r.t_send for r in results if r.turn == 0 and r.times], 0.5),
        "gen_tok_s": sum(r.timings["predicted_n"] for r in results if r.timings) / wall,
        **{k: v for k, v in req_stats(results).items() if not k.startswith("ttft")},
        "errors": errors[:3], "n_errors": len(errors),
        "results": [{"session": r.session, "turn": r.turn, "reply": r.reply, "ttft": r.times[0] - r.t_send if r.times else None, **r.raw()}
                    for r in results],
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k not in ("results", "args")}), flush=True)
    if errors:
        sys.exit(f"returning.py: {len(errors)} requests failed, see errors in {a.out}")


if __name__ == "__main__":
    main()
