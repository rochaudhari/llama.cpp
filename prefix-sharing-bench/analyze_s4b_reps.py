"""Donor test over repetitions: each file holds one set (alone, unrelated, shared) from one server run per config."""
import glob
import json
import statistics
import sys


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


sets = {}
for p in sorted(glob.glob(f"{sys.argv[1]}/*.jsonl")):
    for line in open(p):
        r = json.loads(line)
        sets.setdefault((r["model"], p), {})[r["with_others"] or "alone"] = r

for model in sorted({m for m, _ in sets}):
    rows = [s for (m, _), s in sorted(sets.items()) if m == model]
    print(f"\n## {model}: {len(rows)} sets")
    alone0 = rows[0]["alone"]["donor_tokens"]
    print("alone runs identical to each other:", all(s["alone"]["donor_tokens"] == alone0 for s in rows))
    print("| others | tokens identical to the solo run of the same set | mean change in log-prob before the first difference |")
    print("|---|---|---|")
    for others in ("unrelated", "shared"):
        same, dlp = [], []
        for s in rows:
            a, r = s["alone"], s[others]
            n = first_diff(r["donor_tokens"], a["donor_tokens"])
            same.append(n)
            d = [abs(x - y) for x, y in zip(r["donor_logprobs"][:n], a["donor_logprobs"][:n])] or [0.0]
            dlp.append(statistics.mean(d))
        print(f"| {others} | {', '.join(map(str, same))} | {', '.join(f'{x:.4f}' for x in dlp)} |")
