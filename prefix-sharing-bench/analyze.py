"""Summarize benchmark results: mean (min-max) over repetitions."""
import collections
import glob
import json
import statistics
import sys

d = sys.argv[1]
R = collections.defaultdict(list)
for p in glob.glob(f"{d}/*.jsonl"):
    for line in open(p):
        r = json.loads(line)
        R[(r["model"], r["scenario"])].append(r)


def ms(xs, fmt="{:.2f}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return fmt.format(xs[0])
    return (fmt + " ({}-{})").format(statistics.mean(xs), fmt.format(min(xs)), fmt.format(max(xs)))


def kv_rows(rs, kv, key, label):
    """Rows of off vs attach for each value of key: TTFT of sessions 2-3, decode, KV cells and KV saved."""
    for v in sorted({key(r) for r in rs}):
        base = None
        for cfg in ("off", "attach"):
            x = [r for r in rs if r["config"] == cfg and key(r) == v]
            k = [r for r in kv if r["config"] == cfg and key(r) == v]
            cells = k[0]["kv_cells_used_max"] if k else None
            kib = k[0]["kib_per_cell"] if k else None
            saved = ""
            if cfg == "off":
                base = cells
            elif base and cells:
                saved = f"{1 - cells / base:.0%} ({(base - cells) * kib / 2**20:.2f} GiB)" if kib else f"{1 - cells / base:.0%}"
            print(f"| {label(v)} | {cfg} | {ms([r['ttft_s'][1] for r in x])} | {ms([r['ttft_s'][2] for r in x])} | "
                  f"{ms([r['decode3_tps'] for r in x], '{:.0f}')} | {cells} | {saved} |")


for model in sorted({k[0] for k in R}):
    print(f"\n## {model}")

    rs = R[(model, "s1")]
    if rs:
        print("\n### S1: concurrent sessions differing by session ID (TTFT s; shared tokens; 3-session decode tok/s)")
        print("| config | session 1 | session 2 | session 3 | shared tokens (2nd, 3rd) | decode, 3 sessions |\n|---|---|---|---|---|---|")
        for cfg in ("off", "attach"):
            x = [r for r in rs if r["config"] == cfg]
            print(f"| {cfg} | {ms([r['ttft_s'][0] for r in x])} | {ms([r['ttft_s'][1] for r in x])} | {ms([r['ttft_s'][2] for r in x])} | "
                  f"{x[0]['attached'] if x else ''} | {ms([r['decode3_tps'] for r in x], '{:.0f}')} |")

    rs = R[(model, "s2")]
    if rs:
        print("\n### S2: sequential sessions on 3 days, slots cleared between (TTFT s; tokens restored from RAM)")
        print("| config | day 1 | day 2 | day 3 | restored (day 2, day 3) |\n|---|---|---|---|---|")
        for cfg in ("off", "prefix"):
            x = [r for r in rs if r["config"] == cfg]
            print(f"| {cfg} | {ms([r['ttft_s'][0] for r in x])} | {ms([r['ttft_s'][1] for r in x])} | {ms([r['ttft_s'][2] for r in x])} | "
                  f"{x[0]['prefix_restored'] if x else ''} |")

    rs = [r for r in R[(model, "s3")] if not r["kv_debug"]]
    kv = [r for r in R[(model, "s3")] if r["kv_debug"]]
    if rs:
        print("\n### S3: 3 sessions, same prefix, private history H each (TTFT of sessions 2-3 s; 3-session decode tok/s; KV cells)")
        print("| H | config | TTFT session 2 | TTFT session 3 | decode, 3 sessions | KV cells used | KV saved |\n|---|---|---|---|---|---|---|")
        kv_rows(rs, kv, lambda r: r["history"], str)

    if R[(model, "s5")]:
        # the prefix without extra context is s3 without history
        runs = [r for r in R[(model, "s3")] if r["history"] == 0] + R[(model, "s5")]
        rs = [r for r in runs if not r["kv_debug"]]
        kv = [r for r in runs if r["kv_debug"]]
        extra = lambda r: r.get("extra_tokens", 0)  # noqa: E731
        prefix = {extra(r): r["prefix_saved"][0] for r in rs if r["prefix_saved"]}
        print("\n### S5: 3 sessions sharing one prefix of P tokens, no history (TTFT of sessions 2-3 s; 3-session decode tok/s; KV cells)")
        print("| P | config | TTFT session 2 | TTFT session 3 | decode, 3 sessions | KV cells used | KV saved |\n|---|---|---|---|---|---|---|")
        kv_rows(rs, kv, extra, lambda v: prefix.get(v))

    rs = R[(model, "s4a")]
    if rs:
        by = {r["path"]: r["results"] for r in rs}

        def cmp(p, q):
            agree = tot = 0
            d = []
            for a_, b_ in zip(by[p], by[q]):
                for x, y in zip(a_["rows"], b_["rows"]):
                    tot += 1
                    agree += x["top1"] == y["top1"]
                    if x["lp_ref"] is not None and y["lp_ref"] is not None:
                        d.append(abs(x["lp_ref"] - y["lp_ref"]))
            return agree, tot, max(d), statistics.mean(d)

        print("\n### S4a: same continuation scored at every position (top-1 agreement; |delta log-prob of the reference token|)")
        print("| comparison | top-1 agree | max | mean |\n|---|---|---|---|")
        for p, q in (("reuse", "cold"), ("attach", "cold"), ("attach", "reuse")):
            if p in by and q in by:
                agree, tot, mx, mn = cmp(p, q)
                print(f"| {p} vs {q} | {agree}/{tot} | {mx:.3f} | {mn:.4f} |")

    rs = R[(model, "s4b")]
    if rs:
        alone = next(r for r in rs if not r["with_others"])
        print("\n### S4b: donor output with other sessions running (vs donor alone)")
        print("| others | tokens identical | max / mean delta log-prob | attaches | cache reuse refused |\n|---|---|---|---|---|")
        for r in rs:
            same = next((i for i, (x, y) in enumerate(zip(r["donor_tokens"], alone["donor_tokens"])) if x != y),
                        min(len(r["donor_tokens"]), len(alone["donor_tokens"])))
            dd = [abs(x - y) for x, y in zip(r["donor_logprobs"][:same], alone["donor_logprobs"][:same])] or [0.0]
            print(f"| {r['with_others'] or 'none (alone)'} | {same}/{len(r['donor_tokens'])} | {max(dd):.3f} / {statistics.mean(dd):.4f} | "
                  f"{len(r['attached'])} | {r['cache_reuse_refused']} |")
