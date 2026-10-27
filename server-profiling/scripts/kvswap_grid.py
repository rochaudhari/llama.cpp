"""Run kvswap_sim.py over a grid of workloads and server settings, and rank the policies.

Each grid point runs every policy on the same requests (same seed). A policy's relative goodput at a point is its
goodput over the best policy's there: completed workflows per hour for agents, completed requests per hour for
generic loads. Failed requests, the worst pause between two tokens and the longest wait for a restore are summarized
next to it, since goodput alone hides them.
usage: kvswap_grid.py --out DIR [--grid agents,generic] [--seeds 2] [--jobs 28] [--policies ...]
       kvswap_grid.py --summarize DIR
"""
import argparse
import itertools
import json
import multiprocessing
import os
import sys
import time

import kvswap_sim as ks


def agents_grid():
    for pool, (sub, conc, slots), tool, attn, ram, mt in itertools.product(
            [32768, 65536, 131072], [(4, 1, 8), (6, 1, 8), (8, 2, 16)], [500, 1500, 4000],
            ["shared", "own"], [8, 32], ["4096", "16384", "pool"]):
        yield dict(workload="agents", pool=pool, subagents=sub, concurrent=conc, slots=slots, workflows=10,
                   tool_tokens=tool, attn=attn, cache_ram=ram, max_tokens=pool if mt == "pool" else int(mt))


def generic_grid():
    for pool, slots, prompt, gen, rate, attn, mt in itertools.product(
            [32768, 65536, 131072], [8, 16], [1000, 4000, 16000], [256, 1024, 4096], [0.25, 1.0],
            ["shared", "own"], ["4x", "pool"]):
        yield dict(workload="generic", pool=pool, slots=slots, prompt_tokens=prompt, gen_tokens=gen, rate=rate,
                   requests=200, attn=attn, cache_ram=8, max_tokens=pool if mt == "pool" else 4 * gen)


def argv_of(cfg, seed, wall):
    argv = ["--seed", str(seed), "--wall-limit", str(wall)]
    for k, v in cfg.items():
        argv += ["--" + k.replace("_", "-"), str(v)]
    return argv


def run_point(job):
    cfg, seed, policies, wall, extra = job
    a = ks.parse(argv_of(cfg, seed, wall) + extra)
    rows = []
    for p in policies:
        t0 = time.monotonic()
        try:
            r = ks.run_one(a, p)
        except TimeoutError as e:
            r = dict(policy=p, error=str(e))
        r.update(cfg, seed=seed, wall_s=time.monotonic() - t0)
        rows.append(r)
    return rows


def goodput(r):
    if "error" in r:
        return 0.0
    return r["goodput_wf_h"] if r["workload"] == "agents" else r["goodput_req_h"]


def point_key(r):
    return tuple(sorted((k, v) for k, v in r.items() if k in POINT_KEYS))


POINT_KEYS = ["workload", "pool", "subagents", "concurrent", "slots", "workflows", "tool_tokens", "attn", "cache_ram",
              "max_tokens", "prompt_tokens", "gen_tokens", "rate", "requests", "seed"]


def summarize(path, out=sys.stdout):
    rows = [json.loads(line) for line in open(os.path.join(path, "results.jsonl"))]
    points = {}
    for r in rows:
        points.setdefault(point_key(r), []).append(r)
    for rs in points.values():
        best = max(goodput(r) for r in rs) or 1e-9
        for r in rs:
            r["rel"] = goodput(r) / best
            r["loose"] = r["max_tokens"] == r["pool"]

    def table(title, sel):
        sel = [r for r in rows if sel(r)]
        if not sel:
            return
        n_points = len({point_key(r) for r in sel})
        print(f"\n{title} ({n_points} points)", file=out)
        print(f"{'policy':8} {'rel mean':>8} {'rel p10':>7} {'best':>5} {'w/ fails':>8} {'fails/pt':>8} "
              f"{'gap max p50':>11} {'gap max p90':>11} {'wait max p90':>12} {'errors':>6}", file=out)
        for p in dict.fromkeys(r["policy"] for r in sel):
            rs = [r for r in sel if r["policy"] == p]
            ok = [r for r in rs if "error" not in r]
            rel = sorted(r["rel"] for r in rs)
            print(f"{p:8} {sum(rel) / len(rel):8.2f} {ks.pct(rel, 10):7.2f} "
                  f"{sum(x >= 0.98 for x in rel) / len(rel):5.0%} "
                  f"{sum(r.get('failed', 0) > 0 for r in rs) / len(rs):8.0%} "
                  f"{sum(r.get('failed', 0) for r in ok) / max(1, len(ok)):8.1f} "
                  f"{ks.pct([r['gap_max_s'] for r in ok], 50):11.1f} {ks.pct([r['gap_max_s'] for r in ok], 90):11.1f} "
                  f"{ks.pct([r['wait_max_s'] for r in ok], 90):12.1f} {len(rs) - len(ok):6}", file=out)

    print("rel = goodput / best policy's at the same point; best = share of points within 2% of the best;", file=out)
    print("w/ fails = share of points where the policy errored at least one request; gap max = worst pause", file=out)
    print("between two tokens of a request (s); wait max = longest wait of a suspended request (s)", file=out)
    table("all points", lambda r: True)
    for wl in ("agents", "generic"):
        table(f"{wl}", lambda r, wl=wl: r["workload"] == wl)
        for attn in ("shared", "own"):
            table(f"{wl}, attention {attn}", lambda r, wl=wl, attn=attn: r["workload"] == wl and r["attn"] == attn)
        for loose in (False, True):
            name = "max_tokens = pool (loose)" if loose else "max_tokens tight"
            table(f"{wl}, {name}", lambda r, wl=wl, loose=loose: r["workload"] == wl and r["loose"] == loose)
    for ram in (8, 32):
        table(f"agents, cache-ram {ram} GiB", lambda r, ram=ram: r["workload"] == "agents" and r["cache_ram"] == ram)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--summarize")
    ap.add_argument("--grid", default="agents,generic")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--policies", default=",".join(ks.POLICIES))
    ap.add_argument("--wall-limit", type=float, default=600)
    ap.add_argument("--limit", type=int, default=0, help="only the first N grid points (a quick check)")
    ap.add_argument("--only", default="", help="keep grid points matching key=value,... (e.g. attn=shared,pool=65536)")
    ap.add_argument("--extra", default="", help="kvswap_sim.py flags added at every point, e.g. '--gather'")
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    os.makedirs(a.out, exist_ok=True)
    grids = {"agents": agents_grid, "generic": generic_grid}
    cfgs = [c for g in a.grid.split(",") for c in grids[g]()]
    for cond in filter(None, a.only.split(",")):
        k, v = cond.split("=")
        cfgs = [c for c in cfgs if str(c.get(k)) == v]
    if a.limit:
        cfgs = cfgs[:a.limit]
    extra = a.extra.split()
    jobs = [(c, s, a.policies.split(","), a.wall_limit, extra) for c in cfgs for s in range(1, a.seeds + 1)]
    with open(os.path.join(a.out, "grid.json"), "w") as f:
        json.dump(dict(args=vars(a), points=len(jobs)), f, indent=1)
    t0 = time.monotonic()
    with open(os.path.join(a.out, "results.jsonl"), "a") as f, multiprocessing.Pool(a.jobs) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_point, jobs), 1):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} points, {time.monotonic() - t0:.0f} s", file=sys.stderr)
    summarize(a.out)


if __name__ == "__main__":
    main()
