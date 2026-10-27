# What to do under unified-KV pressure: admission control vs suspension (simulation, 2026-10-26)

Question: when concurrent requests outgrow llama-server's shared `--kv-unified` pool, which policy should the server use? The KV-swap branch (`kv-swap-mvp`, f7d4450c6) suspends the largest active requests to the RAM prompt cache. This compares it with alternatives on the same loads.

## Method

- `scripts/kvswap_sim.py`: a step-level model of the server. It covers the unified pool, slots, prompt chunks up to `n_batch`, the RAM prompt cache (FIFO, byte limit, entry handed back on load) and the `--cache-idle-slots` pass at task launch. It also covers the branch's suspend/restore, with protected entries, one restore per step and a step for the restored request alone. `--selftest` checks invariants and the victim choice.
- Step times: `--model moe30b` (Qwen3-30B-A3B Q4_K_M) is fitted to llama-server measurements on the A40: decode steps at 1, 4 and 16 clients, and prompt processing at 4 x 4K and 4 x 16K with per-slot and shared KV. `--model dense27b` (Qwen3.6-27B Q4_K_M) is fitted to `llama-batched-bench -kvu` at 4K/16K and 1-16 sequences (commands in README.md). It reproduces all 7 points within ~1%.
- Copies:
  - saves: ~2 GB/s into fresh pages plus 6.5 us per copy call (pinbench);
  - restores: 12 GB/s plus 5.2 us per call (`scripts/restorebench.cu`: 12-18 GB/s pageable, 23-25 pinned).
  - There is one call per contiguous run of cells, per layer, for K and V. Generated tokens interleave, one run each.
- Grid (`scripts/kvswap_grid.py`): agents (workflows of 4-8 parallel subagents, multi-turn, ~3 s tool gaps, contexts growing to 10-90K) and generic (Poisson single requests).
  - Pools 32K-128K cells, 8-16 slots, shared or per-request attention, `--cache-ram` 8 or 32 GiB, `max_tokens` tight or unset.
  - 2 seeds: 1,512 points. Every policy at a point serves the same requests.
- Score: goodput, i.e. completed workflows per hour for agents (a workflow with any failed request does not count) and completed requests per hour for generic loads. `rel` is a policy's goodput over the best policy's at the same point. Failures, the worst pause between two tokens, and the longest wait for a restore are reported beside it.

Policies:
- `fail`: master.
- `largest`: the branch. An episode that cannot reserve RAM falls back to master's path, which errors everyone, suspended requests included.
- `bestfit`: the branch with the cheapest-to-copy victim.
- `bestfit-d`: the same, recomputing victims that don't fit in RAM.
- `tiered`: trim, idle slots, cost-based swap/recompute, fail one.
- `admit`: start a task only if every task's prompt + `max_tokens` fits.
- `hybrid`: start a task only if every task's prompt + `--out-margin` x the recent average output fits; tiered handles overflow.
- `hybrid-b`: hybrid admission, bestfit-d overflow.
- `newest`: vLLM-like, recompute the most recently admitted.

## Results

Relative goodput with the launch-order fix below, merged runs: `grid2-moe-fix` + `grid6-moe-m1`, `grid3-dense-fix` + `grid5-dense-m1`.

| policy | dense agents | dense generic | MoE agents | MoE generic | points with failed requests | worst token gap p90 |
|---|---|---|---|---|---|---|
| hybrid, margin 1.0 | 1.00 | 0.97 | 0.92 | 0.97 | 0% | 18-131 s |
| hybrid, margin 2.0 | 0.99 | 0.91 | 0.92 | 0.95 | 0% | 9-43 s |
| admit | 0.82 | 0.64 | 0.97 | 0.87 | 0% | 5-13 s |
| bestfit-d | 0.64 | 0.98 | 0.63 | 0.90 | 0% | 684-1,518 s |
| bestfit | 0.53 | 0.97 | 0.49 | 0.86 | 21-35% | 156-590 s |
| tiered | 0.58 | 0.91 | 0.56 | 0.81 | 0% | 1,856-2,836 s |
| largest (the branch) | 0.51 | 0.82 | 0.45 | 0.71 | 27-53% | 314-794 s |
| newest | 0.25 | 0.62 | 0.27 | 0.62 | 0% | 6,726-12,038 s |
| fail (master) | 0.17 | 0.49 | 0.16 | 0.54 | 70-91% | 4-10 s |

- Admission control is what matters. Swapping without it pays copies and minute-long stalls. With the default 8 GiB `--cache-ram`, the branch also errors requests whenever the snapshots don't fit.
- Under admission, overflows are rare (at most 6 in 300 requests even at 0.7x the average output), so the victim rule hardly matters: hybrid = hybrid-o = hybrid-b within 1%.
- A dense model gains much more from batching. Strict `max_tokens` reservations then cost a lot (admit 0.64 on dense generic), and margin 1.0 beats 2.0.
- Launch order. Master loads a returning task's cached prefix before the `--cache-idle-slots` pass clears idle slots, so under pressure the load fails and the conversation is prefilled again.
  - Clearing idle slots first must pin the entry being loaded and count its bytes as freed. Otherwise the idle saves evict it and things get worse.
  - With admission control the fix raises agent goodput 1.2-1.55x (grid1 vs grid2, grid4 vs grid3). Swap policies don't gain: their pool is full of active requests.
- grid1 (MoE, master's launch order, the first 7 policies) gave the same ranking.

## Gain over the branch, and what swapping adds once admission is in place

Admission (hybrid, margin 1.0, with the launch-order fix) against the branch as it is (largest, master's launch order). Only grid points where the branch had to suspend or errored (588-612 per model and workload); elsewhere every policy behaves the same.

| workload | goodput, points where the branch errored nothing | goodput, points where it errored some | requests completed there |
|---|---|---|---|
| MoE agents | 1.54x | 2.50x | 2.60x |
| dense agents | 1.70x | 2.23x | 2.45x |
| MoE generic | 1.04x | 1.56x | 2.78x |
| dense generic | 1.01x | 1.25x | 1.67x |

- The worst pause between two tokens has a median of 7-10 s with admission, against 78-301 s with the branch.
- The branch is ahead only on dense generic loads, by 6% or more at a tenth of the points where it errors nothing.

Under admission, overflows still happen at about a third of the points: 0.6-0.7 per agent run and about 3 per generic run on average (grid7/grid8 against grid5/grid6).
- With master's error path instead of preemption (hybrid-f), requests fail at 32-38% of points; where an overflow happens it errors 1% (agents) to 23% (generic) of requests. Preemption gives 1.10-1.17x (agents) and 1.03x (generic) goodput over that.
- Swapping (hybrid) and recompute alone (hybrid-r, no RAM snapshots) give the same goodput: median 1.00x, p10-p90 0.99-1.01x. Swapping shortens the worst pause by a few seconds at p90.
- So some preemption is needed, but the protected-snapshot machinery is optional once admission makes overflows rare.

## Check on real hardware

`ab.py` with returning.py (12 sessions, 4 clients, 6 turns, 3000/1500/300 tokens) on a master build (493253fe5), `-c 57344` and `-c 65536 --kv-unified`, 8 GiB cache. The commands are in README.md.

| pool | wall s (sim / real) | gen tok/s (sim / real) | failed cache loads (sim / real) | errors |
|---|---|---|---|---|
| 57344 | 292 / 305 | 74.0 / 70.8 | 3 / 4 | 0 / 0 |
| 65536 | 283 / 296 | 76.4 / 73.0 | 0 / 0 | 0 / 0 |

The simulator under-predicts the re-prefill tokens of returning turns by ~25%: about 140K vs 190-200K, mostly from RAM-cache evictions.

## Limits

- Simulation calibrated on one A40 with two models. The agent loads are synthetic.
- Not modelled: prefix sharing between slots; hybrid models' recurrent state in cache saves and checkpoints; CUDA graph churn.
- The kv-swap branch itself was not run on hardware: its behaviour is modelled from its code.

## Data

No results are committed. Every grid is seeded, so the commands below regenerate them, about 15 minutes per full grid with 28 jobs. Each run writes `<grid>/results.jsonl` (one row per point and policy), and `kvswap_grid.py --summarize <grid>` prints its ranking tables. The grids behind the tables above:

| grid | model | launch order | policies |
|---|---|---|---|
| grid1 | moe30b | master | fail, largest, bestfit, tiered, hybrid, newest, admit |
| grid2-moe-fix | moe30b | fixed (`--idle-before-load`) | the same, plus hybrid-o |
| grid3-dense-fix | dense27b | fixed | the same, plus hybrid-o |
| grid4-dense-master | dense27b | master | the same, plus hybrid-o |
| grid5-dense-m1, grid6-moe-m1 | dense27b, moe30b | fixed | hybrid, hybrid-b, bestfit-d with `--out-margin 1.0` |
| grid7-dense-m1-overflow, grid8-moe-m1-overflow | dense27b, moe30b | fixed | hybrid-f, hybrid-r with `--out-margin 1.0` |

## Reproduce

```
python3 scripts/kvswap_sim.py --selftest
python3 scripts/kvswap_grid.py --out data/kvswap-sim/grid2-moe-fix --extra "--idle-before-load --model moe30b"
python3 scripts/kvswap_grid.py --out data/kvswap-sim/grid5-dense-m1 --policies hybrid,hybrid-b,bestfit-d \
    --extra "--idle-before-load --model dense27b --out-margin 1.0"
python3 scripts/kvswap_grid.py --summarize data/kvswap-sim/grid2-moe-fix
```
