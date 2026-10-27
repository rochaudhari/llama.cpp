# KV swapping: policy simulation and benchmarks

What should llama-server do when concurrent requests outgrow its unified KV pool: suspend requests to RAM, as the kv-swap branch does, or admit only what fits? `KVSWAP.md` has the method, the results and the recommendation. This file says how to run everything. Commands run from this directory.

- `scripts/kvswap_sim.py`: the simulator. It models the server, the policies and the workloads. Python 3, standard library, no GPU.
- `scripts/kvswap_grid.py`: runs the simulator over a grid of settings and ranks the policies.
- `scripts/pinbench.cu`, `scripts/restorebench.cu`: what saving a slot's KV state to host memory and restoring it cost, the two copies a suspension makes.
- `scripts/returning.py`, with `gen.py` and `runinfo.py`: the multi-turn client used to check the simulator against a real server.

## Simulator

```sh
python3 scripts/kvswap_sim.py --selftest
# one setting, every policy, as a table
python3 scripts/kvswap_sim.py --workload agents --pool 65536 --slots 8 --idle-before-load
python3 scripts/kvswap_sim.py --model dense27b --workload generic --rate 1 --prompt-tokens 4000 --gen-tokens 1024 \
    --policies largest,hybrid,admit
```

- Workloads:
  - `agents`: workflows of parallel multi-turn subagents with tool-call gaps;
  - `generic`: Poisson single requests;
  - `returning`: the shape of `returning.py`.
- Policies:
  - `fail`: master;
  - `largest`: the kv-swap branch;
  - `bestfit`, `bestfit-d`, `tiered`: other victim choices;
  - `admit`, `hybrid`, `hybrid-b`, `hybrid-o`, `hybrid-f`, `hybrid-r`: admission control with different overflow paths;
  - `newest`: vLLM-like recompute.

  KVSWAP.md describes each.
- Main options:
  - `--model moe30b|dense27b`: step-time calibration;
  - `--attn shared|own`: attention over all used cells or only the request's own;
  - `--cache-ram GiB`;
  - `--idle-before-load`: the launch-order fix;
  - `--gather`: one copy per layer however scattered;
  - `--out-margin`: hybrid's reservation as a multiple of the average output;
  - `--policies`.

  `--help` lists all.

A grid runs every policy on the same requests at each point. It ranks them by goodput relative to the best policy at that point, with failures and worst pauses beside it:

```sh
python3 scripts/kvswap_grid.py --out data/kvswap-sim/mygrid --extra "--idle-before-load --model moe30b"
python3 scripts/kvswap_grid.py --summarize data/kvswap-sim/mygrid
```

- A full grid takes about 15 minutes with 28 jobs (`--jobs`).
- `--only key=value,...` runs part of it, e.g. `--only workload=agents,pool=65536`.
- KVSWAP.md lists the grids behind its tables and the commands to rerun them. All grids are seeded.

## Copy costs

```sh
nvcc -O2 -o pinbench scripts/pinbench.cu
./pinbench 144 96        # save, contiguous: 96 copies (48 layers x K, V) of one 144 MiB state
./pinbench 144 98500     # save, one copy per ~1.5 KiB run, as for a state whose cells are scattered
nvcc -O2 -o restorebench scripts/restorebench.cu
./restorebench 1560 96 1 # restore: contiguous, and one copy per 1 KiB run, from pageable and pinned buffers
```

The simulator's copy costs come from these runs:
- `--save-gbps`: bytes over the contiguous save time, about 2 GB/s on the A40, page-fault bound.
- `--copy-us`: the per-copy cost of the scattered save, about 6.5 us.
- `--restore-gbps` and `--restore-copy-us`: from `restorebench`, 12-18 GB/s and about 5.2 us.

## Step times of another model

The `dense27b` preset in `MODELS` (`kvswap_sim.py`) comes from two `llama-batched-bench` runs with unified KV:

```sh
llama-batched-bench -m MODEL.gguf -c 139264 -b 2048 -ub 512 -ngl 99 -fa on -kvu -npp 4096 -ntg 64 -npl 1,4,8,16
llama-batched-bench -m MODEL.gguf -c 139264 -b 2048 -ub 512 -ngl 99 -fa on -kvu -npp 16384 -ntg 64 -npl 1,4,8
```

To add a preset for another model:
- **Decode:** a step is B / S_TG. Fit it as a value per batch size (`dec`) plus a cost per used cell (`dec_att`); comparing 4K and 16K at the same B gives that slope.
- **Prompt:** time per token is 1 / S_PP. Fit it as `tok` plus `att` per used cell. The average used cells while the prompts run is B x npp / 2.
- **Server CPU:** `srv` adds the server's CPU time per step on top of the engine: 0.3 ms + 0.32 ms per request, measured on the A40.
- **Cell size:** set `cell_bytes` and `attn_layers` for the model's KV.

## Check against a real server

Run the same `returning` load on a server with a pool small enough to need reloads, and in the simulator. Compare:
- the wall time and generated tokens/s;
- the "failed to restore state" lines in the server log: a cached prompt that could not be loaded because idle slots still held the cells.

```sh
llama-server -m MODEL.gguf -ngl 99 -fa on -fit off -c 57344 -np 8 -b 2048 -ub 512 --kv-unified --port 8091 > server.log 2>&1 &
python3 scripts/returning.py --port 8091 --sessions 12 --clients 4 --turns 6 --prompt-tokens 3000 --user-tokens 1500 \
    --gen-tokens 300 --seed 5 --out returning.json
grep -c "failed to restore state" server.log
python3 scripts/kvswap_sim.py --workload returning --sessions 12 --clients 4 --turns-max 6 --prompt-tokens 3000 \
    --tool-tokens 1500 --gen-tokens 300 --slots 8 --pool 57344 --policies fail
```

`returning.py` prints a summary line and writes the details to `--out`, including `wall_s`, `gen_tok_s` and `n_errors`. Set `CLIENT_CPUS` (e.g. `24-31`) to pin the client away from the server's cores.
