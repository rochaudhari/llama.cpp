# Sharing prompt prefixes across sessions in llama-server

## Problem

Agent clients start every session with the same long prefix: tool schemas plus a system prompt. For Hermes it is 14.1K tokens. llama-server reuses only a slot's own cache, so:

- **Every new session recomputes the prefix:** 10 s alone on Qwen3.8-27B, Hermes's default local model, and about 3 s on the MoE models tested; up to 13 s while other sessions run. With 32K tokens of history per session, a new session waits up to 54 s, 21 s of it because the prefix is not shared. The other sessions stall meanwhile.
- **Saved prefixes get lost.** Idle slots are cleared whenever a new task starts. The RAM cache skips an entry when the prefix is under 25% of the saved conversation. Hybrid models such as Qwen3.6 can resume only at a saved checkpoint, and there is rarely one where two prompts start to differ.
- **Every active session holds its own copy of the prefix.** That wastes KV memory and slows every decode step, because the unified cache scans the cells of all sessions.

## Prior work

- **llama.cpp had a shared system prompt until 2024** (#9811). It was one fixed prompt, computed once and copied into every slot, and was removed as incompatible with chat templates. This proposal instead matches any prefix automatically, on the tokens after the chat template.
- **Since then, maintainers have fixed cross-slot misses with RAM copies rather than sharing** (#16391, #24190). RAM prefix entries extend that path. Related open PRs: #24143, #26204, #30137. The paged KV design (#21961) plans prefix caching for a later phase.
- **vLLM and SGLang share prefixes by default**, through automatic prefix caching and RadixAttention.

## Approach

Three opt-in changes to llama-server; the core llama.cpp library is unchanged:

1. **Cross-slot attach** (`--slot-attach`): a new session takes the shared prefix from another slot with `seq_cp`, which shares the KV cells instead of copying them. Hybrid models attach at a checkpoint of the other slot.
2. **RAM prefix entries** (`--cache-prefix`): the server keeps the state at the first user message in the RAM prompt cache as an entry of its own. The entry is copied on load, exempt from the 25% rule, and evicted least-recently-used first.
3. **Divergence checkpoints** for hybrid models: when a prompt matches cached state up to a point before its first user message, the server saves a checkpoint at that point. Later sessions can then resume exactly where prompts differ, for example at a date or session ID at the end of the system prompt.

Attach needs a unified KV cache. Both features turn themselves off on sliding-window models and DeepSeek-V4, and for requests with speculative decoding, LoRA or images. `--cache-reuse` is refused while attach is on.

## Results

Measured on one NVIDIA A40 (CUDA 12.8), with Hermes's tool schemas and instructions, Hermes's server settings (q8_0 KV, 4 slots) with MTP off, and 3 concurrent sessions.

| | Qwen3.8-27B (dense hybrid, Hermes's default) | Qwen3.6-35B-A3B (hybrid MoE) | Qwen3-30B-A3B (attention-only MoE) |
|---|---|---|---|
| ***Hermes's prompts as they are*** | | | |
| Time to first token, 3rd concurrent session | 12.9 s -> 1.2 s | 3.8 s -> 0.5 s | 6.5 s -> 4.9 s |
| Time to first token, sessions on later days (from RAM) | 10.0 s -> 2.6 s, then 1.25 s | 2.8 s -> 0.9 s, then 0.6 s | 3.0 s -> 2.9 s |
| ***One prefix shared by all sessions*** | | | |
| Time to first token, 3rd concurrent session | 12.9 s -> 0.52 s | 3.9 s -> 0.26 s | 6.6 s -> 0.11 s |
| KV memory, from no history to 32K tokens each | -66% to -23% (0.92 GiB) | -65% to -23% (0.29 GiB) | -65% to -23% (1.35 GiB) |
| Decode speed, from no history to 32K tokens each | +17% to +12% | +17% to +12% | +65% to +23% |

**As the shared prefix grows**, with one prefix shared by all 3 sessions and no history: time to first token of the 3rd session, and the decode gain. Hermes's prompt was grown with more project context:

| Shared prefix | Qwen3.8-27B | Qwen3.6-35B-A3B | Qwen3-30B-A3B |
|---|---|---|---|
| 14K tokens | 12.9 s -> 0.52 s, decode +17% | 3.9 s -> 0.26 s, decode +17% | 6.6 s -> 0.11 s, decode +65% |
| 24K tokens | 25.4 s -> 0.57 s, decode +29% | 7.6 s -> 0.29 s, decode +28% | 16.3 s -> 0.14 s, decode +92% |
| 32K tokens | 37.6 s -> 0.59 s, decode +36% | 11.6 s -> 0.30 s, decode +35% | 27.4 s -> 0.17 s, decode +106% |

Without sharing, the wait grows faster than the prefix itself. With attach it stays flat. The KV memory saved grows with the prefix, up to 2.1 GiB on the 27B, 0.65 GiB on the 35B and 3.1 GiB on the 30B at 32K.

- **The prompt layout limits sharing.** Sharing stops at the first token that differs, and Hermes puts the date and session ID at the end of its system prompt. Qwen3.6's and Qwen3.8's templates put the tools first, so almost the whole prefix is still shared. Qwen3's template puts them after the system prompt, so only 3.5K of the 14K tokens are. If Hermes moved per-session details after the tools, every template would get the "one prefix" numbers.
- **Outputs change about as much as with today's cache reuse.** Attach gives the same top-1 token as a cold run at 93-99% of positions, against 94-99% for today's prompt-cache reuse. The rest are near-ties. Sessions that share a slot's cells disturb that slot no more than unrelated sessions running next to it; on Qwen3.8-27B, its 128 output tokens stayed identical to running alone.

## Costs and limitations

- **Each new prefix is copied to RAM the first time it is seen:** 620 MiB on the 27B, 210 MiB on the 35B and 695 MiB on the 30B. With a session ID in the prefix, every session pays this, up to 0.7 s of its time to first token on the 27B. The planned fix is to copy a prefix only when it is evicted.
- **On hybrids, the first session to diverge at a new point reuses less.** It resumes from an earlier checkpoint, if there is one, and creates the checkpoint that later sessions use.
- **Not supported yet:** `--cache-reuse` stays off while attach is on, until the library supports copy-on-write. MTP is not supported either, and Hermes turns it on for models that have it.

## Next steps

1. **Harden it:** copy prefixes to RAM only on eviction, with an admission rule; add metrics, and tests in the existing suites.
2. **Library work:** copy-on-write for shared cells, and support for MTP and draft models.
3. **Change Hermes:** move per-session details after the tools, and check that answer quality is unchanged.

## How to run the benchmarks

From `prefix-sharing-bench/` on the prototype branch, after building `llama-server`:

```
# build Hermes-like prompts from Hermes's own code, in a throwaway Hermes home
HERMES_HOME=$(mktemp -d) <hermes-agent>/.venv/bin/python make_hermes_prompts.py hermes_prompts.json
# run all scenarios on both models
./run.sh > run.log 2>&1
# print the result tables
python3 analyze.py results
```

The scripts use absolute paths for the server binary, the models and the Hermes checkout; edit them for another machine. `run.sh` appends to `results/`, so move the old results away before a new run.
