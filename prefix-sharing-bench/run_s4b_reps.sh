#!/usr/bin/env bash
# Extra repetitions of the donor test (s4b), to show how much greedy outputs vary between runs of the same config.
# Usage: run_s4b_reps.sh [model ...], model names without .gguf (default: the two models below).
D=$(cd "$(dirname "$0")" && pwd); M=/localhome/local-rochaudhari/roshan/models
mkdir -p "$D/results_s4b_reps"
MODELS=("$@"); [ ${#MODELS[@]} -gt 0 ] || MODELS=(Qwen3.6-35B-A3B-UD-Q4_K_M Qwen3-30B-A3B-Instruct-2507-Q4_K_M)
for m in "${MODELS[@]}"; do
  for rep in 1 2 3; do
    echo "=== $m s4b rep $rep $(date +%T)"
    python3 "$D/bench.py" s4b --model "$M/$m.gguf" --out "$D/results_s4b_reps/$m.rep$rep.jsonl" > /dev/null 2>> "$D/results_s4b_reps/errors.log" || echo "FAILED $m s4b rep $rep"
  done
done
echo "=== done $(date +%T)"
