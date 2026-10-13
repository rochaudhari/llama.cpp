#!/usr/bin/env bash
# Benchmark run: hybrid MoE and attention-only MoE, Hermes-like prompts and server flags.
# Usage: run.sh [model ...], model names without .gguf (default: the two models below).
D=$(cd "$(dirname "$0")" && pwd); M=/localhome/local-rochaudhari/roshan/models
mkdir -p "$D/results"
MODELS=("$@"); [ ${#MODELS[@]} -gt 0 ] || MODELS=(Qwen3.6-35B-A3B-UD-Q4_K_M Qwen3-30B-A3B-Instruct-2507-Q4_K_M)
for m in "${MODELS[@]}"; do
  out="$D/results/$m.jsonl"
  for sc in "s1" "s2" "s3" "s3 --kv-debug --reps 1" "s4a" "s4b" "s5" "s5 --kv-debug --reps 1"; do
    echo "=== $m $sc $(date +%T)"
    python3 "$D/bench.py" $sc --model "$M/$m.gguf" --out "$out" > /dev/null 2>> "$D/results/errors.log" || echo "FAILED $m $sc"
  done
done
echo "=== done $(date +%T)"
