#!/usr/bin/env bash
# RAG sweep over k = {4, 10, 25, 50, 100, 200} on the 50 pilot queries.
# 3 models x 2 modes (EBG, answer-only). Run from the repo root.

PY="${PY:-$HOME/.pyenv/versions/3.11.4/bin/python3}"
SCRIPT="baselines/multihop_rag/run_baseline.py"

run_one () {
  local MODEL="$1"
  local MODE="$2"
  local K="$3"
  local LABEL="EBG"
  if [ -n "$MODE" ]; then LABEL="answer-only"; fi
  local QFILE="baselines/multihop_rag/processed/rag/queries_rag_k${K}.jsonl"
  echo "=================================================="
  echo "Model=$MODEL  Mode=$LABEL  K=$K  ($(date +%T))"
  echo "=================================================="
  "$PY" "$SCRIPT" \
    --model "$MODEL" \
    --queries "$QFILE" \
    --n 50 \
    --parallel 5 \
    --seed 0 \
    --out-dir baselines/multihop_rag/experiment_outputs/rag_k${K} \
    $MODE
  echo
}

for K in 4 10 25 50 100 200; do
  for MODEL in gpt-5 sonnet-4-5 gemini-3-flash; do
    run_one "$MODEL" ""             "$K"
    run_one "$MODEL" "--answer-only" "$K"
  done
done

echo "RAG sweep done."
