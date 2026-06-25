#!/usr/bin/env bash
# Experiment A pilot on MultiHop-RAG.
# 3 models x 50 queries x {EBG, answer-only}, fixed haystack size (d=20 distractors).
#
# Run from the long-context-benchmark repo root.
PY="${PY:-$HOME/.pyenv/versions/3.11.4/bin/python3}"
SCRIPT="baselines/multihop_rag/run_baseline.py"
QUERIES="baselines/multihop_rag/processed/queries_d20.jsonl"

run_one () {
  local MODEL="$1"
  local MODE="$2"   # "" or "--answer-only"
  local LABEL="EBG"
  if [ -n "$MODE" ]; then LABEL="answer-only"; fi
  echo "=================================================="
  echo "Model: $MODEL  Mode: $LABEL  (started $(date +%T))"
  echo "=================================================="
  "$PY" "$SCRIPT" \
    --model "$MODEL" \
    --queries "$QUERIES" \
    --n 50 \
    --parallel 5 \
    --seed 0 \
    $MODE
  echo
}

for MODEL in gpt-5 sonnet-4-5 gemini-3-flash; do
  run_one "$MODEL" ""
  run_one "$MODEL" "--answer-only"
done

echo "Pilot done."
