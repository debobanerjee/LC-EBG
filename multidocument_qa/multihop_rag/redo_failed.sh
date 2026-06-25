#!/usr/bin/env bash
PY="${PY:-$HOME/.pyenv/versions/3.11.4/bin/python3}"
SCRIPT="baselines/multihop_rag/run_baseline.py"

run () {
  local MODEL="$1" K="$2" MODE="$3"
  local Q="baselines/multihop_rag/processed/rag/queries_rag_k${K}.jsonl"
  echo "=== model=$MODEL K=$K mode=${MODE:-ebg} $(date +%T) ==="
  "$PY" "$SCRIPT" --model "$MODEL" --queries "$Q" --n 50 --parallel 5 --seed 0 \
    --out-dir "baselines/multihop_rag/experiment_outputs/rag_k${K}" $MODE
}

run gpt-5      50 "--answer-only"
run sonnet-4-5 50 ""
run sonnet-4-5 50 "--answer-only"
run gemini-3-flash 50 ""
run sonnet-4-5 200 ""

echo "redo done"
