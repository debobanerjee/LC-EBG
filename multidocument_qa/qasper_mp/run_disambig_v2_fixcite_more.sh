#!/usr/bin/env bash
# Extend the v2-fixcite sweep with gpt-5.5 and claude-opus-4-6.
# 2 models x 3 conditions x EBG-only = 6 runs.
set +e
cd "$(dirname "$0")/../.."

LC_QUERIES="baselines/qasper/processed/queries_dev_d10_disambig_v2.jsonl"
RAG_DIR="baselines/qasper/processed/rag_disambig_v2"
OUT_ROOT="baselines/qasper/experiment_outputs_disambig_v2_fixcite"
PYTHON="./.venv/bin/python"

run() {
  local model=$1
  local queries=$2
  local outdir=$3
  echo "=================================================="
  echo "Model=$model  Queries=$(basename "$queries")  $(date +%H:%M:%S)"
  echo "=================================================="
  $PYTHON baselines/qasper/run_baseline.py \
      --model "$model" \
      --queries "$queries" \
      --n 50 --parallel 5 --seed 0 --stratified \
      --out-dir "$outdir"
}

for MODEL in gpt-5.5 claude-opus-4-6; do
  run "$MODEL" "$LC_QUERIES"                              "$OUT_ROOT/lc"
  run "$MODEL" "$RAG_DIR/queries_flat_pool11_k3.jsonl"    "$OUT_ROOT/flat_pool11_k3"
  run "$MODEL" "$RAG_DIR/queries_struct_pool11_k3.jsonl"  "$OUT_ROOT/struct_pool11_k3"
done

echo "fixcite extension sweep done."
