#!/usr/bin/env bash
# Re-run EBG with citation-restriction prompt (no <title>/<abstract>/<paper> cites).
# Only re-runs configurations that contain non-paragraph citable lines (struct + LC),
# plus flat as a control. 3 models x 3 conditions x 1 mode (EBG) = 9 runs.
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

mkdir -p "$OUT_ROOT/lc" "$OUT_ROOT/flat_pool11_k3" "$OUT_ROOT/struct_pool11_k3"

for MODEL in gpt-5 sonnet-4-5 gemini-3-flash; do
  run "$MODEL" "$LC_QUERIES"                              "$OUT_ROOT/lc"
  run "$MODEL" "$RAG_DIR/queries_flat_pool11_k3.jsonl"    "$OUT_ROOT/flat_pool11_k3"
  run "$MODEL" "$RAG_DIR/queries_struct_pool11_k3.jsonl"  "$OUT_ROOT/struct_pool11_k3"
done

echo "fixcite sweep done."
