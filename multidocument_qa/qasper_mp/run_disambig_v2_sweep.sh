#!/usr/bin/env bash
# v2 disambiguation sweep: short author / title cues (RAG task is harder).
# Same matrix as v1 so we can compare directly.
#   LC (11-paper haystack)            : 3 models x 2 modes = 6 runs
#   RAG: flat_pool11@3, struct_pool11@3, flat_full@10
#         x 3 models x 2 modes = 18 runs
# Total: 24 runs
set +e
cd "$(dirname "$0")/../.."

LC_QUERIES="baselines/qasper/processed/queries_dev_d10_disambig_v2.jsonl"
RAG_DIR="baselines/qasper/processed/rag_disambig_v2"
OUT_ROOT="baselines/qasper/experiment_outputs_disambig_v2"
PYTHON="$(cd .. && pwd)/long-context-benchmark/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="./.venv/bin/python"

run() {
  local model=$1
  local queries=$2
  local outdir=$3
  local mode=$4   # "" or "--answer-only"
  echo "=================================================="
  echo "Model=$model  Queries=$(basename "$queries")  Mode=${mode:-EBG}  $(date +%H:%M:%S)"
  echo "=================================================="
  $PYTHON baselines/qasper/run_baseline.py \
      --model "$model" \
      --queries "$queries" \
      --n 50 --parallel 5 --seed 0 --stratified \
      --out-dir "$outdir" \
      $mode
}

mkdir -p "$OUT_ROOT/lc" "$OUT_ROOT/flat_pool11_k3" "$OUT_ROOT/struct_pool11_k3" "$OUT_ROOT/flat_full_k10"

for MODEL in gpt-5 sonnet-4-5 gemini-3-flash; do
  run "$MODEL" "$LC_QUERIES"                              "$OUT_ROOT/lc"               ""
  run "$MODEL" "$LC_QUERIES"                              "$OUT_ROOT/lc"               "--answer-only"
  run "$MODEL" "$RAG_DIR/queries_flat_pool11_k3.jsonl"    "$OUT_ROOT/flat_pool11_k3"   ""
  run "$MODEL" "$RAG_DIR/queries_flat_pool11_k3.jsonl"    "$OUT_ROOT/flat_pool11_k3"   "--answer-only"
  run "$MODEL" "$RAG_DIR/queries_struct_pool11_k3.jsonl"  "$OUT_ROOT/struct_pool11_k3" ""
  run "$MODEL" "$RAG_DIR/queries_struct_pool11_k3.jsonl"  "$OUT_ROOT/struct_pool11_k3" "--answer-only"
  run "$MODEL" "$RAG_DIR/queries_flat_full_k10.jsonl"     "$OUT_ROOT/flat_full_k10"    ""
  run "$MODEL" "$RAG_DIR/queries_flat_full_k10.jsonl"     "$OUT_ROOT/flat_full_k10"    "--answer-only"
done

echo "v2 Disambiguation sweep done."
