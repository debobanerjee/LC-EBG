#!/usr/bin/env bash
# Run Experiment B at full scale across embed-model x chunker.
#
# Usage:  bash baselines/multihop_rag/run_full_retrieval.sh
# Run from the long-context-benchmark repo root.
set -euo pipefail

PY="${PY:-.venv-emb/bin/python}"
SCRIPT="baselines/multihop_rag/embed_and_rank.py"

run() {
  local model="$1"; local chunker="$2"
  echo "=================================================="
  echo "Config: $model + $chunker (started $(date +%T))"
  echo "=================================================="
  "$PY" "$SCRIPT" --embed-model "$model" --chunker "$chunker"
  echo "Config: $model + $chunker (finished $(date +%T))"
  echo
}

# Clear any stale pilot caches first.
rm -rf baselines/multihop_rag/retrieval_results/text-embedding-3-small_tokens256

run "text-embedding-3-small" "tokens256"
run "text-embedding-3-small" "sentences"
run "text-embedding-3-large" "tokens256"
run "text-embedding-3-large" "sentences"

echo "All configs done."
