# MultiHop-RAG Dataset

[← Repo overview](../../README.md)

We build on the public [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG/)
benchmark (a multi-hop news QA dataset). We do **not** re-vendor it here; instead
we download the upstream corpus and query files and re-instrument them with
**line-level evidence** so the benchmark can be scored under EBG.

## Download

Two files are needed (≈4.9 MB + 6.5 MB) from the upstream HuggingFace dataset:

```bash
# Download into the evaluation code's dataset/ directory.
mkdir -p ../../multidocument_qa/multihop_rag/dataset
cd ../../multidocument_qa/multihop_rag/dataset

curl -L -o MultiHopRAG.json \
  https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json
curl -L -o corpus.json \
  https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json
```

- `corpus.json` — the news article corpus (the documents).
- `MultiHopRAG.json` — the multi-hop questions, each with its gold supporting
  facts (evidence) and answer.

## Generate the EBG-ready artifacts

The line-level haystacks and RAG contexts used in the paper are **derived** from
the two files above by the scripts in
[`../../multidocument_qa/multihop_rag`](../../multidocument_qa/multihop_rag). In brief:

```bash
cd ../../multidocument_qa/multihop_rag

# Per-query long-context mini-haystacks (gold docs + 20 distractors),
# with each line numbered so evidence can be cited by line.
python build_haystack.py --num-distractors 20 --out processed/queries_d20.jsonl

# Chunk + embed the corpus and rank the gold facts (retrieval artifacts).
bash run_full_retrieval.sh

# Per-K RAG contexts for the pilot queries.
python build_rag_haystack.py \
  --qids-from experiment_outputs/multihop_rag_gpt-5_<timestamp>.jsonl \
  --ks 4 10 25 50 100 200 \
  --retrieval retrieval_results/text-embedding-3-large_tokens256
```

See the evaluation README for the full pipeline and the run/aggregate steps:
[`../../multidocument_qa/README.md`](../../multidocument_qa/README.md).

## License / attribution

The underlying corpus and questions are the property of the MultiHop-RAG
authors; please cite their work and respect their license when using this data.
