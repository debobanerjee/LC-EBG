# QASPER-MP Dataset (Multi-Paper QASPER)

[← Repo overview](../../README.md)

**QASPER-MP** adapts the single-paper
[QASPER](https://allenai.org/data/qasper) benchmark to a realistic
**multi-document** setting. Each question is placed in an 11-paper haystack
(its gold paper + 10 distractor papers) and rewritten with a short, realistic
**paper cue** (an author surname or a 2–3 word verbatim title fragment) so that
it is well-posed when several papers are present. The model must answer **and**
cite the supporting paragraph numbers.

QASPER-MP is **generated from the public QASPER dev split** — there is no
separate file to download beyond QASPER itself.

## 1. Download QASPER (AllenAI)

```bash
# Download into the evaluation code's dataset/ directory.
mkdir -p ../../multidocument_qa/qasper_mp/dataset
cd ../../multidocument_qa/qasper_mp/dataset

curl -O https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
tar xzf qasper-train-dev-v0.3.tgz   # provides qasper-dev-v0.3.json
```

We use the **dev** split (`qasper-dev-v0.3.json`).

## 2. Generate QASPER-MP artifacts

The multi-paper haystacks, disambiguated questions, and RAG contexts are built
by the scripts in
[`../../multidocument_qa/qasper_mp`](../../multidocument_qa/qasper_mp). The chain is:

```bash
cd ../../multidocument_qa/qasper_mp

# Embed all dev paragraphs + queries (OpenAI text-embedding-3-large).
python embed_paragraphs.py

# Per-query LC mini-haystacks: gold paper + 10 distractor papers, XML-tagged.
python build_haystack.py

# Rewrite questions with short author / 2-3 word title cues (deterministic 50/50).
python disambiguate_questions_v2.py

# Glue: re-embed disambiguated queries and build the RAG haystacks
# (flat/struct x pool11/full, with `...` discontinuity markers).
python build_disambig_v2_artifacts.py
```

This yields, under the evaluation directory:

- `processed/queries_dev_d10_disambig_v2.jsonl` — the LC QASPER-MP haystacks.
- `processed/disambiguated_questions_v2.json` — the disambiguated questions.
- `processed/rag_disambig_v2/queries_<variant>_k<K>.jsonl` — the RAG contexts.
- `retrieval_results/dev_large_disambig_v2/` — re-embedded queries.

See the evaluation README for the run / judge / aggregate steps:
[`../../multidocument_qa/README.md`](../../multidocument_qa/README.md).

## License / attribution

QASPER is released by AllenAI; please cite the QASPER paper and respect its
license. QASPER-MP is a derived multi-paper arrangement of the same data.
