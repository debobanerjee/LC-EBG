# EBG — Multi-Document QA (LC vs. RAG)

[← Repo overview](../README.md)

Evidence-based generation (EBG) evaluation comparing **long-context (LC)** and
**retrieval-augmented (RAG)** pipelines on two realistic multi-document QA
benchmarks:

- [`multihop_rag/`](multihop_rag) — multi-hop news QA (MultiHop-RAG).
- [`qasper_mp/`](qasper_mp) — multi-paper scientific QA (QASPER-MP).

Under EBG the model must return both an **answer** and the **evidence** it used
(line numbers for MultiHop-RAG, paragraph numbers for QASPER-MP), for both LC
(full haystack) and RAG (top-`k` retrieved chunks) inputs. An **answer-only**
baseline is supported throughout.

`experiments.py` here is the **shared engine** (model registry, API clients,
EBG / answer-only prompts, JSON parsing, error handling) imported by both
pipelines' `run_baseline.py`. Keep it one level up from each pipeline directory,
as laid out in this folder.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# API keys (exported, or placed in a .env in this directory which the
# scripts auto-load). Embedding + judging use OpenAI.
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```

Models are registered in `experiments.MODELS` / `experiments.Model`.

---

## MultiHop-RAG (`multihop_rag/`)

Get the data first: [`../datasets/MultiHopRAG/README.md`](../datasets/MultiHopRAG)
(downloads `corpus.json` + `MultiHopRAG.json` into `multihop_rag/dataset/`).

```bash
cd multihop_rag

# 1. Per-query LC mini-haystacks (gold docs + 20 distractors), line-numbered.
python build_haystack.py --num-distractors 20 --out processed/queries_d20.jsonl

# 2. Chunk + embed corpus and rank gold facts (all 4 retriever configs).
bash run_full_retrieval.sh

# 3. Per-K RAG haystacks for the 50-query pilot.
python build_rag_haystack.py \
  --qids-from experiment_outputs/multihop_rag_gpt-5_<timestamp>.jsonl \
  --ks 4 10 25 50 100 200 \
  --retrieval retrieval_results/text-embedding-3-large_tokens256

# 4. LC pilot: EBG + answer-only on full haystacks (3 models x 2 modes x 50 q).
bash run_pilot.sh

# 5. RAG sweep: EBG + answer-only across K.
bash run_rag_sweep.sh

# 6. Aggregate.
python aggregate_pilot.py
python aggregate_rag_sweep.py
python analyze_a_vs_b.py \
  --fact-ranks retrieval_results/text-embedding-3-large_tokens256/fact_ranks.jsonl \
  --exp-a-paths experiment_outputs/multihop_rag_*.jsonl
```

| File | Purpose |
|---|---|
| `build_haystack.py` | Per-query LC mini-haystacks. |
| `embed_and_rank.py` | Chunk corpus, embed, rank gold facts → `retrieval_results/<config>/`. |
| `build_rag_haystack.py` | Per-K RAG contexts with gold-fact re-mapping + needle-coverage diagnostics. |
| `run_baseline.py` | LLM evaluation on LC or RAG haystacks; supports `--answer-only`. |
| `run_pilot.sh` / `run_rag_sweep.sh` / `run_full_retrieval.sh` | LC pilot / RAG sweep / retriever configs. |
| `aggregate_pilot.py` / `aggregate_rag_sweep.py` / `analyze_a_vs_b.py` | Result aggregation + cross-stratification by gold-fact rank. |
| `util.py` | Small IO helpers. |

---

## QASPER-MP (`qasper_mp/`)

Get the data + build the QASPER-MP artifacts first:
[`../datasets/QASPER-MP/README.md`](../datasets/QASPER-MP) (downloads QASPER dev
into `qasper_mp/dataset/`, then runs `embed_paragraphs.py`, `build_haystack.py`,
`disambiguate_questions_v2.py`, `build_disambig_v2_artifacts.py`).

```bash
cd qasper_mp
# (after the dataset README's build steps)

# Full sweep: 3 models x 8 (condition, mode) cells.
bash run_disambig_v2_sweep.sh
# EBG-only sweep with citation-eligibility prompt + `...` markers.
bash run_disambig_v2_fixcite.sh

# LLM-judge scoring (GPT-5 grader, content-hash cached) over the outputs.
python score_with_judge.py --in-dir experiment_outputs_disambig_v2_fixcite
python score_with_judge.py --in-dir experiment_outputs_disambig_v2

# Aggregate into the LC vs. RAG table + answerable/unanswerable breakdown.
python aggregate_disambig_v2.py
python breakdown_answerability.py
```

| File | Purpose |
|---|---|
| `embed_paragraphs.py` | Embed dev paragraphs + queries (OpenAI `text-embedding-3-large`). |
| `build_haystack.py` | Per-query LC haystacks (gold + 10 distractor papers, XML-tagged). |
| `build_rag_haystack.py` | Per-query RAG haystacks (4 variants x K, with `...` discontinuity markers). |
| `disambiguate_questions_v2.py` | Rewrite questions with short author / 2–3 word title cues (deterministic 50/50). |
| `build_disambig_v2_artifacts.py` | Glue: re-embed disambiguated queries, build RAG haystacks. |
| `run_baseline.py` | Run EBG / answer-only over a `queries.jsonl` with a chosen model. |
| `score_with_judge.py` | LLM-judge scoring (GPT-5 grader, cached). |
| `aggregate_disambig_v2.py` / `breakdown_answerability.py` | LC vs. RAG table; answerable vs. unanswerable split. |
| `qasper_evaluator.py` | Token / paragraph F1 utilities (from the QASPER release). |

---

## Notes

- All scripts read/write their data under their own directory (`dataset/`,
  `processed/`, `retrieval_results/`, `experiment_outputs*/`). Those folders are
  created by the dataset-build steps above and are not checked in.
- The two `run_baseline.py` files import the shared `experiments.py` from this
  directory (one level up from each pipeline). Run scripts from inside their
  respective pipeline directory so relative paths resolve.
