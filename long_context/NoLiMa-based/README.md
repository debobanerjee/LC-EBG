# EBG — NoLiMa-based Long-Context Benchmark

[← Repo overview](../../README.md)

This directory holds the NoLiMa-based portion of the long-context (LC) EBG benchmark: an evaluation project for testing whether language models can retrieve and reason over information in large contexts when the question and the supporting evidence do not share obvious lexical overlap (the "lexical gap" the model must bridge with commonsense knowledge).

It contains:

- An evaluation runner for current API-based models (OpenAI, Anthropic, Google)
- Validation and error-analysis utilities
- Focused follow-up experiments such as answer-only prompting, two-needle reasoning, and contradictory context tests
- Plotting scripts for publication-style analysis

The benchmark data (needle sets, haystacks, helper scripts) lives in the shared repo-level `datasets/NoLiMa/` directory so it can be reused across sub-benchmarks. See [Data](#data) below.

## Project Layout

Scripts resolve paths from their own location, so they can be launched from the repo root or from this directory. Commands below are shown from `long_context/NoLiMa-based/`.

`experimentation/` — evaluation runners and experiments

- `run_full_scale.py` — Main runner for the standard benchmark. By default it uses a reduced setup that is practical for iterative testing: 1 book, 4 depths, 1 test variant per needle, and fixed canonical characters.
- `validate_pipeline.py` — Runs one or more random end-to-end checks and writes a detailed human-readable report to `validation_reports/`.
- `run_answer_only.py` / `run_answer_only_all_models.py` — Test whether removing the line-citation requirement improves answer accuracy.
- `run_two_needle.py` — Evaluates two-hop reasoning when the evidence is split across two separate needles.
- `run_contradictory_two_needle.py` — Tests whether models follow the provided context even when it contradicts world knowledge.
- `run_controlled_standard.py` — Repairs or merges canonical character/depth entries into existing standard results.

`plotting/` — analysis and figures

- `research_analysis.py` and `plot_*.py` — Generate aggregate tables and publication-style figures from completed runs.
- `generate_answer_only_csv.py` — Summarizes answer-only results into CSV.
- `plot_style.py` — Shared matplotlib styling imported by the plot scripts.

`../../datasets/NoLiMa/` — Shared NoLiMa dataset directory. The runners reference this global folder directly; there is no local `data/` copy or symlink in this benchmark.

`evaluation/` — Raw per-model run outputs written by the runners (`results_<model>/`, `special_experiments/`). These are inputs to the plotting scripts and are not tracked in git.

`results/` — Post-processed outputs produced by the plotting scripts, organized per experiment:

```
results/
├── standard/                    # research_analysis.py, plot_combined_accuracy_heatmap.py,
│   ├── plots/                   # research_analysis.py, plot_combined_accuracy_heatmap.py
│   └── tables/
├── answer_only/                 # plot_answer_only_*, plot_ablation_ans_only.py,
│   ├── plots/                   # generate_answer_only_csv.py
│   └── tables/
├── two_needle/                  # plot_two_needle_comparison.py
└── contradictory_two_needle/    # plot_contradictory_two_needle.py
```

Each experiment subdirectory holds `plots/` (`.pdf`, `.png`) and `tables/` (`.csv`). These are tracked in git so collaborators get the current figures and summaries on clone. Regenerating them requires raw JSON under `evaluation/`, which is intentionally not tracked because it is large.

## Data

The dataset lives in the shared `datasets/NoLiMa/` directory at the repository root. Runners and validation scripts resolve it directly via `../../datasets/NoLiMa/` from this project, so collaborators should update that one shared folder rather than creating benchmark-local data copies.

Relevant data files:

- `../../datasets/NoLiMa/needlesets/needle_set.json` — standard needle set
- `../../datasets/NoLiMa/needlesets/two_needle_set.json` — two-needle reasoning
- `../../datasets/NoLiMa/needlesets/contradictory_two_needle_set.json` — contradictory two-needle
- `../../datasets/NoLiMa/haystacks/rand_shuffle_*` — pre-shuffled book contexts (large, built or downloaded on demand)

To download or rebuild the haystacks and needle sets, run the utilities under
`../../datasets/NoLiMa/scripts/` from the repo root:

```bash
bash datasets/NoLiMa/scripts/download_NoLiMa_data.sh
```

See [`../../datasets/NoLiMa/README.md`](../../datasets/NoLiMa/README.md) for the full data-processing reference.

## What The Main Runner Evaluates

The current standard runner focuses on the four commonsense needles:

- `0402`
- `0402Inv`
- `0405`
- `0405Inv`

It evaluates them across:

- One-hop and two-hop question forms
- Four insertion depths: `0%`, `33%`, `67%`, `100%`
- Character-based haystacks in `../../datasets/NoLiMa/haystacks/rand_shuffle_*`

Results are stored under `evaluation/results_{model}/commonsense_knowledge/...`.

## Controlled Setup

The standard setup used by the current top-level runner and downstream analysis is intentionally controlled and narrower than the full historical benchmark. In practice, it uses:

- Haystack: `book 1` only by default
- Needle family: `0402`, `0402Inv`, `0405`, `0405Inv`
- Characters: fixed canonical names instead of sampling from the full character list
- `Yuki` for `0402` and `0402Inv`
- `Stuart` for `0405` and `0405Inv`
- Question types: both `onehop` and `twohop`
- Test variant: only the first variant, `T01_C02`
- Depths: `0%`, `33%`, `67%`, `100%`
- Books per context: `1`
- Characters per needle: `1`

This is the setup reflected in `experimentation/run_full_scale.py`, `plotting/research_analysis.py`, and the strict-filter plotting scripts. The goal is to keep comparisons stable across models and experiments while making reruns practical.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file or export the API keys you need:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```

Only the keys required by the model(s) you run need to be present.

## Reproducibility Checklist

For a fresh clone:

1. Install dependencies from this directory: `pip install -r requirements.txt`.
2. Verify the shared needle sets exist under `../../datasets/NoLiMa/needlesets/`.
3. Download or regenerate haystacks under `../../datasets/NoLiMa/haystacks/` before making model calls.
4. Run `python experimentation/run_full_scale.py --list-models` and `python experimentation/run_full_scale.py --dry-run --model gpt-4o --context-length 10000` as local smoke checks.
5. Run the desired experiment scripts to populate `evaluation/`.
6. Run the plotting scripts in [Analysis And Plotting](#analysis-and-plotting) to regenerate `results/`.

`results/` is tracked so plots and summary tables are available immediately. `evaluation/` and haystacks are not tracked; they must be produced locally or restored from an experiment artifact before plots can be regenerated from raw model outputs.

## Quick Start

List supported models:

```bash
python experimentation/run_full_scale.py --list-models
```

Preview what would run without making API calls:

```bash
python experimentation/run_full_scale.py --dry-run
```

Run one model at one context length:

```bash
python experimentation/run_full_scale.py --model gpt-4o --context-length 100000
```

Run providers in parallel:

```bash
python experimentation/run_full_scale.py --parallel
```

Run only Gemini models:

```bash
python experimentation/run_full_scale.py --gemini-only
```

Override the canonical book or character during debugging:

```bash
python experimentation/run_full_scale.py --model gpt-4.1 --book-num 3 --character Yuki
```

## Validation Workflow

Before running large sweeps, it is usually helpful to validate one or two end-to-end cases:

```bash
python experimentation/validate_pipeline.py --model gemini-2.5-flash --num-tests 2
```

Useful options:

- `--context-length 10000`
- `--depth 50`
- `--book-num 1`
- `--show-full-haystack`

Reports are written to `validation_reports/`.

## Follow-Up Experiments

Answer-only prompting:

```bash
python experimentation/run_answer_only.py --dry-run
python experimentation/run_answer_only.py --context-length 100000
```

Two-needle reasoning:

```bash
python experimentation/run_two_needle.py --model gpt-4o
```

Contradictory two-needle reasoning:

```bash
python experimentation/run_contradictory_two_needle.py --model gpt-4o
```

Controlled repair of standard results:

```bash
python experimentation/run_controlled_standard.py --dry-run
```

Experiment outputs are written either under `evaluation/results_*` or `evaluation/special_experiments/`, depending on the script.

## Analysis And Plotting

Generate the main aggregate analysis:

```bash
python plotting/research_analysis.py
```

Generate experiment-specific plots:

```bash
python plotting/plot_answer_only_comparison.py
python plotting/plot_answer_only_heatmap.py
python plotting/plot_ablation_ans_only.py
python plotting/generate_answer_only_csv.py
python plotting/plot_two_needle_comparison.py
python plotting/plot_contradictory_two_needle.py
python plotting/plot_combined_accuracy_heatmap.py
```

Figures and tables are written under `results/<experiment>/plots/` and `results/<experiment>/tables/`.

## Output Format

Standard result files are JSON objects containing:

- Run metadata such as model, test name, prompt template, and haystack path
- A `results` array with one entry per evaluated depth
- Parsed model responses
- `answer_metric` and `evidence_metric`
- Token usage and summary statistics

This makes it easy to re-score, filter, or re-plot past runs without rerunning the model.

## Attribution

This project is derived from **NoLiMa** ("No Literal Matching"), associated with the ICML 2025 paper ["NoLiMa: Long-Context Evaluation Beyond Literal Matching"](https://arxiv.org/abs/2502.05167). The original needle sets, haystack pipeline, and data-download scripts originate from that project.

Licensing applies repo-wide and is described in the [root README](../../README.md#license) and [`LICENSE`](../../LICENSE).
