# EBG — Synthetic Two-Hop Benchmark

[← Repo overview](../../README.md)

Long-context **evidence-based generation** on the synthetic two-hop
needle-in-a-haystack task. The driver grows a haystack of distractor facts from
10K up to millions of characters, inserts a fixed set of two-hop needles, and
asks the model to return an answer **plus the line numbers** of the supporting
facts. An answer-only baseline (no citation requirement) is also supported.

See [`../../datasets/Synthetic`](../../datasets/Synthetic) for how the haystack
data is generated.

## Files

| File | Purpose |
|---|---|
| `experiments.py` | Shared engine: model registry (`MODELS`), API clients, EBG / answer-only prompts, the `TWO_HOP_EXAMPLES` needles, haystack assembly (`add_random_facts`), scoring, and the `run_experiments` loop. |
| `run_experiments.py` | Command-line driver (model selection, context-length presets, trials, resume). |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# API keys for the providers you evaluate (exported or in a .env here).
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...

# Distractor pool (see datasets/Synthetic/README.md to (re)generate it).
cp ../../datasets/Synthetic/random_facts.txt .   # or pass --facts-file <path>
```

## Running

```bash
# Discover models and context-length presets.
python run_experiments.py --list-models
python run_experiments.py --list-presets

# Answer + evidence (EBG) — e.g. the "short" preset (10K–100K chars).
python run_experiments.py --model gpt-5 --preset short --trials 100 --parallel 5

# Answer-only baseline (no citation requirement).
python run_experiments.py --model gpt-5 --preset short --trials 100 --answer-only

# Explicit context lengths instead of a preset.
python run_experiments.py --model sonnet-4.5 --characters 10000,50000,100000

# Resume an interrupted / partial run.
python run_experiments.py --resume experiment_outputs/<model>_<timestamp>.jsonl
```

Presets (characters): `tiny`, `short` (10K–100K), `mid` (200K–1M), `long`
(2M–5M), `full` (10K–5M).

## Outputs

Results are written as JSONL to `experiment_outputs/<model>[_answer_only]_<timestamp>.jsonl`,
one record per (context length, trial) with the model's parsed answer, cited
lines, the gold answer/lines, and per-record correctness. Aggregate accuracy
vs. context length to reproduce the synthetic figures.
