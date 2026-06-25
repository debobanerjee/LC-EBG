# Synthetic Two-Hop Needle-in-a-Haystack Dataset

[← Repo overview](../../README.md)

The synthetic benchmark probes long-context **evidence-based generation**: a
small number of two-hop "needles" (pairs of facts that must be combined to
answer a question) are inserted at random positions into a haystack of
distractor facts, and the haystack is grown from 10K up to several million
characters. The model must return both the answer **and** the line numbers of
the supporting facts.

The dataset is **generated programmatically at run time**, so there is no large
file to download. It has two ingredients:

1. **A distractor pool** of short, self-contained facts (`random_facts.txt`).
2. **A fixed set of two-hop needles** (questions + gold fact pairs), defined as
   `TWO_HOP_EXAMPLES` in the evaluation engine
   ([`../../long_context/Synthetic-based/experiments.py`](../../long_context/Synthetic-based/experiments.py)).

At evaluation time, `experiments.add_random_facts(...)` shuffles the distractor
pool, truncates it to the target character budget, and inserts the needles at
random positions; each line is then numbered `"{line_no}: {text}"` so the model
can cite by line.

## Generating the distractor pool

The distractor pool was produced with the two notebooks in this directory:

| Notebook | Purpose |
|---|---|
| `creation_of_facts.ipynb` | Prompts an LLM to synthesize large batches of short, diverse, self-contained facts (people, companies, products, places). |
| `generate_random_facts.ipynb` | Cleans, de-duplicates, and flattens the generated facts into the one-fact-per-line `random_facts.txt` used by the evaluator. |

Run both notebooks top-to-bottom (an `OPENAI_API_KEY` is required for
generation), then save the result as `random_facts.txt`:

```bash
# Output format: one fact per line, e.g.
#   Roald Dahl wrote "Charlie and the Chocolate Factory."
#   Tech innovator IntelliSoft introduced a project-management platform.
```

The released pool contains ~66.8K facts (~5.6 MB). Any plain-text file with one
short fact per line works; larger pools simply allow larger haystacks.

## Using the pool

Place `random_facts.txt` next to the evaluation driver (or pass `--facts-file`)
and follow [`../../long_context/Synthetic-based/README.md`](../../long_context/Synthetic-based/README.md):

```bash
cp random_facts.txt ../../long_context/Synthetic-based/
```

## Editing the needles

To change the two-hop questions, edit `TWO_HOP_EXAMPLES` in
`long_context/Synthetic-based/experiments.py`. Each entry lists the gold facts, the
question, and a set of accepted answer strings.
