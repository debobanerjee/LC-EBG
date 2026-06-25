# NoLiMa Dataset

[← Repo overview](../../README.md)

Shared dataset directory for the NoLiMa-based long-context benchmark. The
sub-benchmark in `long_context/NoLiMa-based/` resolves this directory directly from
the repository root, so there is no per-benchmark copy of the NoLiMa data.

The dataset and pipeline originate from the upstream NoLiMa project
([paper](https://arxiv.org/abs/2502.05167), [HuggingFace](https://huggingface.co/datasets/amodaresi/NoLiMa)).
The Adobe Research License for the upstream materials is retained at the
repository root ([`LICENSE`](../../LICENSE)) and applies here as well.

## Layout

```
datasets/NoLiMa/
├── needlesets/   # JSON needle definitions
├── haystacks/    # Source books and pre-shuffled contexts (large, regenerable)
└── scripts/      # Dataset preparation and download utilities
```

### `needlesets/`

JSON files describing the needles inserted into haystacks for each task.

- `needle_set.json` — Standard needle set used by
  `long_context/NoLiMa-based/experimentation/run_full_scale.py` and related standard
  prompt experiments.
- `needle_set_MC.json`, `needle_set_ONLYDirect.json`, `needle_set_hard.json`,
  `needle_set_w_CoT.json`, `needle_set_w_Distractor.json` — Upstream needle
  variants for alternative prompt/answer formats.
- `two_needle_set.json` — Two-needle (two-hop) needle set used by
  `run_two_needle.py`.
- `contradictory_two_needle_set.json` — Two-needle set with contradictory
  evidence, used by `run_contradictory_two_needle.py`.

### `haystacks/`

Pre-shuffled book contexts at each evaluated length.

- `books/III-filter/` — Source books used to build the random-shuffled contexts.
- `rand_shuffle_<N>/rand_book_<k>.txt` — Five shuffled context files of length
  `N` characters, indexed `k = 1..5`. The standard runner uses book 1.
- `rand_shuffle/`, `rand_shuffle_long/` — Upstream pre-built haystacks downloaded
  by `scripts/download_NoLiMa_data.sh`.

These files are **not tracked in git** (they are large and regenerable). See
`scripts/` below for how to (re)build them.

### `scripts/`

Dataset preparation utilities. None of these are imported by the evaluation
runners; they only produce or refresh files in `needlesets/` and `haystacks/`.

- `download_NoLiMa_data.sh` — Downloads the upstream NoLiMa needle sets and
  haystacks from HuggingFace into `needlesets/` and `haystacks/`. Safe to run
  from anywhere — paths are resolved relative to the script.
- `generate_missing_haystacks.py` — Builds additional `rand_shuffle_<N>/`
  context sizes from `haystacks/books/III-filter/`. Uses seed 43 to stay
  independent from the upstream generation (seed 42).
- `update_needlesets.py` — Generates task-specific needle-set variants
  (different prompt/answer templates) from a base `needlesets/` directory.
- `book_haystack.py` — Reference `BookHaystack` helper class used during the
  original dataset construction. Kept for reproducibility; not imported by the
  evaluation code.
- `remove_distractors.ipynb` — Notebook used to strip distracting tokens from
  the source books.

## Quick start

From the repo root:

```bash
bash datasets/NoLiMa/scripts/download_NoLiMa_data.sh
```

This populates `needlesets/` and `haystacks/rand_shuffle{,_long}/`. To
generate other context lengths, edit `char_range_list` in
`scripts/generate_missing_haystacks.py` and run it.

## License

Our NoLiMa-based dataset is an adaptation of the original NoLiMa data. The needle sets, haystack pipeline, and data-download scripts of NoLiMa are released under the Adobe Research License (non-commercial research use only); please cite the NoLiMa paper and respect its license. 