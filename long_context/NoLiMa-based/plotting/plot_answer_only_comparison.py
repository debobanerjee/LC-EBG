#!/usr/bin/env python3
"""
Answer-Only vs Answer+Evidence Comparison (Strict Setup)
=========================================================

Compares answer accuracy between two prompting conditions:
  - Answer + Evidence: standard prompt requiring both answer and line citations
  - Answer Only: simplified prompt requesting only the answer

Strict data filters:
  - Model:      Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
  - Book:       1 only
  - Tests:      T01 only (4 needles x 2 hops = 8 tests)
  - Depths:     4 canonical (0%, 33%, 67%, 100%)
  - Characters: Yuki (0402/0402Inv), Stuart (0405/0405Inv)
  - Contexts:   Only those present in BOTH experiments

Outputs:
  results/answer_only/plots/answer_only_comparison.png
  results/answer_only/tables/answer_only_comparison.csv

Usage:
    python plot_answer_only_comparison.py
"""

import json
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from plot_style import PALETTE, apply_publication_style, save_publication_figure

warnings.filterwarnings("ignore", category=FutureWarning)
apply_publication_style()

# ─── Configuration ────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_DIR / "evaluation"
STD_DIR = EVALUATION_DIR / "results_claude-sonnet-4-5-20250929" / "commonsense_knowledge"
AO_DIR = EVALUATION_DIR / "special_experiments" / "results_claude-sonnet-4-5-20250929-answer-only" / "commonsense_knowledge"
OUTPUT_DIR = PROJECT_DIR / "results" / "answer_only"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

# Strict filters
BOOK = 1
DEPTH_TOLERANCE = 0.01
STRICT_DEPTHS = (0.0, 1/3, 2/3, 1.0)

# Canonical character assignment
NEEDLE_CHARACTER = {
    "0402":    "Yuki",
    "0402Inv": "Yuki",
    "0405":    "Stuart",
    "0405Inv": "Stuart",
}

# T01 tests only
T01_TESTS = {
    "0402_T01_C02_onehop", "0402_T01_C02_twohop",
    "0402Inv_T01_C02_onehop", "0402Inv_T01_C02_twohop",
    "0405_T01_C02_onehop", "0405_T01_C02_twohop",
    "0405Inv_T01_C02_onehop", "0405Inv_T01_C02_twohop",
}

MODEL_DIR_NAME = "claude-sonnet-4-5-20250929"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def fmt_ctx(length: int) -> str:
    if length >= 1_000_000:
        return f"{length / 1_000_000:.1f}M"
    return f"{int(length / 1_000)}K"


def is_canonical_depth(d: float) -> bool:
    return any(abs(d - sd) < DEPTH_TOLERANCE for sd in STRICT_DEPTHS)


def is_api_error(r: dict) -> bool:
    if r.get("error") or r.get("error_type"):
        return True
    if r.get("response") is None and r.get("input_tokens", 0) == 0:
        return True
    return False


def _savefig(fig, name: str, dpi: int = 300):
    out = save_publication_figure(fig, PLOTS_DIR, name, dpi=dpi)
    print(f"  Saved: {out}")


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_strict(base_dir: Path, label: str) -> pd.DataFrame:
    """Load results with strict filtering: book 1, T01, canonical depths/chars.

    Returns DataFrame with columns:
        context_length, hop, needle_id, depth, character,
        answer_correct, evidence_correct, both_correct
    """
    rows = []

    for ctx_dir in sorted(base_dir.iterdir()):
        if not ctx_dir.is_dir() or not ctx_dir.name.startswith("rand_shuffle_"):
            continue
        ctx_len = int(ctx_dir.name.replace("rand_shuffle_", ""))

        for test_dir in ctx_dir.iterdir():
            if not test_dir.is_dir():
                continue
            test_name = test_dir.name

            # Only T01 tests
            if test_name not in T01_TESTS:
                continue

            # Parse hop and needle_id
            hop = "twohop" if test_name.endswith("twohop") else "onehop"
            needle_id = test_name.split("_T")[0]

            # Expected character
            expected_char = NEEDLE_CHARACTER.get(needle_id)
            if not expected_char:
                continue

            # Only book 1
            book_file = test_dir / f"{MODEL_DIR_NAME}_rand_book_{BOOK}_{test_name}.json"
            if not book_file.exists():
                continue

            try:
                data = json.load(open(book_file))
            except Exception:
                continue

            for r in data.get("results", []):
                if not isinstance(r, dict) or is_api_error(r):
                    continue

                depth = r.get("placement_metadata", {}).get("depth")
                if depth is None:
                    continue
                depth_f = float(depth)

                # Filter to canonical depths
                if not is_canonical_depth(depth_f):
                    continue

                # Verify character
                char = r.get("selected_character", "")
                if char != expected_char:
                    continue

                ans = int(r.get("answer_metric", 0) or 0)
                evi = int(r.get("evidence_metric", 0) or 0)

                rows.append({
                    "context_length": ctx_len,
                    "hop": hop,
                    "needle_id": needle_id,
                    "depth": depth_f,
                    "character": char,
                    "answer_correct": ans,
                    "evidence_correct": evi,
                    "both_correct": 1 if (ans == 1 and evi == 1) else 0,
                })

    df = pd.DataFrame(rows)
    print(f"  [{label}] Loaded {len(df)} rows across "
          f"{df['context_length'].nunique()} contexts, "
          f"{df['needle_id'].nunique()} needles, "
          f"{df['hop'].nunique()} hops")

    if len(df) > 0:
        char_dist = df.groupby("character").size().to_dict()
        print(f"  [{label}] Characters: {char_dist}")
        depth_vals = sorted(df["depth"].unique())
        print(f"  [{label}] Depths: {[f'{d:.2f}' for d in depth_vals]}")

    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Answer-Only vs Answer+Evidence Comparison (Strict Setup)")
    print("=" * 65)

    print("\nLoading standard (Answer+Evidence) results...")
    df_std = load_strict(STD_DIR, "Standard")

    print("\nLoading answer-only results...")
    df_ao = load_strict(AO_DIR, "Answer-Only")

    if df_std.empty or df_ao.empty:
        print("ERROR: One or both datasets are empty. Cannot plot.")
        return

    # ── Find shared context lengths ──────────────────────────────────────
    shared_ctx = sorted(
        set(df_std["context_length"].unique()) & set(df_ao["context_length"].unique())
    )
    print(f"\nShared context lengths: {len(shared_ctx)}")

    df_std = df_std[df_std["context_length"].isin(shared_ctx)]
    df_ao = df_ao[df_ao["context_length"].isin(shared_ctx)]

    # ── Aggregate: mean answer accuracy per (hop, context) ───────────────
    # For standard, we have both answer_correct and both_correct
    # For answer-only, evidence_correct is always 0, so both_correct == 0
    # We compare on answer_correct (the only fair metric)
    std_agg = (
        df_std.groupby(["hop", "context_length"])["answer_correct"]
        .mean().reset_index(name="acc")
    )
    ao_agg = (
        df_ao.groupby(["hop", "context_length"])["answer_correct"]
        .mean().reset_index(name="acc")
    )

    # ── Also compute combined accuracy for standard ──────────────────────
    std_combined = (
        df_std.groupby(["hop", "context_length"])["both_correct"]
        .mean().reset_index(name="acc")
    )

    # ── Compute per-hop and overall means ────────────────────────────────
    hops = ["onehop", "twohop"]
    hop_labels = {"onehop": "One-hop", "twohop": "Two-hop"}

    # Context-weighted means: mean of per-context means
    metrics = {}  # (group_label) -> {condition_label: value}
    for hop in hops:
        label = hop_labels[hop]
        s_ans = 100 * std_agg[std_agg["hop"] == hop]["acc"].mean()
        s_comb = 100 * std_combined[std_combined["hop"] == hop]["acc"].mean()
        ao_ans = 100 * ao_agg[ao_agg["hop"] == hop]["acc"].mean()
        metrics[label] = {
            "Ans+Evid\n(answer)": s_ans,
            "Ans+Evid\n(combined)": s_comb,
            "Answer\nOnly": ao_ans,
        }

    # Overall
    metrics["Overall"] = {
        "Ans+Evid\n(answer)": 100 * std_agg["acc"].mean(),
        "Ans+Evid\n(combined)": 100 * std_combined["acc"].mean(),
        "Answer\nOnly": 100 * ao_agg["acc"].mean(),
    }

    # ── Plot: grouped bar chart ──────────────────────────────────────────
    groups = ["One-hop", "Two-hop", "Overall"]
    conditions = ["Ans+Evid\n(answer)", "Ans+Evid\n(combined)", "Answer\nOnly"]
    colors = [PALETTE[0], PALETTE[1], PALETTE[3]]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(groups))
    n_bars = len(conditions)
    bar_width = 0.22
    offsets = np.arange(n_bars) - (n_bars - 1) / 2

    for i, (cond, color) in enumerate(zip(conditions, colors)):
        vals = [metrics[g][cond] for g in groups]
        offset = offsets[i] * bar_width
        bars = ax.bar(
            x + offset, vals, bar_width,
            color=color, edgecolor="white", linewidth=0.8,
            label=cond.replace("\n", " "), alpha=0.88,
        )
        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

    ax.set_title(
        "Claude Sonnet 4.5: Answer-Only vs Answer+Evidence\n"
        "Book 1 · T01 · 4 depths · Yuki (0402) + Stuart (0405)"
        f" · {len(shared_ctx)} contexts ({fmt_ctx(shared_ctx[0])}–{fmt_ctx(shared_ctx[-1])})",
        fontsize=12, fontweight="bold", pad=14,
    )

    plt.tight_layout()
    _savefig(fig, "answer_only_comparison.png")

    # ── Summary CSV ──────────────────────────────────────────────────────
    csv_rows = []
    for hop in hops:
        for ctx in shared_ctx:
            std_val = std_agg[(std_agg["hop"] == hop) & (std_agg["context_length"] == ctx)]
            ao_val = ao_agg[(ao_agg["hop"] == hop) & (ao_agg["context_length"] == ctx)]
            comb_val = std_combined[(std_combined["hop"] == hop) & (std_combined["context_length"] == ctx)]

            s_ans = round(100 * std_val["acc"].values[0], 2) if len(std_val) > 0 else None
            ao_ans = round(100 * ao_val["acc"].values[0], 2) if len(ao_val) > 0 else None
            s_comb = round(100 * comb_val["acc"].values[0], 2) if len(comb_val) > 0 else None
            delta = round(ao_ans - s_ans, 2) if (ao_ans is not None and s_ans is not None) else None

            csv_rows.append({
                "hop": hop,
                "context_length": ctx,
                "standard_answer_acc": s_ans,
                "standard_combined_acc": s_comb,
                "answer_only_acc": ao_ans,
                "delta_pp": delta,
            })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = TABLES_DIR / "answer_only_comparison.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ── Console Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)

    for hop in hops:
        std_vals = std_agg[std_agg["hop"] == hop]["acc"].values
        ao_vals = ao_agg[ao_agg["hop"] == hop]["acc"].values
        comb_vals = std_combined[std_combined["hop"] == hop]["acc"].values

        std_mean = 100 * np.mean(std_vals)
        ao_mean = 100 * np.mean(ao_vals)
        comb_mean = 100 * np.mean(comb_vals)
        delta = ao_mean - std_mean

        print(f"\n  {hop_labels[hop]}:")
        print(f"    Standard (answer acc.):   {std_mean:.1f}%")
        print(f"    Standard (combined acc.): {comb_mean:.1f}%")
        print(f"    Answer Only (answer acc.):{ao_mean:.1f}%")
        print(f"    Delta (AO - Std answer):  {delta:+.1f} pp")

    # Overall
    std_all = 100 * std_agg["acc"].mean()
    ao_all = 100 * ao_agg["acc"].mean()
    comb_all = 100 * std_combined["acc"].mean()
    print(f"\n  Overall:")
    print(f"    Standard (answer):   {std_all:.1f}%")
    print(f"    Standard (combined): {comb_all:.1f}%")
    print(f"    Answer Only:         {ao_all:.1f}%")
    print(f"    Delta (AO - Std):    {ao_all - std_all:+.1f} pp")

    # Samples
    print(f"\n  Data points: Standard={len(df_std)}, Answer-Only={len(df_ao)}")
    print(f"  Shared contexts: {len(shared_ctx)} ({fmt_ctx(shared_ctx[0])} - {fmt_ctx(shared_ctx[-1])})")
    print("=" * 65)


if __name__ == "__main__":
    main()
