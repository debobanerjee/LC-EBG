#!/usr/bin/env python3
"""
Contradictory Two-Needle Comparison Plot
=========================================

Compares two-needle accuracy between:
  - Factual two-needle (TN): secondary needle aligns with world knowledge
  - Contradictory two-needle (CTN): secondary needle contradicts world knowledge

Tests whether models trust context over parametric memory.

Five plots:
  (a) Overall bar chart — TN vs CTN per model (answer + combined)
  (a2) Combined-only bar chart — TN vs CTN per model (answer + evidence only)
  (b) Scaling curves — accuracy vs context length, aggregated
  (c) Secondary depth effect — accuracy by secondary needle placement
  (d) Citation breakdown — primary vs secondary citation rate

CSV output with one row per model × context_length × secondary_depth.

Outputs:
  results/contradictory_two_needle/plots/contradictory_two_needle_comparison.png
  results/contradictory_two_needle/plots/contradictory_two_needle_both_accuracy.png
  results/contradictory_two_needle/plots/contradictory_two_needle_scaling.png
  results/contradictory_two_needle/plots/contradictory_two_needle_depth_effect.png
  results/contradictory_two_needle/plots/contradictory_two_needle_citation_breakdown.png
  results/contradictory_two_needle/tables/contradictory_two_needle_comparison.csv

Usage:
    python plot_contradictory_two_needle.py
"""

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from plot_style import (
    PALETTE,
    apply_publication_style,
    build_color_map,
    save_publication_figure,
    _shorten_label,
)

warnings.filterwarnings("ignore", category=FutureWarning)
apply_publication_style()

# ─── Configuration ────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_DIR / "evaluation"
SPECIAL_ROOT = RESULTS_ROOT / "special_experiments"
OUTPUT_DIR = PROJECT_DIR / "results" / "contradictory_two_needle"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"
STRICT_BOOK = 1

# All 8 models
MODEL_DISPLAY = {
    "claude-sonnet-4-20250514":   "Claude Sonnet 4",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "gemini-2-5-flash":           "Gemini 2.5 Flash",
    "gemini-3-flash-preview":     "Gemini 3 Flash",
    "gpt-4o":                     "GPT-4o",
    "gpt-4-1":                    "GPT-4.1",
    "gpt-5-2025-08-07":           "GPT-5",
    "o3-mini-2025-01-31":         "o3-mini",
}

MODEL_COLORS = build_color_map(MODEL_DISPLAY.keys())


# ─── Helpers ──────────────────────────────────────────────────────────────────

def dname(m: str) -> str:
    return MODEL_DISPLAY.get(m, m)


def fmt_ctx(length: int) -> str:
    if length >= 1_000_000:
        return f"{length / 1_000_000:.1f}M"
    return f"{int(length / 1_000)}K"


def is_api_error(r: dict) -> bool:
    if r.get("error") or r.get("error_type"):
        return True
    if r.get("response") is None and r.get("input_tokens", 0) == 0:
        return True
    return False


def model_dir_name(m: str) -> str:
    return m.replace(".", "-").replace("/", "-")


def _savefig(fig, name: str, dpi: int = 300):
    out = save_publication_figure(fig, PLOTS_DIR, name, dpi=dpi)
    print(f"  Saved: {out}")


# ─── Data Loading ─────────────────────────────────────────────────────────────

def _load_two_needle_data(experiment_dir_suffix: str, reasoning_dir: str) -> pd.DataFrame:
    """Generic loader for two-needle experiment results."""
    rows = []

    for model_key in MODEL_DISPLAY:
        mdir = model_dir_name(model_key)
        base = SPECIAL_ROOT / f"results_{mdir}-{experiment_dir_suffix}" / reasoning_dir
        if not base.is_dir():
            continue

        for ctx_dir in sorted(base.iterdir()):
            if not ctx_dir.is_dir() or not ctx_dir.name.startswith("rand_shuffle_"):
                continue
            ctx_len = int(ctx_dir.name.replace("rand_shuffle_", ""))

            for test_dir in ctx_dir.iterdir():
                if not test_dir.is_dir():
                    continue
                test_name = test_dir.name

                book_file = test_dir / f"{mdir}_rand_book_{STRICT_BOOK}_{test_name}.json"
                if not book_file.exists():
                    continue

                try:
                    data = json.load(open(book_file))
                except Exception:
                    continue

                needle_base = test_name.replace("_twohop", "")

                for r in data.get("results", []):
                    if not isinstance(r, dict) or is_api_error(r):
                        continue

                    sec_depth = r.get("secondary_depth")
                    if sec_depth is None:
                        continue

                    ans = int(r.get("answer_metric", 0) or 0)
                    evi = int(r.get("evidence_metric", 0) or 0)
                    pri = int(r.get("primary_cited", 0) or 0)
                    sec = int(r.get("secondary_cited", 0) or 0)

                    rows.append({
                        "model": model_key,
                        "context_length": ctx_len,
                        "needle_id": needle_base,
                        "secondary_depth": float(sec_depth),
                        "answer_correct": ans,
                        "evidence_correct": evi,
                        "both_correct": 1 if (ans == 1 and evi == 1) else 0,
                        "primary_cited": pri,
                        "secondary_cited": sec,
                    })

    return pd.DataFrame(rows)


def load_factual_two_needle() -> pd.DataFrame:
    """Load factual (standard) two-needle results."""
    df = _load_two_needle_data("two-needle", "two_needle")
    print(f"  [Factual TN] {len(df)} rows, "
          f"{df['model'].nunique() if len(df) else 0} models, "
          f"{df['context_length'].nunique() if len(df) else 0} contexts")
    return df


def load_contradictory_two_needle() -> pd.DataFrame:
    """Load contradictory two-needle results."""
    df = _load_two_needle_data("contradictory-two-needle", "contradictory_two_needle")
    print(f"  [Contradictory TN] {len(df)} rows, "
          f"{df['model'].nunique() if len(df) else 0} models, "
          f"{df['context_length'].nunique() if len(df) else 0} contexts")
    return df


# ─── Plot (a): Overall Bar Chart ─────────────────────────────────────────────

def plot_overall_bars(df_tn: pd.DataFrame, df_ctn: pd.DataFrame,
                      shared_models: list, shared_ctx: set):
    """Grouped bar chart: factual TN vs contradictory TN per model.
    4 bars per model: TN-answer, TN-both, CTN-answer, CTN-both."""

    tn = df_tn[df_tn["context_length"].isin(shared_ctx)]
    ctn = df_ctn[df_ctn["context_length"].isin(shared_ctx)]

    tn_ans = (tn.groupby("model")["answer_correct"].mean() * 100).to_dict()
    tn_both = (tn.groupby("model")["both_correct"].mean() * 100).to_dict()
    ctn_ans = (ctn.groupby("model")["answer_correct"].mean() * 100).to_dict()
    ctn_both = (ctn.groupby("model")["both_correct"].mean() * 100).to_dict()

    models = sorted(shared_models, key=lambda m: tn_ans.get(m, 0), reverse=True)
    x = np.arange(len(models))
    width = 0.18
    offsets = [-(1.5 * width), -(0.5 * width), (0.5 * width), (1.5 * width)]

    fig, ax = plt.subplots(figsize=(14, 6.5))

    tn_ans_vals  = [tn_ans.get(m, 0) for m in models]
    tn_both_vals = [tn_both.get(m, 0) for m in models]
    ctn_ans_vals  = [ctn_ans.get(m, 0) for m in models]
    ctn_both_vals = [ctn_both.get(m, 0) for m in models]

    bar_specs = [
        (offsets[0], tn_ans_vals,  PALETTE[2], "TN: Ans"),
        (offsets[1], tn_both_vals, PALETTE[3], "TN: Ans+Evid"),
        (offsets[2], ctn_ans_vals,  PALETTE[1], "CTN: Ans"),
        (offsets[3], ctn_both_vals, PALETTE[0], "CTN: Ans+Evid"),
    ]

    all_bars = []
    for off, vals, color, label in bar_specs:
        bars = ax.bar(x + off, vals, width,
                      color=color, edgecolor="white", linewidth=0.6,
                      label=label, alpha=0.88)
        all_bars.append((bars, vals, color))

    for bars, vals, color in all_bars:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold", color=color)

    # Font sizes: everything 20 except title (22) and legend (16).
    BASE_FS = 20
    TITLE_FS = 22
    LEGEND_FS = 16

    ax.set_xticks(x)
    ax.set_xticklabels([_shorten_label(dname(m)) for m in models],
                       fontsize=BASE_FS, rotation=30, ha="right", fontweight="bold")
    ax.set_ylim(0, 112)
    yticks = list(range(0, 101, 20))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(t) for t in yticks],
                       fontsize=BASE_FS, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=BASE_FS, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=LEGEND_FS, loc="upper right", ncol=2)

    return fig


def plot_both_accuracy_bars(df_tn: pd.DataFrame, df_ctn: pd.DataFrame,
                            shared_models: list, shared_ctx: set):
    """Grouped bar chart for combined metric only: TN-both vs CTN-both."""

    tn = df_tn[df_tn["context_length"].isin(shared_ctx)]
    ctn = df_ctn[df_ctn["context_length"].isin(shared_ctx)]

    tn_both = (tn.groupby("model")["both_correct"].mean() * 100).to_dict()
    ctn_both = (ctn.groupby("model")["both_correct"].mean() * 100).to_dict()

    models = sorted(shared_models, key=lambda m: tn_both.get(m, 0), reverse=True)
    x = np.arange(len(models))
    width = 0.34
    offsets = [-(0.5 * width), (0.5 * width)]

    fig, ax = plt.subplots(figsize=(14, 6.5))

    tn_both_vals = [tn_both.get(m, 0) for m in models]
    ctn_both_vals = [ctn_both.get(m, 0) for m in models]

    bar_specs = [
        (offsets[0], tn_both_vals, PALETTE[3], "Factual TN: Answer + Evidence"),
        (offsets[1], ctn_both_vals, PALETTE[0], "Contradictory TN: Answer + Evidence"),
    ]

    for off, vals, color, label in bar_specs:
        bars = ax.bar(
            x + off, vals, width,
            color=color, edgecolor="white", linewidth=0.6,
            label=label, alpha=0.88
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold", color=color
            )

    ax.set_xticks(x)
    ax.set_xticklabels([dname(m) for m in models], fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")

    ax.set_title(
        "Factual vs Contradictory Two-Needle: Combined Accuracy (Answer + Evidence)\n"
        "Two-hop \u00b7 {} shared contexts \u00b7 Book 1".format(len(shared_ctx)),
        fontsize=12, fontweight="bold", pad=10,
    )

    return fig


# ─── Plot (b): Scaling Curves ────────────────────────────────────────────────

def plot_scaling_curves(df_tn: pd.DataFrame, df_ctn: pd.DataFrame,
                        shared_models: list, shared_ctx: set):
    """Accuracy vs context length. Two panels: answer (left) and combined (right)."""

    tn = df_tn[(df_tn["context_length"].isin(shared_ctx)) &
               (df_tn["model"].isin(shared_models))]
    ctn = df_ctn[(df_ctn["context_length"].isin(shared_ctx)) &
                 (df_ctn["model"].isin(shared_models))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, metric, label in [(ax1, "answer_correct", "Ans Accuracy"),
                               (ax2, "both_correct", "Ans+Evid Accuracy")]:
        tn_ctx = (tn.groupby("context_length")[metric].mean() * 100).reset_index(name="acc")
        ctn_ctx = (ctn.groupby("context_length")[metric].mean() * 100).reset_index(name="acc")

        ax.plot(tn_ctx["context_length"], tn_ctx["acc"],
                color=PALETTE[3], marker="o", markersize=5, linewidth=2.2,
                label="Factual TN", alpha=0.9)
        ax.plot(ctn_ctx["context_length"], ctn_ctx["acc"],
                color=PALETTE[0], marker="s", markersize=5, linewidth=2.2,
                label="Contradictory TN", alpha=0.9)

        merged = tn_ctx.merge(ctn_ctx, on="context_length", suffixes=("_tn", "_ctn"))
        ax.fill_between(merged["context_length"], merged["acc_tn"], merged["acc_ctn"],
                        alpha=0.12, color=PALETTE[1])

        ax.set_xscale("log")
        ax.set_xlabel("Context Length", fontsize=11)
        ax.set_ylabel(label + " (%)", fontsize=11)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=10, loc="lower left")

        milestones = [c for c in sorted(shared_ctx) if c in
                      [10000, 20000, 50000, 100000, 200000, 300000, 500000, 700000, 1000000]]
        if milestones:
            ax.set_xticks(milestones)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}K")
        )

        tn_mean = tn_ctx["acc"].mean()
        ctn_mean = ctn_ctx["acc"].mean()
        delta = ctn_mean - tn_mean
        ax.text(0.98, 0.02,
                "Mean gap: {:+.1f} pp".format(delta),
                ha="right", va="bottom", fontsize=10, fontweight="bold",
                color=PALETTE[0] if delta < 0 else PALETTE[3],
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9))

        ax.set_title(label, fontsize=11, fontweight="bold")

    fig.suptitle(
        "Factual vs Contradictory Two-Needle: Accuracy vs Context Length\n"
        "Two-hop only \u00b7 Averaged across {} models".format(len(shared_models)),
        fontsize=12, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    return fig


# ─── Plot (c): Secondary Depth Effect ────────────────────────────────────────

def plot_secondary_depth(df_tn: pd.DataFrame, df_ctn: pd.DataFrame,
                         shared_models: list):
    """Per-model subplots: accuracy by secondary depth.
    Paired bars: factual (left, lighter) vs contradictory (right, darker)."""

    def round_depth(d):
        for target in [0.0, 1/3, 2/3, 1.0]:
            if abs(d - target) < 0.02:
                return target
        return d

    tn = df_tn[df_tn["model"].isin(shared_models)].copy()
    ctn = df_ctn[df_ctn["model"].isin(shared_models)].copy()
    tn["sec_depth_rounded"] = tn["secondary_depth"].apply(round_depth)
    ctn["sec_depth_rounded"] = ctn["secondary_depth"].apply(round_depth)
    depth_vals = sorted(set(tn["sec_depth_rounded"].unique()) | set(ctn["sec_depth_rounded"].unique()))

    models = sorted(shared_models, key=dname)
    ncols = min(len(models), 4)
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), sharey=True)
    axes_flat = np.array(axes).flatten() if len(models) > 1 else [axes]
    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    depth_labels = {0.0: "0%", 1/3: "33%", 2/3: "67%", 1.0: "100%"}
    bw = 0.35

    for ax_idx, model in enumerate(models):
        ax = axes_flat[ax_idx]
        tn_sub = tn[tn["model"] == model]
        ctn_sub = ctn[ctn["model"] == model]

        tn_by_depth = (tn_sub.groupby("sec_depth_rounded")["answer_correct"].mean() * 100).to_dict()
        ctn_by_depth = (ctn_sub.groupby("sec_depth_rounded")["answer_correct"].mean() * 100).to_dict()
        x = np.arange(len(depth_vals))
        tn_vals = [tn_by_depth.get(d, 0) for d in depth_vals]
        ctn_vals = [ctn_by_depth.get(d, 0) for d in depth_vals]

        base_color = MODEL_COLORS.get(model, "#555")
        bars_tn = ax.bar(x - bw/2, tn_vals, bw, color=base_color,
                         edgecolor="white", linewidth=0.8, alpha=0.55, label="Factual")
        bars_ctn = ax.bar(x + bw/2, ctn_vals, bw, color=base_color,
                          edgecolor="white", linewidth=0.8, alpha=0.95, label="Contradictory")

        for bar, v in zip(bars_tn, tn_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#555")
        for bar, v in zip(bars_ctn, ctn_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([depth_labels.get(d, "{:.0%}".format(d)) for d in depth_vals], fontsize=10)
        ax.set_ylim(0, 115)
        ax.set_title(dname(model), fontsize=11, fontweight="bold",
                     color=MODEL_COLORS.get(model, "#333"))
        ax.grid(axis="y", alpha=0.25)

        if ax_idx % ncols == 0:
            ax.set_ylabel("Answer Accuracy (%)", fontsize=10)
        if ax_idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Secondary Needle Depth", fontsize=10)
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Factual vs Contradictory: Answer Accuracy by Secondary Needle Depth\n"
        "Primary needle fixed at 50% depth \u00b7 Two-hop only",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig


# ─── Plot (d): Citation Breakdown ────────────────────────────────────────────

def plot_citation_breakdown(df_tn: pd.DataFrame, df_ctn: pd.DataFrame,
                            shared_models: list, shared_ctx: set):
    """4 bars per model: TN-primary, TN-secondary, CTN-primary, CTN-secondary."""

    tn = df_tn[(df_tn["model"].isin(shared_models)) &
               (df_tn["context_length"].isin(shared_ctx))]
    ctn = df_ctn[(df_ctn["model"].isin(shared_models)) &
                 (df_ctn["context_length"].isin(shared_ctx))]

    # Sort by factual primary_cited descending
    tn_pri_acc = (tn.groupby("model")["primary_cited"].mean() * 100).to_dict()
    models = sorted(shared_models, key=lambda m: tn_pri_acc.get(m, 0), reverse=True)
    x = np.arange(len(models))
    width = 0.18
    offsets = [-(1.5 * width), -(0.5 * width), (0.5 * width), (1.5 * width)]

    fig, ax = plt.subplots(figsize=(14, 6.5))

    metrics = [
        ("primary_cited",   tn,  offsets[0], PALETTE[3], "Factual: Primary"),
        ("secondary_cited", tn,  offsets[1], PALETTE[2], "Factual: Secondary"),
        ("primary_cited",   ctn, offsets[2], PALETTE[0], "Contradictory: Primary"),
        ("secondary_cited", ctn, offsets[3], PALETTE[1], "Contradictory: Secondary"),
    ]

    for col, df, off, color, label in metrics:
        accs = (df.groupby("model")[col].mean() * 100).to_dict()
        vals = [accs.get(m, 0) for m in models]
        bars = ax.bar(x + off, vals, width,
                      color=color, edgecolor="white", linewidth=0.6,
                      label=label, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=7,
                    fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([dname(m) for m in models], fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Citation Rate (%)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper right", ncol=2)

    ax.set_title(
        "Factual vs Contradictory Two-Needle: Citation Rates\n"
        "Two-hop \u00b7 {} contexts \u00b7 Book 1".format(len(shared_ctx)),
        fontsize=12, fontweight="bold", pad=10,
    )

    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Factual vs Contradictory Two-Needle Comparison")
    print("=" * 65)

    print("\nLoading factual two-needle data...")
    df_tn = load_factual_two_needle()

    print("\nLoading contradictory two-needle data...")
    df_ctn = load_contradictory_two_needle()

    if df_tn.empty or df_ctn.empty:
        print("ERROR: Missing data. Cannot plot.")
        if df_tn.empty:
            print("  → No factual two-needle data found.")
        if df_ctn.empty:
            print("  → No contradictory two-needle data found.")
            print("  → Run: python run_contradictory_two_needle.py")
        return

    # Find shared models and contexts
    shared_models = sorted(
        set(df_tn["model"].unique()) & set(df_ctn["model"].unique())
    )
    shared_ctx = sorted(
        set(df_tn["context_length"].unique()) & set(df_ctn["context_length"].unique())
    )
    print(f"\nShared models: {len(shared_models)} — {[dname(m) for m in shared_models]}")
    print(f"Shared contexts: {len(shared_ctx)} ({fmt_ctx(shared_ctx[0])}\u2013{fmt_ctx(shared_ctx[-1])})")

    shared_ctx_set = set(shared_ctx)

    # ── Plot (a): Overall bars ───────────────────────────────────────────
    print("\n(a) Overall bar chart...")
    fig_a = plot_overall_bars(df_tn, df_ctn, shared_models, shared_ctx_set)
    # Save directly: this figure uses its own custom font sizes, so we
    # bypass save_publication_figure / style_figure (which would override them).
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_a_path = PLOTS_DIR / "contradictory_two_needle_comparison.pdf"
    fig_a.savefig(fig_a_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_a)
    print(f"  Saved: {fig_a_path}")

    # ── Plot (a2): Combined-only overall bars ───────────────────────────
    print("\n(a2) Combined-only bar chart...")
    fig_a2 = plot_both_accuracy_bars(df_tn, df_ctn, shared_models, shared_ctx_set)
    _savefig(fig_a2, "contradictory_two_needle_both_accuracy.png")

    # ── Plot (b): Scaling curves ─────────────────────────────────────────
    print("\n(b) Scaling curves...")
    fig_b = plot_scaling_curves(df_tn, df_ctn, shared_models, shared_ctx_set)
    _savefig(fig_b, "contradictory_two_needle_scaling.png")

    # ── Plot (c): Secondary depth effect ─────────────────────────────────
    print("\n(c) Secondary depth effect...")
    fig_c = plot_secondary_depth(df_tn, df_ctn, shared_models)
    _savefig(fig_c, "contradictory_two_needle_depth_effect.png")

    # ── Plot (d): Citation breakdown ─────────────────────────────────────
    print("\n(d) Citation breakdown...")
    fig_d = plot_citation_breakdown(df_tn, df_ctn, shared_models, shared_ctx_set)
    _savefig(fig_d, "contradictory_two_needle_citation_breakdown.png")

    # ── Summary CSV ──────────────────────────────────────────────────────
    csv_rows = []
    tn_filt = df_tn[df_tn["context_length"].isin(shared_ctx_set)]
    ctn_filt = df_ctn[df_ctn["context_length"].isin(shared_ctx_set)]

    def _round_sd(d):
        for t in [0.0, 1/3, 2/3, 1.0]:
            if abs(d - t) < 0.02:
                return round(t, 4)
        return round(d, 4)

    tn_filt = tn_filt.copy()
    ctn_filt = ctn_filt.copy()
    tn_filt["sec_depth_rounded"] = tn_filt["secondary_depth"].apply(_round_sd)
    ctn_filt["sec_depth_rounded"] = ctn_filt["secondary_depth"].apply(_round_sd)
    sec_depths = sorted(
        set(tn_filt["sec_depth_rounded"].unique()) | set(ctn_filt["sec_depth_rounded"].unique())
    )
    depth_pct = {0.0: 0, round(1/3, 4): 33, round(2/3, 4): 67, 1.0: 100}

    for model in shared_models:
        tn_m = tn_filt[tn_filt["model"] == model]
        ctn_m = ctn_filt[ctn_filt["model"] == model]
        for ctx in shared_ctx:
            tn_v = tn_m[tn_m["context_length"] == ctx]
            ctn_v = ctn_m[ctn_m["context_length"] == ctx]

            if len(tn_v) == 0 or len(ctn_v) == 0:
                continue

            for sd in sec_depths:
                tn_sd = tn_v[tn_v["sec_depth_rounded"] == sd]
                ctn_sd = ctn_v[ctn_v["sec_depth_rounded"] == sd]
                if len(tn_sd) == 0 or len(ctn_sd) == 0:
                    continue

                csv_rows.append({
                    "model": dname(model),
                    "context_length": ctx,
                    "primary_depth": 50,
                    "secondary_depth": depth_pct.get(sd, int(round(sd * 100))),
                    "tn_answer": round(100 * tn_sd["answer_correct"].mean(), 2),
                    "tn_primary_cited": round(100 * tn_sd["primary_cited"].mean(), 2),
                    "tn_secondary_cited": round(100 * tn_sd["secondary_cited"].mean(), 2),
                    "tn_both": round(100 * tn_sd["both_correct"].mean(), 2),
                    "ctn_answer": round(100 * ctn_sd["answer_correct"].mean(), 2),
                    "ctn_primary_cited": round(100 * ctn_sd["primary_cited"].mean(), 2),
                    "ctn_secondary_cited": round(100 * ctn_sd["secondary_cited"].mean(), 2),
                    "ctn_both": round(100 * ctn_sd["both_correct"].mean(), 2),
                })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = TABLES_DIR / "contradictory_two_needle_comparison.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}  ({len(csv_df)} rows)")

    # ── Console Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY (shared contexts only)")
    print("=" * 70)
    hdr = "  {:<18s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "Model", "TN-Ans", "TN-Pri", "TN-Sec", "TN-Both",
        "CTN-Ans", "CTN-Pri", "CTN-Sec", "CTN-Bot")
    sep = "  " + "\u2500" * 18 + " " + ("\u2500" * 8 + " ") * 8
    print(hdr)
    print(sep)

    for model in sorted(shared_models, key=dname):
        tn_m = tn_filt[tn_filt["model"] == model]
        ctn_m = ctn_filt[ctn_filt["model"] == model]
        vals = []
        for df_m in [tn_m, ctn_m]:
            vals.extend([
                100 * df_m["answer_correct"].mean(),
                100 * df_m["primary_cited"].mean(),
                100 * df_m["secondary_cited"].mean(),
                100 * df_m["both_correct"].mean(),
            ])
        print("  {:<18s} {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}%".format(
            dname(model), *vals))

    # Overall
    tn_vals = [
        100 * tn_filt["answer_correct"].mean(),
        100 * tn_filt["primary_cited"].mean(),
        100 * tn_filt["secondary_cited"].mean(),
        100 * tn_filt["both_correct"].mean(),
    ]
    ctn_vals = [
        100 * ctn_filt["answer_correct"].mean(),
        100 * ctn_filt["primary_cited"].mean(),
        100 * ctn_filt["secondary_cited"].mean(),
        100 * ctn_filt["both_correct"].mean(),
    ]
    print("\n  {:<18s} {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}%".format(
        "Overall", *(tn_vals + ctn_vals)))
    print(f"\n  Data points: Factual={len(tn_filt)}, Contradictory={len(ctn_filt)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
