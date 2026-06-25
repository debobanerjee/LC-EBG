#!/usr/bin/env python3
"""
Two-Needle vs Single-Needle Comparison Plot
=============================================

Compares two-hop accuracy between:
  - Single-needle: standard experiment (both facts in one needle, twohop questions)
  - Two-needle: primary needle at 50% depth, secondary at varying depths

Three plot panels:
  (a) Overall bar chart — single vs two-needle per model
  (b) Scaling curves — accuracy vs context length, aggregated
  (c) Secondary depth effect — accuracy by secondary needle placement

Strict filters (single-needle baseline):
  - Book 1, T01, 4 depths, canonical chars, twohop only

Two-needle data:
  - Book 1, 4 secondary depths, twohop only, all 4 needle pairs

Outputs:
  results/two_needle/plots/two_needle_comparison.png
  results/two_needle/tables/two_needle_comparison.csv

Usage:
    python plot_two_needle_comparison.py
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
REASONING_TYPE = "commonsense_knowledge"
OUTPUT_DIR = PROJECT_DIR / "results" / "two_needle"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

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

# Strict filters for single-needle baseline
STRICT_BOOK = 1
DEPTH_TOLERANCE = 0.02
STRICT_DEPTHS = (0.0, 1/3, 2/3, 1.0)
NEEDLE_CHARACTER = {
    "0402": "Yuki", "0402Inv": "Yuki",
    "0405": "Stuart", "0405Inv": "Stuart",
}
# Only twohop T01 tests for single-needle baseline
TWOHOP_T01_TESTS = {
    "0402_T01_C02_twohop",
    "0402Inv_T01_C02_twohop",
    "0405_T01_C02_twohop",
    "0405Inv_T01_C02_twohop",
}

# Two-needle test names
TN_TESTS = ["TN_0402_twohop", "TN_0402Inv_twohop", "TN_0405_twohop", "TN_0405Inv_twohop"]
TN_CHARACTERS = {
    "TN_0402": "Yuki", "TN_0402Inv": "Yuki",
    "TN_0405": "Stuart", "TN_0405Inv": "Stuart",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def dname(m: str) -> str:
    return MODEL_DISPLAY.get(m, m)


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


def model_dir_name(m: str) -> str:
    return m.replace(".", "-").replace("/", "-")


def _savefig(fig, name: str, dpi: int = 300):
    out = save_publication_figure(fig, PLOTS_DIR, name, dpi=dpi)
    print(f"  Saved: {out}")


# ─── Data Loading: Single-Needle (standard twohop) ───────────────────────────

def load_single_needle() -> pd.DataFrame:
    """Load standard single-needle twohop results (strict setup)."""
    rows = []

    for model_key in MODEL_DISPLAY:
        mdir = model_dir_name(model_key)
        base = RESULTS_ROOT / f"results_{mdir}" / REASONING_TYPE
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
                if test_name not in TWOHOP_T01_TESTS:
                    continue

                needle_id = test_name.split("_T")[0]
                expected_char = NEEDLE_CHARACTER.get(needle_id)
                if not expected_char:
                    continue

                book_file = test_dir / f"{mdir}_rand_book_{STRICT_BOOK}_{test_name}.json"
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
                    if not is_canonical_depth(depth_f):
                        continue

                    char = r.get("selected_character", "")
                    if char != expected_char:
                        continue

                    ans = int(r.get("answer_metric", 0) or 0)
                    evi = int(r.get("evidence_metric", 0) or 0)

                    rows.append({
                        "model": model_key,
                        "context_length": ctx_len,
                        "needle_id": needle_id,
                        "depth": depth_f,
                        "answer_correct": ans,
                        "evidence_correct": evi,
                        "both_correct": 1 if (ans == 1 and evi == 1) else 0,
                    })

    df = pd.DataFrame(rows)
    print(f"  [Single-needle] {len(df)} rows, "
          f"{df['model'].nunique()} models, "
          f"{df['context_length'].nunique()} contexts")
    return df


# ─── Data Loading: Two-Needle ────────────────────────────────────────────────

def load_two_needle() -> pd.DataFrame:
    """Load two-needle experiment results (book 1, all secondary depths)."""
    rows = []

    for model_key in MODEL_DISPLAY:
        mdir = model_dir_name(model_key)
        base = SPECIAL_ROOT / f"results_{mdir}-two-needle" / "two_needle"
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

                # Only book 1
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

    df = pd.DataFrame(rows)
    print(f"  [Two-needle] {len(df)} rows, "
          f"{df['model'].nunique()} models, "
          f"{df['context_length'].nunique()} contexts")
    return df


# ─── Plot (a): Overall Bar Chart ─────────────────────────────────────────────

def plot_overall_bars(df_sn: pd.DataFrame, df_tn: pd.DataFrame,
                      shared_models: list, shared_ctx: set):
    """Grouped bar chart: single-needle vs two-needle per model.
    Shows 4 bars per model: SN-answer, SN-both, TN-answer, TN-both."""

    # Filter to shared contexts
    sn = df_sn[df_sn["context_length"].isin(shared_ctx)]
    tn = df_tn[df_tn["context_length"].isin(shared_ctx)]

    # Per-model accuracy for both metrics
    sn_ans = (sn.groupby("model")["answer_correct"].mean() * 100).to_dict()
    sn_both = (sn.groupby("model")["both_correct"].mean() * 100).to_dict()
    tn_ans = (tn.groupby("model")["answer_correct"].mean() * 100).to_dict()
    tn_both = (tn.groupby("model")["both_correct"].mean() * 100).to_dict()

    models = sorted(shared_models, key=lambda m: sn_ans.get(m, 0), reverse=True)
    x = np.arange(len(models))
    n_bars = 4
    width = 0.18
    offsets = [-(1.5 * width), -(0.5 * width), (0.5 * width), (1.5 * width)]

    fig, ax = plt.subplots(figsize=(14, 6.5))

    sn_ans_vals  = [sn_ans.get(m, 0) for m in models]
    sn_both_vals = [sn_both.get(m, 0) for m in models]
    tn_ans_vals  = [tn_ans.get(m, 0) for m in models]
    tn_both_vals = [tn_both.get(m, 0) for m in models]

    bar_specs = [
        (offsets[0], sn_ans_vals,  PALETTE[3], "SN: Answer"),
        (offsets[1], sn_both_vals, PALETTE[3], "SN: Answer + Evidence"),
        (offsets[2], tn_ans_vals,  PALETTE[0], "TN: Answer"),
        (offsets[3], tn_both_vals, PALETTE[0], "TN: Answer + Evidence"),
    ]

    all_bars = []
    for off, vals, color, label in bar_specs:
        bars = ax.bar(x + off, vals, width,
                      color=color, edgecolor="white", linewidth=0.6,
                      label=label, alpha=0.88)
        all_bars.append((bars, vals, color))

    # Value labels
    for bars, vals, color in all_bars:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([dname(m) for m in models], fontsize=14, rotation=10, ha="right")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Accuracy (%)", fontsize=16)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=14, loc="upper right", ncol=2)

    ax.set_title(
        "Single-Needle vs Two-Needle: Answer & Combined Accuracy\n"
        f"Two-hop · {len(shared_ctx)} shared contexts · Book 1",
        fontsize=12, fontweight="bold", pad=10,
    )

    return fig


# ─── Plot (b): Scaling Curves ────────────────────────────────────────────────

def plot_scaling_curves(df_sn: pd.DataFrame, df_tn: pd.DataFrame,
                        shared_models: list, shared_ctx: set):
    """Accuracy vs context length, single vs two-needle.
    Two panels: answer_correct (left) and both_correct (right)."""

    sn = df_sn[(df_sn["context_length"].isin(shared_ctx)) &
               (df_sn["model"].isin(shared_models))]
    tn = df_tn[(df_tn["context_length"].isin(shared_ctx)) &
               (df_tn["model"].isin(shared_models))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, metric, label in [(ax1, "answer_correct", "Answer Accuracy"),
                               (ax2, "both_correct", "Answer + Evidence Accuracy")]:
        sn_ctx = (sn.groupby("context_length")[metric].mean() * 100).reset_index(name="acc")
        tn_ctx = (tn.groupby("context_length")[metric].mean() * 100).reset_index(name="acc")

        ax.plot(sn_ctx["context_length"], sn_ctx["acc"],
                color=PALETTE[3], marker="o", markersize=5, linewidth=2.2,
                label="Single-needle", alpha=0.9)
        ax.plot(tn_ctx["context_length"], tn_ctx["acc"],
                color=PALETTE[0], marker="s", markersize=5, linewidth=2.2,
                label="Two-needle", alpha=0.9)

        merged = sn_ctx.merge(tn_ctx, on="context_length", suffixes=("_sn", "_tn"))
        ax.fill_between(merged["context_length"], merged["acc_sn"], merged["acc_tn"],
                        alpha=0.12, color=PALETTE[0])

        ax.set_xscale("log")
        ax.set_xlabel("Context Length", fontsize=16)
        ax.set_ylabel(label + " (%)", fontsize=16)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=14, loc="lower left")

        milestones = [c for c in sorted(shared_ctx) if c in
                      [10000, 20000, 50000, 100000, 200000, 300000, 500000, 700000, 1000000]]
        if milestones:
            ax.set_xticks(milestones)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{int(x/1e3)}K")
        )

        sn_mean = sn_ctx["acc"].mean()
        tn_mean = tn_ctx["acc"].mean()
        delta = tn_mean - sn_mean
        ax.text(0.98, 0.02,
                f"Mean gap: {delta:+.1f} pp",
                ha="right", va="bottom", fontsize=10, fontweight="bold",
                color=PALETTE[0] if delta < 0 else PALETTE[3],
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9))

        ax.set_title(label, fontsize=11, fontweight="bold")

    fig.suptitle(
        "Single-Needle vs Two-Needle: Accuracy vs Context Length\n"
        f"Two-hop only · Averaged across {len(shared_models)} models",
        fontsize=12, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    return fig


# ─── Plot (c): Secondary Depth Effect ────────────────────────────────────────

def plot_secondary_depth(df_tn: pd.DataFrame, shared_models: list):
    """Accuracy by secondary needle depth, per model.
    Each subplot shows two bar groups: answer_correct and both_correct."""

    tn = df_tn[df_tn["model"].isin(shared_models)].copy()

    # Round secondary_depth to canonical values for grouping
    def round_depth(d):
        for target in [0.0, 1/3, 2/3, 1.0]:
            if abs(d - target) < 0.02:
                return target
        return d

    tn["sec_depth_rounded"] = tn["secondary_depth"].apply(round_depth)
    depth_vals = sorted(tn["sec_depth_rounded"].unique())

    models = sorted(shared_models, key=dname)
    ncols = min(len(models), 4)
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), sharey=True)
    axes_flat = np.array(axes).flatten() if len(models) > 1 else [axes]
    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    depth_labels = {0.0: "0%", 1/3: "33%", 2/3: "67%", 1.0: "100%"}
    bw = 0.35  # bar width

    for ax_idx, model in enumerate(models):
        ax = axes_flat[ax_idx]
        sub = tn[tn["model"] == model]

        ans_by_depth = (sub.groupby("sec_depth_rounded")["answer_correct"].mean() * 100).to_dict()
        both_by_depth = (sub.groupby("sec_depth_rounded")["both_correct"].mean() * 100).to_dict()
        x = np.arange(len(depth_vals))
        ans_vals = [ans_by_depth.get(d, 0) for d in depth_vals]
        both_vals = [both_by_depth.get(d, 0) for d in depth_vals]

        base_color = MODEL_COLORS.get(model, "#555")
        bars_a = ax.bar(x - bw/2, ans_vals, bw, color=base_color,
                        edgecolor="white", linewidth=0.8, alpha=0.55, label="Answer")
        bars_b = ax.bar(x + bw/2, both_vals, bw, color=base_color,
                        edgecolor="white", linewidth=0.8, alpha=0.95, label="Ans+Evid")

        for bar, v in zip(bars_a, ans_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#555")
        for bar, v in zip(bars_b, both_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([depth_labels.get(d, f"{d:.0%}") for d in depth_vals], fontsize=10)
        ax.set_ylim(0, 115)
        ax.set_title(dname(model), fontsize=11, fontweight="bold",
                     color=MODEL_COLORS.get(model, "#333"))
        ax.grid(axis="y", alpha=0.25)

        if ax_idx % ncols == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=16)
        if ax_idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Secondary Needle Depth", fontsize=16)
        if ax_idx == 0:
            ax.legend(fontsize=14, loc="upper right")

    fig.suptitle(
        "Two-Needle: Accuracy by Secondary Needle Placement\n"
        "Primary needle fixed at 50% depth · Two-hop only",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig


# ─── Plot (d): Citation Breakdown ────────────────────────────────────────────

def plot_citation_breakdown(df_tn: pd.DataFrame, shared_models: list, shared_ctx: set):
    """Bar chart showing primary_cited and secondary_cited per model."""

    tn = df_tn[(df_tn["model"].isin(shared_models)) &
              (df_tn["context_length"].isin(shared_ctx))]

    metrics = [
        ("primary_cited",   "Primary Evidence",    PALETTE[3]),
        ("secondary_cited", "Secondary Evidence",  PALETTE[0]),
    ]

    # Sort models by primary evidence accuracy descending
    pri_acc = (tn.groupby("model")["primary_cited"].mean() * 100).to_dict()
    models = sorted(shared_models, key=lambda m: pri_acc.get(m, 0), reverse=True)
    x = np.arange(len(models))
    n = len(metrics)
    width = 0.32

    fig, ax = plt.subplots(figsize=(13, 6))

    for m_idx, (col, label, color) in enumerate(metrics):
        accs = (tn.groupby("model")[col].mean() * 100).to_dict()
        vals = [accs.get(m, 0) for m in models]
        offset = (m_idx - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      color=color, edgecolor="white", linewidth=0.6,
                      label=label, alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5,
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
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=LEGEND_FS, loc="upper right", bbox_to_anchor=(0.85, 1.0))

    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Two-Needle vs Single-Needle Comparison")
    print("=" * 65)

    print("\nLoading single-needle (standard twohop) baseline...")
    df_sn = load_single_needle()

    print("\nLoading two-needle experiment data...")
    df_tn = load_two_needle()

    if df_sn.empty or df_tn.empty:
        print("ERROR: Missing data. Cannot plot.")
        return

    # Find shared models and contexts
    shared_models = sorted(
        set(df_sn["model"].unique()) & set(df_tn["model"].unique())
    )
    shared_ctx = sorted(
        set(df_sn["context_length"].unique()) & set(df_tn["context_length"].unique())
    )
    print(f"\nShared models: {len(shared_models)} — {[dname(m) for m in shared_models]}")
    print(f"Shared contexts: {len(shared_ctx)} ({fmt_ctx(shared_ctx[0])}–{fmt_ctx(shared_ctx[-1])})")

    shared_ctx_set = set(shared_ctx)

    # ── Plot (a): Overall bars ───────────────────────────────────────────
    print("\n(a) Overall bar chart...")
    fig_a = plot_overall_bars(df_sn, df_tn, shared_models, shared_ctx_set)
    _savefig(fig_a, "two_needle_comparison.png")

    # ── Plot (b): Scaling curves ─────────────────────────────────────────
    print("\n(b) Scaling curves...")
    fig_b = plot_scaling_curves(df_sn, df_tn, shared_models, shared_ctx_set)
    _savefig(fig_b, "two_needle_scaling.png")

    # ── Plot (c): Secondary depth effect ─────────────────────────────────
    print("\n(c) Secondary depth effect...")
    fig_c = plot_secondary_depth(df_tn, shared_models)
    _savefig(fig_c, "two_needle_depth_effect.png")

    # ── Plot (d): Citation breakdown ─────────────────────────────────────
    print("\n(d) Citation breakdown...")
    fig_d = plot_citation_breakdown(df_tn, shared_models, shared_ctx_set)
    # Save directly: this figure uses its own custom font sizes, so we
    # bypass save_publication_figure / style_figure (which would override them).
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_d_path = PLOTS_DIR / "two_needle_citation_breakdown.pdf"
    fig_d.savefig(fig_d_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_d)
    print(f"  Saved: {fig_d_path}")

    # ── Summary CSV ──────────────────────────────────────────────────────
    # One row per model × context_length × secondary_depth
    csv_rows = []
    sn_filt = df_sn[df_sn["context_length"].isin(shared_ctx_set)]
    tn_filt = df_tn[df_tn["context_length"].isin(shared_ctx_set)]

    # Round secondary depths for grouping
    def _round_sd(d):
        for t in [0.0, 1/3, 2/3, 1.0]:
            if abs(d - t) < 0.02:
                return round(t, 4)
        return round(d, 4)

    tn_filt = tn_filt.copy()
    tn_filt["sec_depth_rounded"] = tn_filt["secondary_depth"].apply(_round_sd)
    sec_depths = sorted(tn_filt["sec_depth_rounded"].unique())
    depth_pct = {0.0: 0, round(1/3, 4): 33, round(2/3, 4): 67, 1.0: 100}

    for model in shared_models:
        sn_m = sn_filt[sn_filt["model"] == model]
        tn_m = tn_filt[tn_filt["model"] == model]
        for ctx in shared_ctx:
            sn_v = sn_m[sn_m["context_length"] == ctx]
            tn_v = tn_m[tn_m["context_length"] == ctx]

            # Skip rows where either SN or TN has no data for this context
            if len(sn_v) == 0 or len(tn_v) == 0:
                continue

            sn_ans = round(100 * sn_v["answer_correct"].mean(), 2)
            sn_both = round(100 * sn_v["both_correct"].mean(), 2)

            for sd in sec_depths:
                sd_sub = tn_v[tn_v["sec_depth_rounded"] == sd]
                if len(sd_sub) == 0:
                    continue

                csv_rows.append({
                    "model": dname(model),
                    "context_length": ctx,
                    "primary_depth": 50,
                    "secondary_depth": depth_pct.get(sd, int(round(sd * 100))),
                    "sn_answer": sn_ans,
                    "sn_both": sn_both,
                    "tn_answer": round(100 * sd_sub["answer_correct"].mean(), 2),
                    "tn_primary_cited": round(100 * sd_sub["primary_cited"].mean(), 2),
                    "tn_secondary_cited": round(100 * sd_sub["secondary_cited"].mean(), 2),
                    "tn_both": round(100 * sd_sub["both_correct"].mean(), 2),
                })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = TABLES_DIR / "two_needle_comparison.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}  ({len(csv_df)} rows)")

    # ── Console Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY (shared contexts only)")
    print("=" * 65)
    hdr = "  {:<18s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "Model", "SN-Ans", "SN-Both", "TN-Ans", "TN-Pri", "TN-Sec", "TN-Both")
    sep = "  " + "\u2500"*18 + " " + ("\u2500"*8 + " ")*6
    print(hdr)
    print(sep)

    for model in sorted(shared_models, key=dname):
        sn_a = 100 * sn_filt[sn_filt["model"] == model]["answer_correct"].mean()
        sn_b = 100 * sn_filt[sn_filt["model"] == model]["both_correct"].mean()
        tn_a = 100 * tn_filt[tn_filt["model"] == model]["answer_correct"].mean()
        tn_p = 100 * tn_filt[tn_filt["model"] == model]["primary_cited"].mean()
        tn_s = 100 * tn_filt[tn_filt["model"] == model]["secondary_cited"].mean()
        tn_b = 100 * tn_filt[tn_filt["model"] == model]["both_correct"].mean()
        print("  {:<18s} {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}%".format(
            dname(model), sn_a, sn_b, tn_a, tn_p, tn_s, tn_b))

    sn_a_all = 100 * sn_filt["answer_correct"].mean()
    sn_b_all = 100 * sn_filt["both_correct"].mean()
    tn_a_all = 100 * tn_filt["answer_correct"].mean()
    tn_p_all = 100 * tn_filt["primary_cited"].mean()
    tn_s_all = 100 * tn_filt["secondary_cited"].mean()
    tn_b_all = 100 * tn_filt["both_correct"].mean()
    print("\n  {:<18s} {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>6.1f}%".format(
        "Overall", sn_a_all, sn_b_all, tn_a_all, tn_p_all, tn_s_all, tn_b_all))
    print(f"\n  Data points: Single={len(sn_filt)}, Two-needle={len(tn_filt)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
