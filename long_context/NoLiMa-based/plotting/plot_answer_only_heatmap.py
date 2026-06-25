#!/usr/bin/env python3
"""
Ans* vs Ans Accuracy Delta Heatmap
==================================

Heatmap for all models across context lengths:
  Δ = Ans* accuracy − Ans accuracy

Aggregated across one-hop and two-hop.
Strict filters: Book 1, T01 only, 4 canonical depths, Yuki/Stuart characters.

Output: results/answer_only/plots/answer_only_delta_heatmap.pdf
"""

import json
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from plot_style import (
    TICK_FONTSIZE,
    LABEL_FONTSIZE,
    HEATMAP_NA_COLOR,
    HEATMAP_NA_TEXT,
    HEATMAP_NA_TEXT_COLOR,
    HEATMAP_CELL_FONTSIZE,
    HEATMAP_NA_FONTSIZE,
    apply_publication_style,
)

warnings.filterwarnings("ignore", category=FutureWarning)
apply_publication_style()

# ─── Configuration ────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT  = PROJECT_DIR / "evaluation"
SPECIAL_ROOT  = RESULTS_ROOT / "special_experiments"
REASONING_TYPE = "commonsense_knowledge"
OUTPUT_DIR = PROJECT_DIR / "results" / "answer_only"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

STRICT_BOOK   = "1"
STRICT_DEPTHS = {0.0, 0.33, 0.67, 1.0}
CANONICAL_TESTS_T01 = {
    "0402_T01_C02_onehop",   "0402_T01_C02_twohop",
    "0405_T01_C02_onehop",   "0405_T01_C02_twohop",
    "0402Inv_T01_C02_onehop","0402Inv_T01_C02_twohop",
    "0405Inv_T01_C02_onehop","0405Inv_T01_C02_twohop",
}
NEEDLE_CHARACTER = {
    "0402": "Yuki", "0402Inv": "Yuki",
    "0405": "Stuart","0405Inv": "Stuart",
}

# Preferred model row order and display names
MODEL_DISPLAY = {
    "gpt-4o":                    "GPT-4o",
    "gpt-4-1":                   "GPT-4.1",
    "o3-mini-2025-01-31":        "O3-mini",
    "gpt-5-2025-08-07":          "GPT-5",
    "claude-sonnet-4-20250514":  "Sonnet-4",
    "claude-sonnet-4-5-20250929":"Sonnet-4.5",
    "gemini-2-5-flash":          "Gemini-2.5-Fl",
    "gemini-3-flash-preview":    "Gemini-3-Fl",
}
MODEL_ORDER = list(MODEL_DISPLAY.keys())

# Context lengths to display as columns
DISPLAY_CTXS = [
    10_000, 20_000, 30_000, 40_000, 50_000,
    100_000, 200_000, 300_000, 400_000, 500_000,
    600_000, 700_000, 1_000_000,
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def fmt_ctx(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    return f"{n // 1_000}K"


def _model_dir(model_key: str) -> str:
    """Convert model key to filesystem directory name (dots → dashes)."""
    return model_key.replace(".", "-")


def _is_api_error(r: dict) -> bool:
    if r.get("error") or r.get("error_type"):
        return True
    if r.get("response") is None and r.get("input_tokens", 0) == 0:
        return True
    return False


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_condition(base_dir: Path, model_key: str, label: str) -> pd.DataFrame:
    """Load answer_correct for one model under one prompt condition.

    Filters: Book 1, T01 canonical tests, 4 standard depths, correct characters.
    Aggregates across one-hop and two-hop.
    """
    mdir = _model_dir(model_key)
    comm = base_dir / REASONING_TYPE
    if not comm.is_dir():
        return pd.DataFrame()

    rows = []
    for ctx_dir in sorted(comm.glob("rand_shuffle_*")):
        try:
            ctx_len = int(ctx_dir.name.split("_")[-1])
        except ValueError:
            continue

        for test_dir in ctx_dir.iterdir():
            if not test_dir.is_dir():
                continue
            tname = test_dir.name
            if tname not in CANONICAL_TESTS_T01:
                continue

            needle_id = re.match(r"^(0402Inv|0402|0405Inv|0405)_", tname)
            if not needle_id:
                continue
            expected_char = NEEDLE_CHARACTER.get(needle_id.group(1))

            book_file = test_dir / f"{mdir}_rand_book_{STRICT_BOOK}_{tname}.json"
            if not book_file.exists():
                continue

            try:
                data = json.loads(book_file.read_text())
            except Exception:
                continue

            for r in data.get("results", []):
                if _is_api_error(r):
                    continue
                depth = (r.get("placement_metadata") or {}).get("depth")
                if depth is None:
                    continue
                depth_r = round(float(depth), 2)
                if depth_r not in STRICT_DEPTHS:
                    continue
                char = r.get("selected_character", "")
                if expected_char and char != expected_char:
                    continue
                rows.append({
                    "context_length": ctx_len,
                    "answer_correct": int(r.get("answer_metric", 0) or 0),
                })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    print(f"  [{label}] {model_key}: {len(df)} rows, "
          f"{df['context_length'].nunique()} contexts")
    return df


def load_all_models() -> tuple[dict, dict]:
    """Return (std_data, ao_data): model_key -> DataFrame with answer_correct."""
    std_data = {}
    ao_data  = {}

    for model_key in MODEL_ORDER:
        mdir = _model_dir(model_key)

        # Standard (Answer+Evidence prompt)
        std_base = RESULTS_ROOT / f"results_{mdir}" / REASONING_TYPE
        # results_ dir may not exist for all keys — try with the raw key too
        if not std_base.parent.is_dir():
            # Some dirs might use the original dot-notation key
            alt = RESULTS_ROOT / f"results_{model_key}"
            if alt.is_dir():
                std_base = alt / REASONING_TYPE
        df_std = load_condition(std_base.parent, model_key, "Ans")
        if not df_std.empty:
            std_data[model_key] = df_std

        # Answer-Only prompt
        ao_base = SPECIAL_ROOT / f"results_{mdir}-answer-only"
        df_ao = load_condition(ao_base, model_key, "Ans*")
        if not df_ao.empty:
            ao_data[model_key] = df_ao

    return std_data, ao_data


# ─── Pivot Building ───────────────────────────────────────────────────────────

def build_pivot(data: dict, display_ctxs: list) -> pd.DataFrame:
    """Build model × context pivot of mean answer accuracy (%)."""
    rows = {}
    for model_key in MODEL_ORDER:
        if model_key not in data:
            rows[MODEL_DISPLAY[model_key]] = {fmt_ctx(c): np.nan for c in display_ctxs}
            continue
        df = data[model_key]
        ctx_means = (df.groupby("context_length")["answer_correct"].mean() * 100).to_dict()
        rows[MODEL_DISPLAY[model_key]] = {
            fmt_ctx(c): ctx_means.get(c, np.nan) for c in display_ctxs
        }
    return pd.DataFrame(rows, index=[fmt_ctx(c) for c in display_ctxs]).T


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_heatmap():
    print("Loading data …")
    std_data, ao_data = load_all_models()
    print(f"  Ans: {len(std_data)} models  |  Ans*: {len(ao_data)} models")

    pivot_std = build_pivot(std_data, DISPLAY_CTXS)
    pivot_ao  = build_pivot(ao_data,  DISPLAY_CTXS)

    n_rows = len(MODEL_ORDER)
    n_cols = len(DISPLAY_CTXS)

    # Delta pivot: Ans* − Ans (positive = Ans* better, negative = Ans better)
    pivot_delta = pivot_ao - pivot_std

    # Figure sizing: single heatmap + colorbar
    cell_w, cell_h = 0.82, 0.70
    fig_w = 1.8 + n_cols * cell_w + 1.4
    fig_h = 1.0 + n_rows * cell_h

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    gs = GridSpec(
        1, 2, figure=fig,
        width_ratios=[n_cols, 0.45],
        left=0.15, right=0.97,
        top=0.88, bottom=0.22,
        wspace=0.04,
    )
    ax_delta = fig.add_subplot(gs[0, 0])
    ax_cbar  = fig.add_subplot(gs[0, 1])

    # Diverging colormap centred at 0; empty (N/A) cells rendered gray
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(HEATMAP_NA_COLOR)
    abs_max = 50
    vmin, vmax = -abs_max, abs_max

    data = pivot_delta.values.astype(float)
    ax_delta.imshow(data, aspect="auto", cmap=cmap,
                    vmin=vmin, vmax=vmax, interpolation="nearest")

    # Grid lines
    ax_delta.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax_delta.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax_delta.grid(which="minor", color="white", linewidth=0.8)
    ax_delta.tick_params(which="minor", bottom=False, left=False)

    # Cell annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                ax_delta.text(j, i, HEATMAP_NA_TEXT, ha="center", va="center",
                              fontsize=HEATMAP_NA_FONTSIZE, color=HEATMAP_NA_TEXT_COLOR, style="italic")
            else:
                tc = "white" if abs(val) > 30 else "black"
                sign = "+" if val > 0 else ""
                ax_delta.text(j, i, f"{sign}{val:.0f}", ha="center", va="center",
                              fontsize=HEATMAP_CELL_FONTSIZE, color=tc, fontweight="semibold")

    ax_delta.set_xticks(range(n_cols))
    ax_delta.set_xticklabels(pivot_delta.columns.tolist(),
                             fontsize=TICK_FONTSIZE, rotation=45, ha="right",
                             fontweight="bold")
    ax_delta.set_yticks(range(n_rows))
    ax_delta.set_yticklabels(pivot_delta.index.tolist(), fontsize=TICK_FONTSIZE,
                             fontweight="bold")
    ax_delta.set_title("Δ Accuracy (Ans* − Ans)",
                       fontsize=LABEL_FONTSIZE-2, pad=10, fontweight="normal")

    # Colorbar
    cb = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)),
        cax=ax_cbar,
    )
    cb.set_label("Δ Accuracy (pp)", fontsize=TICK_FONTSIZE, labelpad=8)
    cb.ax.tick_params(labelsize=TICK_FONTSIZE - 4)
    cb.set_ticks([-40, -20, 0, 20, 40])

    # Save
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "answer_only_delta_heatmap.pdf"
    legacy_out_path = PLOTS_DIR / "answer_only_heatmap.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(legacy_out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✓ Saved → {out_path}")
    print(f"✓ Saved → {legacy_out_path}")

    # Console summary
    print("\n" + "=" * 65)
    print(f"  {'Model':<16} {'Ans':>9} {'Ans*':>9} {'Δ (Ans*−Ans)':>12}")
    print("  " + "─" * 50)
    for model_key in MODEL_ORDER:
        name = MODEL_DISPLAY[model_key]
        s = std_data.get(model_key)
        a = ao_data.get(model_key)
        s_mean = 100 * s["answer_correct"].mean() if s is not None and not s.empty else float("nan")
        a_mean = 100 * a["answer_correct"].mean() if a is not None and not a.empty else float("nan")
        delta  = a_mean - s_mean if not (np.isnan(s_mean) or np.isnan(a_mean)) else float("nan")
        print(f"  {name:<16} {s_mean:>8.1f}% {a_mean:>8.1f}% {delta:>+11.1f} pp")
    print("=" * 65)


if __name__ == "__main__":
    plot_heatmap()
