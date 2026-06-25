#!/usr/bin/env python3
"""
Side-by-side heatmap: Ans+Evid accuracy for all models
across context lengths, split by One-hop / Two-hop.

Output: results/standard/plots/combined_accuracy_heatmap.pdf
"""

import json
import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT  = PROJECT_DIR / "evaluation"
REASONING_TYPE = "commonsense_knowledge"
OUTPUT_DIR = PROJECT_DIR / "results" / "standard"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"
SKIP_DIRS     = {"DEPRECATED_results_claude-3-7-sonnet-20250219"}

STRICT_BOOK   = "1"
STRICT_DEPTHS = {0.0, 0.33, 0.67, 1.0}
CANONICAL_TESTS_T01 = {
    "0402_T01_C02_onehop",  "0402_T01_C02_twohop",
    "0405_T01_C02_onehop",  "0405_T01_C02_twohop",
    "0402Inv_T01_C02_onehop","0402Inv_T01_C02_twohop",
    "0405Inv_T01_C02_onehop","0405Inv_T01_C02_twohop",
}
NEEDLE_CHARACTER = {
    "0402": "Yuki", "0402Inv": "Yuki",
    "0405": "Stuart","0405Inv": "Stuart",
}

# Context lengths to show (columns); models without data get NaN
DISPLAY_CTXS = [
    10_000, 20_000, 30_000, 40_000, 50_000,
    100_000, 200_000, 300_000, 400_000, 500_000,
    600_000, 700_000, 1_000_000,
]

# Model display names and preferred row order
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


def fmt_ctx(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    return f"{n // 1_000}K"


# ── Data loading (mirrors research_analysis.load_raw + strict filters) ────────

def _is_api_error(r: dict) -> bool:
    if r.get("error") or r.get("error_type"):
        return True
    if r.get("response") is None and r.get("input_tokens", 0) == 0:
        return True
    return False


def _reparse(resp) -> dict | None:
    if resp is None:
        return None
    if isinstance(resp, dict):
        return resp
    if not isinstance(resp, str) or len(resp.strip()) < 3:
        return None
    text = re.sub(r",\s*}", "}", resp)
    text = re.sub(r",\s*]", "]", text)
    for pat in [
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r'(\{[^{}]*"answer"[^{}]*\})',
        r"(\{.*\})",
    ]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(re.sub(r",\s*}", "}", m.group(1)))
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def _score(r: dict, reparsed: dict | None):
    if reparsed is None or "answer" not in reparsed:
        return int(r.get("answer_metric", 0) or 0), int(r.get("evidence_metric", 0) or 0)
    char = r.get("selected_character", "")
    line = (r.get("placement_metadata") or {}).get("needle_line_num")
    ans = int(char.lower() in str(reparsed.get("answer", "")).lower()) if char else 0
    lines = reparsed.get("lines", [])
    if not isinstance(lines, list):
        lines = [lines] if lines is not None else []
    int_lines = []
    for ln in lines:
        try:
            int_lines.append(int(ln))
        except (ValueError, TypeError):
            pass
    evi = int(line is not None and line in int_lines)
    return ans, evi


def load_data() -> pd.DataFrame:
    rows = []
    for model_dir in sorted(RESULTS_ROOT.glob("results_*")):
        if model_dir.name in SKIP_DIRS:
            continue
        model = model_dir.name.replace("results_", "")
        comm = model_dir / REASONING_TYPE
        if not comm.is_dir():
            continue
        for ctx_dir in sorted(comm.glob("rand_shuffle_*")):
            try:
                ctx_len = int(ctx_dir.name.split("_")[-1])
            except ValueError:
                continue
            for test_dir in ctx_dir.iterdir():
                if not test_dir.is_dir():
                    continue
                tname = test_dir.name
                if tname.endswith("onehop"):
                    hop = "onehop"
                elif tname.endswith("twohop"):
                    hop = "twohop"
                else:
                    continue
                for jf in test_dir.glob("*.json"):
                    try:
                        data = json.loads(jf.read_text())
                    except Exception:
                        continue
                    bm = re.search(r"rand_book_(\d+)", jf.stem)
                    book = bm.group(1) if bm else "?"
                    for r in data.get("results", []):
                        if _is_api_error(r):
                            continue
                        resp = r.get("response")
                        reparsed = _reparse(resp) if isinstance(resp, str) else resp
                        ans, evi = _score(r, reparsed)
                        depth = (r.get("placement_metadata") or {}).get("depth")
                        rows.append({
                            "model": model,
                            "context_length": ctx_len,
                            "hop": hop,
                            "test_name": tname,
                            "book": book,
                            "depth": float(depth) if depth is not None else np.nan,
                            "character": r.get("selected_character", ""),
                            "both_correct": 1 if (ans == 1 and evi == 1) else 0,
                        })
    if not rows:
        sys.exit("No results found.")
    return pd.DataFrame(rows)


def apply_strict_filters(df: pd.DataFrame) -> pd.DataFrame:
    # T01 canonical tests only
    df = df[df["test_name"].isin(CANONICAL_TESTS_T01)].copy()
    # Book 1 only
    df = df[df["book"] == STRICT_BOOK].copy()
    # Standard depths only
    df = df.dropna(subset=["depth"])
    df = df[df["depth"].round(2).isin(STRICT_DEPTHS)].copy()
    # Canonical characters
    needle_id = df["test_name"].str.extract(r"^(0402Inv|0402|0405Inv|0405)_", expand=False)
    expected = needle_id.map(NEEDLE_CHARACTER)
    df = df[expected.isna() | (df["character"] == expected)].copy()
    return df


# ── Plot ──────────────────────────────────────────────────────────────────────

def build_pivot(df: pd.DataFrame, hop: str) -> pd.DataFrame:
    sub = df[df["hop"] == hop]
    pivot = sub.pivot_table(
        index="model", columns="context_length",
        values="both_correct", aggfunc=lambda x: 100.0 * x.mean()
    )
    # Re-index to desired model order and context lengths
    models_present = [m for m in MODEL_ORDER if m in pivot.index]
    ctx_present    = [c for c in DISPLAY_CTXS if c in pivot.columns]
    return pivot.reindex(index=models_present, columns=ctx_present)


def plot_heatmap():
    print("Loading results …")
    df = load_data()
    df = apply_strict_filters(df)
    print(f"  {len(df)} depth-level rows after filtering")

    pivot_one  = build_pivot(df, "onehop")
    pivot_two  = build_pivot(df, "twohop")

    # Consistent model list (union, in preferred order)
    all_models = [m for m in MODEL_ORDER
                  if m in pivot_one.index or m in pivot_two.index]
    pivot_one  = pivot_one.reindex(index=all_models)
    pivot_two  = pivot_two.reindex(index=all_models)

    display_labels = [MODEL_DISPLAY.get(m, m) for m in all_models]
    ctx_labels_one = [fmt_ctx(c) for c in pivot_one.columns]
    ctx_labels_two = [fmt_ctx(c) for c in pivot_two.columns]

    # ── Figure layout ──────────────────────────────────────────────────────
    n_rows = len(all_models)
    n_cols_one = len(pivot_one.columns)
    n_cols_two = len(pivot_two.columns)

    # Two equal-width heatmap panels + narrow colorbar column
    fig_w = 2.0 + n_cols_one * 0.72 + 0.4 + n_cols_two * 0.72 + 1.0
    fig_h = 1.2 + n_rows * 0.62

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

    # GridSpec: [heatmap_one | gap | heatmap_two | cbar]
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        1, 4,
        figure=fig,
        width_ratios=[n_cols_one, 0.08, n_cols_two, 0.35],
        left=0.10, right=0.97,
        top=0.88,  bottom=0.18,
        wspace=0.02,
    )
    ax_one  = fig.add_subplot(gs[0, 0])
    ax_two  = fig.add_subplot(gs[0, 2])
    ax_cbar = fig.add_subplot(gs[0, 3])

    # Blue colormap matching the reference figure
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(HEATMAP_NA_COLOR)

    vmin, vmax = 0, 100

    def draw_panel(ax, pivot, xlabels, title):
        data = pivot.values.astype(float)
        im = ax.imshow(data, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")

        # White grid lines
        ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Cell annotations
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    ax.text(j, i, HEATMAP_NA_TEXT, ha="center", va="center",
                            fontsize=HEATMAP_NA_FONTSIZE, color=HEATMAP_NA_TEXT_COLOR, style="italic")
                else:
                    # White text on dark cells, black on light
                    tc = "white" if val > 55 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=HEATMAP_CELL_FONTSIZE, color=tc, fontweight="semibold")

        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(xlabels, fontsize=TICK_FONTSIZE, rotation=45, ha="right", fontweight="bold")
        ax.set_title(title, fontsize=LABEL_FONTSIZE, pad=8)
        return im

    im = draw_panel(ax_one, pivot_one, ctx_labels_one, "One-hop")
    draw_panel(ax_two, pivot_two, ctx_labels_two, "Two-hop")

    # Y-axis: model labels on left panel only
    ax_one.set_yticks(range(len(display_labels)))
    ax_one.set_yticklabels(display_labels, fontsize=TICK_FONTSIZE, fontweight="bold")
    ax_two.set_yticks(range(len(display_labels)))
    ax_two.set_yticklabels([], fontsize=TICK_FONTSIZE)

    # Colorbar
    cb = plt.colorbar(
        plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=vmin, vmax=vmax)
        ),
        cax=ax_cbar,
    )
    cb.set_label("Ans+Evid Accuracy (%)", fontsize=LABEL_FONTSIZE, labelpad=8)
    cb.ax.tick_params(labelsize=TICK_FONTSIZE)
    cb.set_ticks([0, 20, 40, 60, 80, 100])
    fig.suptitle("Ans+Evid Accuracy",
                 fontsize=LABEL_FONTSIZE + 2, fontweight="normal", y=0.98)

    # ── Save ──────────────────────────────────────────────────────────────
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "combined_accuracy_heatmap.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✓ Saved → {out_path}")


if __name__ == "__main__":
    plot_heatmap()
