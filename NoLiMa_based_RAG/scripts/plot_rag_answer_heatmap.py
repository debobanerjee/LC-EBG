#!/usr/bin/env python3
"""
Plot a heatmap of RAG answer accuracy (answer_metric) from rag_results_raw.csv.

  X-axis : models
  Y-axis : haystack sizes (topk_context_length labels)
  Cells  : mean answer_metric × 100  (integer %)
  Missing: shown as "—"

Output: NoLiMa_based_RAG/plots/rag_answer_heatmap.pdf

Usage:
  python3 NoLiMa_based_RAG/scripts/plot_rag_answer_heatmap.py
  python3 NoLiMa_based_RAG/scripts/plot_rag_answer_heatmap.py --input-csv <path> --out-dir <dir>
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed fonts (ACL requirement)
matplotlib.rcParams["ps.fonttype"]  = 42
import matplotlib.pyplot as plt
import numpy as np

# ── Data paths ────────────────────────────────────────────────────────────────
INPUT_CSV = Path("NoLiMa_based_RAG/rag_experiment_results/tables/rag_results_raw.csv")
OUT_DIR   = Path("NoLiMa_based_RAG/plots")

# ── Display order & labels ────────────────────────────────────────────────────
MODEL_ORDER = [
    "gpt-4o",
    "gpt-4.1",
    "o3-mini-2025-01-31",
    "gpt-5-2025-08-07",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
]
MODEL_LABELS = {
    "gpt-4o":                     "GPT-4o",
    "gpt-4.1":                    "GPT-4.1",
    "o3-mini-2025-01-31":         "O3-mini",
    "gpt-5-2025-08-07":           "GPT-5",
    "claude-sonnet-4-20250514":   "Sonnet-4",
    "claude-sonnet-4-5-20250929": "Sonnet-4.5",
}

CONTEXT_ORDER = ["50K", "70K", "90K", "100K", "200K"]   # top → bottom on Y-axis


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_matrix(csv_path: Path):
    """Return (matrix, present) numpy arrays of shape (n_model, n_ctx).

    matrix[i, j] = mean answer_metric * 100 for model i, context j
    present[i, j] = True if data exists
    """
    sums   = defaultdict(float)
    counts = defaultdict(int)

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            ctx   = row["topk_context_length"].strip()
            model = row["model"].strip()
            val   = float(row["answer_metric"])
            sums[(model, ctx)]   += val
            counts[(model, ctx)] += 1

    n_model = len(MODEL_ORDER)
    n_ctx   = len(CONTEXT_ORDER)
    matrix  = np.full((n_model, n_ctx), np.nan)
    present = np.zeros((n_model, n_ctx), dtype=bool)

    for i, model in enumerate(MODEL_ORDER):
        for j, ctx in enumerate(CONTEXT_ORDER):
            if counts[(model, ctx)] > 0:
                matrix[i, j]  = round(sums[(model, ctx)] / counts[(model, ctx)] * 100)
                present[i, j] = True

    return matrix, present


def cell_text_color(value: float, vmin: float, vmax: float) -> str:
    """Return white for dark cells, near-black for light cells."""
    norm = (value - vmin) / max(vmax - vmin, 1e-9)
    return "white" if norm > 0.55 else "#1a1a1a"


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Plot RAG answer accuracy heatmap")
    p.add_argument("--input-csv", default=str(INPUT_CSV))
    p.add_argument("--out-dir",   default=str(OUT_DIR))
    p.add_argument("--title",     default="RAG Ans (Ans+Evid prompt)",
                   help="Plot title (use empty string to suppress)")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix, present = load_matrix(Path(args.input_csv))

    n_model = len(MODEL_ORDER)
    n_ctx   = len(CONTEXT_ORDER)

    # ── Figure layout: rows=models, cols=context lengths ──────────────────────
    cell_w, cell_h = 1.1, 0.72
    fig_w = n_ctx   * cell_w + 2.2   # left margin for model name labels
    fig_h = n_model * cell_h + 1.2   # top margin for title + x-labels

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── Colormap: Blues (NaN → light grey for missing cells) ──────────────────
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#e8e8e8")

    vmin, vmax = 0, 100
    masked = np.where(present, matrix, np.nan)
    ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # ── Cell annotations ──────────────────────────────────────────────────────
    for i in range(n_model):
        for j in range(n_ctx):
            if present[i, j]:
                val = int(matrix[i, j])
                fc  = cell_text_color(val, vmin, vmax)
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=13, fontweight="bold", color=fc)
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=13, color="#aaaaaa")

    # ── X-axis: context lengths (bottom) ──────────────────────────────────────
    ax.set_xticks(range(n_ctx))
    ax.set_xticklabels(CONTEXT_ORDER, fontsize=13, fontweight="bold")
    ax.set_xlabel("Context Length", fontsize=14, labelpad=8)

    # ── Y-axis: model names ───────────────────────────────────────────────────
    ax.set_yticks(range(n_model))
    ax.set_yticklabels(
        [MODEL_LABELS.get(m, m) for m in MODEL_ORDER],
        fontsize=13, fontweight="bold",
    )

    # ── Grid lines between cells ──────────────────────────────────────────────
    ax.set_xticks(np.arange(n_ctx)   - 0.5, minor=True)
    ax.set_yticks(np.arange(n_model) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    # ── Title ─────────────────────────────────────────────────────────────────
    if args.title:
        ax.set_title(args.title, fontsize=15, fontweight="bold", pad=10)

    fig.tight_layout()

    out_path = out_dir / "rag_answer_heatmap.pdf"
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
