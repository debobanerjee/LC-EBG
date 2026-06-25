#!/usr/bin/env python3
"""
Plot Δ Accuracy heatmap:  Ans_LC  −  Ans_RAG

  Rows    : models
  Columns : context lengths (50K → 200K, topk chars)
  Cells   : mean LC ans_accuracy − mean RAG answer_accuracy  (integer %, signed)
  Color   : diverging RdBu — blue (LC > RAG) / red (LC < RAG) / white (0)
  Missing : shown as "—"

Sources:
  LC  — NoLiMa_based_RAG/tables/ans_accuracy_selected_contexts.csv
           ans_accuracy column is already 0–100
  RAG — NoLiMa_based_RAG/rag_experiment_results/tables/rag_results_raw.csv
           answer_metric column is binary 0/1  (×100 → %)

Both are averaged over all rows for a given (model, context_length) before
computing the delta.

Usage:
  python3 NoLiMa_based_RAG/scripts/plot_rag_lc_delta_heatmap.py
  python3 NoLiMa_based_RAG/scripts/plot_rag_lc_delta_heatmap.py \\
      --lc-csv  <path> --rag-csv <path> --out-dir <dir>
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

# ── Default paths ─────────────────────────────────────────────────────────────
LC_CSV  = Path("NoLiMa_based_RAG/tables/ans_accuracy_selected_contexts.csv")
RAG_CSV = Path("NoLiMa_based_RAG/rag_experiment_results/tables/rag_results_raw.csv")
OUT_DIR = Path("NoLiMa_based_RAG/plots")

# ── Context lengths shared between LC and RAG ─────────────────────────────────
# LC context_length values → display labels used in RAG topk_context_length
LC_CTX_TO_LABEL = {
    50_000:  "50K",
    70_000:  "70K",
    90_000:  "90K",
    100_000: "100K",
    200_000: "200K",
}
CONTEXT_ORDER = ["50K", "70K", "90K", "100K", "200K"]

# ── Model display order & name mapping ───────────────────────────────────────
# RAG CSV model IDs  →  display label
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
# LC CSV model strings  →  RAG model IDs
LC_MODEL_MAP = {
    "GPT-4o":    "gpt-4o",
    "GPT-4.1":   "gpt-4.1",
    "O3-mini":   "o3-mini-2025-01-31",
    "GPT-5":     "gpt-5-2025-08-07",
    "Sonnet-4":  "claude-sonnet-4-20250514",
    "Sonnet-4.5":"claude-sonnet-4-5-20250929",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_lc(path: Path) -> dict:
    """Return {(rag_model_id, ctx_label): mean_ans_accuracy_pct}."""
    sums, counts = defaultdict(float), defaultdict(int)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rag_id = LC_MODEL_MAP.get(row["model"].strip())
            label  = LC_CTX_TO_LABEL.get(int(row["context_length"]))
            if rag_id is None or label is None:
                continue
            sums[(rag_id, label)]   += float(row["ans_accuracy"])
            counts[(rag_id, label)] += 1
    return {k: sums[k] / counts[k] for k in sums}


def load_rag(path: Path) -> dict:
    """Return {(rag_model_id, ctx_label): mean_answer_accuracy_pct}."""
    sums, counts = defaultdict(float), defaultdict(int)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            model = row["model"].strip()
            label = row["topk_context_length"].strip()
            if model not in MODEL_ORDER or label not in CONTEXT_ORDER:
                continue
            sums[(model, label)]   += float(row["answer_metric"]) * 100
            counts[(model, label)] += 1
    return {k: sums[k] / counts[k] for k in sums}


def build_delta(lc: dict, rag: dict):
    """Return (matrix, present) of shape (n_model, n_ctx).
    delta[i,j] = LC_acc − RAG_acc, rounded to nearest integer.
    """
    n_m, n_c = len(MODEL_ORDER), len(CONTEXT_ORDER)
    matrix  = np.full((n_m, n_c), np.nan)
    present = np.zeros((n_m, n_c), dtype=bool)
    for i, m in enumerate(MODEL_ORDER):
        for j, c in enumerate(CONTEXT_ORDER):
            k = (m, c)
            if k in lc and k in rag:
                matrix[i, j]  = round(lc[k] - rag[k])
                present[i, j] = True
    return matrix, present


# ── Plot helpers ──────────────────────────────────────────────────────────────

def text_color(val: float, vabs: float) -> str:
    """White for strongly saturated cells, near-black for pale cells."""
    return "white" if abs(val) / max(vabs, 1e-9) > 0.55 else "#1a1a1a"


def fmt(val: int) -> str:
    return f"+{val}" if val > 0 else ("0" if val == 0 else str(val))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Plot Δ Accuracy (LC − RAG) heatmap")
    p.add_argument("--lc-csv",  default=str(LC_CSV))
    p.add_argument("--rag-csv", default=str(RAG_CSV))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lc  = load_lc(Path(args.lc_csv))
    rag = load_rag(Path(args.rag_csv))
    matrix, present = build_delta(lc, rag)

    n_m, n_c = len(MODEL_ORDER), len(CONTEXT_ORDER)

    # ── Figure ────────────────────────────────────────────────────────────────
    cell_w, cell_h = 1.1, 0.72
    fig_w = n_c * cell_w + 2.6   # extra right margin for colorbar
    fig_h = n_m * cell_h + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── Diverging colormap RdBu (red=neg, white=0, blue=pos) ──────────────────
    vabs = 50
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color="#e8e8e8")

    masked = np.where(present, matrix, np.nan)
    im = ax.imshow(masked, cmap=cmap, vmin=-vabs, vmax=vabs, aspect="auto")

    # ── Cell annotations ──────────────────────────────────────────────────────
    for i in range(n_m):
        for j in range(n_c):
            if present[i, j]:
                val = int(matrix[i, j])
                ax.text(j, i, fmt(val), ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color=text_color(val, vabs))
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=13, color="#aaaaaa")

    # ── X-axis: context lengths (bottom) ──────────────────────────────────────
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(CONTEXT_ORDER, fontsize=13, fontweight="bold")
    ax.set_xlabel("Context Length", fontsize=14, labelpad=8)

    # ── Y-axis: model names ───────────────────────────────────────────────────
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(
        [MODEL_LABELS[m] for m in MODEL_ORDER],
        fontsize=13, fontweight="bold",
    )

    # ── White grid lines between cells ────────────────────────────────────────
    ax.set_xticks(np.arange(n_c) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_m) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(r"$\Delta$ Accuracy (pp)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        r"$\Delta$ Accuracy (Ans$_{LC}$ $-$ Ans$_{RAG}$)",
        fontsize=15, fontweight="bold", pad=10,
    )

    fig.tight_layout()

    out_path = out_dir / "rag_lc_delta_heatmap.pdf"
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
