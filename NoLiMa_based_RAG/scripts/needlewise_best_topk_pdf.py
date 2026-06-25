#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

# =============================================================================
# COLOR PALETTE
# Single sequential ColorBrewer YlOrBr ramp — same for ALL needles.
# Low value → light yellow, high value → dark brown-orange.
# Edit the four stops below to restyle globally.
# =============================================================================
# SEQ_RAMP = [
#     "#ffffd4",   # stop 0  — very low  (light yellow)
#     "#fed98e",   # stop 1  — low-mid   (pale orange)
#     "#fe9929",   # stop 2  — mid-high  (orange)
#     "#cc4c02",   # stop 3  — very high (dark brown-orange)
# ]

SEQ_RAMP = [
    "#f1eef6",   # stop 0  — very low  (very light blue)
    "#bdc9e1",   # stop 1  — low-mid   (pale blue)
    "#74a9cf",   # stop 2  — mid-high  (medium blue)
    "#0570b0",   # stop 3  — very high (dark blue)
]

# t (log-normalised 0→1) above which cell text switches to white
HEATMAP_TEXT_THRESHOLD = 0.50

COLOR_LABEL_TEXT    = "#222222"
COLOR_ANNOTATION_BG = "#FFFFFF"

# =============================================================================
# ACL font sizes
# =============================================================================
LABEL_FONT_SIZE   = 18   # xlabel / ylabel
TICK_FONT_SIZE    = 12   # x/y tick labels (compact for heatmap)
NEEDLE_LABEL_SIZE = 12   # y-tick needle names
CELL_FONT_SIZE    = 9    # value annotations inside cells
TITLE_FONT_SIZE   = 13   # "One-hop" / "Two-hop" panel title above each panel

# =============================================================================
# ACL COLUMN GUIDANCE
# figsize=(7.0, 4.5) → DOUBLE column.
# For single column change to figsize=(3.5, 3.5).
# =============================================================================
FIGSIZE = (7.0, 5.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ACL-ready heatmap PDF: required top-k per needle × context."
    )
    p.add_argument(
        "--input-csv",
        default="NoLiMa_based_RAG/tables/rag_accuracy_full_binary_from_results_summary_full.csv",
    )
    p.add_argument("--out-dir",  default="NoLiMa_based_RAG/plots")
    p.add_argument("--out-name", default="needle_topk_requirement_heatmap.pdf")
    p.add_argument(
        "--selected-contexts",
        default="200000,400000,600000,800000,1000000",
        help="Comma-separated context lengths to include.",
    )
    p.add_argument(
        "--target-retrieval", type=float, default=1.0,
        help="Target retrieval rate (0–1) for required top-k.",
    )
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


# =============================================================================
# Utilities
# =============================================================================
def to_f(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def to_i(v: str) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0


def fmt_ctx(value: int) -> str:
    if value >= 1_000_000:
        return f"{value // 1_000_000}M" if value % 1_000_000 == 0 else f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value // 1_000}k" if value % 1_000 == 0 else f"{value / 1_000:.1f}k"
    return str(value)


def mean(vs: list[float]) -> float:
    return sum(vs) / len(vs) if vs else 0.0


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)


def _seq_color(t: float) -> tuple[float, float, float]:
    """
    Map t in [0, 1] through the 4-stop sequential ramp SEQ_RAMP.
    Segments: [0,1/3], [1/3,2/3], [2/3,1]
    """
    n_segs = len(SEQ_RAMP) - 1          # 3 segments for 4 stops
    seg    = min(int(t * n_segs), n_segs - 1)
    local_t = t * n_segs - seg           # 0→1 within this segment
    r0, g0, b0 = _hex_to_rgb(SEQ_RAMP[seg])
    r1, g1, b1 = _hex_to_rgb(SEQ_RAMP[seg + 1])
    return (r0 + local_t * (r1 - r0),
            g0 + local_t * (g1 - g0),
            b0 + local_t * (b1 - b0))


# =============================================================================
# Plot
# =============================================================================
def plot_heatmap(
    req: dict[tuple[str, str], dict[int, int]],
    hops: list[str],
    needles: list[str],
    contexts: list[int],
    out_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import matplotlib.cm as mcm
    import numpy as np

    # Global log-scale normalisation across ALL hops / needles / contexts
    all_vals = [
        req.get((hop, nd), {}).get(ctx, 0)
        for hop in hops for nd in needles for ctx in contexts
    ]
    nz        = [v for v in all_vals if v > 0]
    log_min   = np.log10(min(nz)) if nz else 0.0
    log_max   = np.log10(max(nz)) if nz else 1.0
    log_range = max(log_max - log_min, 1e-9)

    n_nd  = len(needles)
    n_ctx = len(contexts)

    fig, axs = plt.subplots(
        1, 2,
        figsize=FIGSIZE,
        gridspec_kw={"wspace": 0.30},
    )
    # Extra top margin for panel titles, extra bottom for rotated x-ticks
    fig.subplots_adjust(top=0.82, bottom=0.30)

    for pidx, hop in enumerate(hops):
        ax = axs[pidx]
        ax.set_xlim(-0.5, n_ctx - 0.5)
        ax.set_ylim(-0.5, n_nd  - 0.5)

        for ri, nd in enumerate(needles):
            for ci, ctx in enumerate(contexts):
                val = req.get((hop, nd), {}).get(ctx, 0)

                # Normalise on log scale → t in [0, 1]
                t = float(np.clip(
                    (np.log10(max(val, 1)) - log_min) / log_range,
                    0.0, 1.0,
                ))

                # Map through sequential ramp (same for all needles)
                fc = _seq_color(t)

                rect = mpatches.FancyBboxPatch(
                    (ci - 0.46, ri - 0.44), 0.92, 0.88,
                    boxstyle="round,pad=0.04",
                    linewidth=0,
                    facecolor=fc,
                )
                ax.add_patch(rect)

                txt_col = "white" if t > HEATMAP_TEXT_THRESHOLD else COLOR_LABEL_TEXT
                ax.text(
                    ci, ri,
                    str(val) if val > 0 else "–",
                    ha="center", va="center",
                    fontsize=CELL_FONT_SIZE,
                    fontweight="bold",
                    color=txt_col,
                )

        # X-axis ticks
        ax.set_xticks(range(n_ctx))
        ax.set_xticklabels(
            [fmt_ctx(c) for c in contexts],
            rotation=45, ha="right",
            fontsize=TICK_FONT_SIZE,
        )

        # Y-axis ticks: only show labels on LEFT panel (pidx=0)
        ax.set_yticks(range(n_nd))
        if pidx == 0:
            # Left panel: show needle labels
            ax.set_yticklabels(needles, fontsize=NEEDLE_LABEL_SIZE)
        else:
            # Right panel: hide y-tick labels
            ax.set_yticklabels([])

        ax.tick_params(length=0)

        # Remove spines for clean look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Panel title ABOVE the axes (outside the heatmap cells)
        ax.set_title(
            "One-hop" if hop == "onehop" else "Two-hop",
            fontsize=TITLE_FONT_SIZE,
            fontweight="normal",
            pad=12,
        )

    # Add shared axis label ONCE using fig.text (positioned below)
    # Only "Context length" — needle labels appear once on the left panel only
    # fig.text(0.5, 0.05, "Context length", ha="center", fontsize=LABEL_FONT_SIZE, fontweight="normal")

    # Colorbar to show the sequential scale
    # Build a simple ScalarMappable from the ramp for the colorbar
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "seq_ramp",
        [_hex_to_rgb(c) for c in SEQ_RAMP],
    )
    norm = mcolors.LogNorm(
        vmin=10 ** log_min,
        vmax=10 ** log_max,
    )
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, shrink=0.75, pad=0.02, aspect=20)
    cbar.set_label("Required top-k", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(out_path, dpi=max(300, dpi), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}  [DOUBLE column]")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args         = parse_args()
    selected_ctx = {x.strip() for x in args.selected_contexts.split(",") if x.strip()}
    target       = max(0.0, min(1.0, args.target_retrieval))
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42   # Type-1 fonts (ACL requirement)
        matplotlib.rcParams["ps.fonttype"]  = 42
    except Exception as e:
        raise RuntimeError("pip install matplotlib") from e

    # Load CSV and compute required top-k per (hop, needle, context)
    retr_series: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    with Path(args.input_csv).open(newline="") as fh:
        for row in csv.DictReader(fh):
            hop    = row["reasoning_hop"].strip()
            needle = row["needle_id"].strip()
            ctx    = row["context_length"].strip()
            topk   = to_i(row["topk"])
            hit    = to_f(row["retrieval_hit_at_topk"])
            retr_series[(hop, needle, ctx)][topk].append(hit)

    hops    = ["onehop", "twohop"]
    needles = sorted({k[1] for k in retr_series.keys()})

    req: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
    for (hop, needle, ctx), topk_map in retr_series.items():
        if selected_ctx and ctx not in selected_ctx:
            continue
        series = {tk: mean(vs) for tk, vs in topk_map.items()}
        if not series:
            continue
        topks         = sorted(series)
        required_topk = next((v for v in topks if series[v] >= target), topks[-1])
        req[(hop, needle)][to_i(ctx)] = required_topk

    contexts = sorted(
        {ctx for m in req.values() for ctx in m}
        | {to_i(c) for c in selected_ctx}
    )
    contexts = [c for c in contexts if c > 0]

    plot_heatmap(
        req=req,
        hops=hops,
        needles=needles,
        contexts=contexts,
        out_path=out_dir / args.out_name,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
