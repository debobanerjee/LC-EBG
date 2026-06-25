#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ragwithtopk.embeddings.openai_embedder import OpenAIEmbedder

# =============================================================================
# COLOR PALETTES
# Edit these hex codes later if you want a different paper style.
# You can change any individual color here without touching the plotting logic.
# =============================================================================
SIMILARITY_BAR_COLORS = [
    "#08306B",
    "#08519C",
    "#2171B5",
    "#4292C6",
    "#6BAED6",
    "#9ECAE1",
    "#C6DBEF",
    "#D5E7F7",
]

COMPARISON_BAR_COLORS = {
    "original": "#4C78A8",
    "rephrased": "#F58518",
}

ANNOTATION_COLOR = "#333333"
GRID_COLOR = "#BDBDBD"

TICK_FONT_SIZE = 16
LABEL_FONT_SIZE = 18
LEGEND_FONT_SIZE = 12
DEFAULT_DPI = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ACL-ready PDF plots for character-question similarity analysis."
    )
    parser.add_argument(
        "--out-dir",
        default="NoLiMa_based_RAG/plots",
        help="Output directory for PDF plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="PDF DPI. Minimum recommended: 300.",
    )
    return parser.parse_args()


def cosine_similarity_matrix(query_vec: np.ndarray, doc_mat: np.ndarray) -> np.ndarray:
    """Return cosine similarity between one query vector and N document vectors."""
    query = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    docs = doc_mat / (np.linalg.norm(doc_mat, axis=1, keepdims=True) + 1e-12)
    return docs @ query


def apply_acl_tick_style(ax) -> None:
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)


def shorten_sentence(sentence: str, limit: int = 58) -> str:
    return sentence if len(sentence) <= limit else f"{sentence[:limit - 3]}..."


def extract_name(sentence: str) -> str:
    parts = sentence.split()
    if len(parts) > 1 and parts[0] == "Then":
        return parts[1]
    return sentence[:20]


# COLUMN FIT:
# - `figsize=(7.0, 4.8)` is intended for DOUBLE column.
# - Needle text is long, so horizontal bars are used for readability.
def plot_similarity_scores(
    question: str,
    needles: List[str],
    sims: np.ndarray,
    out_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    order = np.argsort(-sims)
    labels = [shorten_sentence(needles[i]) for i in order]
    values = sims[order]
    cmap = LinearSegmentedColormap.from_list("similarity_blues", SIMILARITY_BAR_COLORS)
    if len(order) == 1:
        colors = [SIMILARITY_BAR_COLORS[0]]
    else:
        positions = np.linspace(0.0, 1.0, len(order))
        colors = [to_hex(cmap(pos)) for pos in positions]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    y_pos = np.arange(len(order))
    bars = ax.barh(y_pos, values, color=colors)

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color=ANNOTATION_COLOR,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Similarity Score", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Needle", fontsize=LABEL_FONT_SIZE)
    ax.set_xlim(0, 0.25)
    apply_acl_tick_style(ax)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color=GRID_COLOR)

    fig.tight_layout()
    fig.savefig(out_path, dpi=max(DEFAULT_DPI, dpi), format="pdf", bbox_inches="tight")
    plt.close(fig)


# COLUMN FIT:
# - `figsize=(7.0, 4.6)` is intended for DOUBLE column.
# - Model/character labels are short enough for vertical grouped bars.
def plot_rephrased_similarity_comparison(
    needles: List[str],
    new_needles: List[str],
    sims_original: np.ndarray,
    sims_rephrased: np.ndarray,
    out_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    pair_count = min(len(needles), len(new_needles), len(sims_original), len(sims_rephrased))
    if pair_count == 0:
        return

    labels = [extract_name(needles[i]) for i in range(pair_count)]
    original_vals = np.array(sims_original[:pair_count], dtype=np.float32)
    rephrased_vals = np.array(sims_rephrased[:pair_count], dtype=np.float32)

    x = np.arange(pair_count, dtype=np.float32)
    bar_width = 0.38

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bars_original = ax.bar(
        x - bar_width / 2,
        original_vals,
        width=bar_width,
        color=COMPARISON_BAR_COLORS["original"],
        label="Original",
    )
    bars_rephrased = ax.bar(
        x + bar_width / 2,
        rephrased_vals,
        width=bar_width,
        color=COMPARISON_BAR_COLORS["rephrased"],
        label="Rephrased",
    )

    for i, (value_original, value_rephrased) in enumerate(zip(original_vals, rephrased_vals)):
        pair_top = max(float(value_original), float(value_rephrased))
        delta = float(value_rephrased) - float(value_original)
        ax.annotate(
            f"{delta:+.3f}",
            xy=(x[i], pair_top),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=ANNOTATION_COLOR,
            bbox=dict(boxstyle="round,pad=0.14", fc="#ffffff", ec="none", alpha=0.9),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Similarity Score", fontsize=LABEL_FONT_SIZE)
    ax.set_xlabel("Needle Pair", fontsize=LABEL_FONT_SIZE)
    ax.set_ylim(0, 0.3)
    apply_acl_tick_style(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color=GRID_COLOR)
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=max(DEFAULT_DPI, dpi), format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY env var")

    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
    except Exception as exc:
        raise RuntimeError("matplotlib is required. Install with: pip install matplotlib") from exc

    embedder = OpenAIEmbedder(api_key=api_key, model="text-embedding-3-small", batch_size=128)

    needles = [
        "Then Yuki mentioned that he has been vegan for years.",
        "Then Stuart mentioned that he has been vegan for years.",
        "Then Katie mentioned that he has been vegan for years.",
        "Then Veronica mentioned that he has been vegan for years.",
        "Then Gary mentioned that he has been vegan for years.",
        "Then Megan mentioned that he has been vegan for years.",
        "Then Calvin mentioned that he has been vegan for years.",
        "Then Mandy mentioned that he has been vegan for years.",
        "Then Diana mentioned that he has been vegan for years.",
        "Then Caxleb mentioned that he has been vegan for years.",
    ]
    new_needles = [
        "Then Yuki mentioned being vegan for years.",
        "Then Stuart mentioned being vegan for years.",
        "Then Katie mentioned being vegan for years.",
        "Then Veronica mentioned being vegan for years.",
        "Then Gary mentioned being vegan for years.",
        "Then Megan mentioned being vegan for years.",
        "Then Calvin mentioned being vegan for years.",
        "Then Mandy mentioned being vegan for years.",
        "Then Diana mentioned being vegan for years.",
        "Then Caxleb mentioned being vegan for years.",
    ]
    questions = [
        "Which character cannot eat Brandade?",
    ]

    needle_mat = np.array(embedder.embed_texts(needles), dtype=np.float32)
    new_needle_mat = np.array(embedder.embed_texts(new_needles), dtype=np.float32)

    for question in questions:
        query_vec = np.array(embedder.embed_texts([question])[0], dtype=np.float32)
        sims = cosine_similarity_matrix(query_vec, needle_mat)
        new_sims = cosine_similarity_matrix(query_vec, new_needle_mat)

        print(f"\nQUESTION: {question}")
        for i, (text, score) in enumerate(zip(needles, sims)):
            print(f"{i:02d} sim={score:.4f} needle={text}")
        for i, (text, score) in enumerate(zip(new_needles, new_sims)):
            print(f"{i:02d} sim={score:.4f} needle={text}")

        plot_similarity_scores(
            question=question,
            needles=needles,
            sims=sims,
            out_path=out_dir / "characters_similarity_scores.pdf",
            dpi=args.dpi,
        )
        plot_rephrased_similarity_comparison(
            needles=needles,
            new_needles=new_needles,
            sims_original=sims,
            sims_rephrased=new_sims,
            out_path=out_dir / "characters_rephrased_similarity_comparison.pdf",
            dpi=args.dpi,
        )

    print(f"Wrote PDF plots to: {out_dir}")
    print("- characters_similarity_scores.pdf [DOUBLE column]")
    print("- characters_rephrased_similarity_comparison.pdf [DOUBLE column]")


if __name__ == "__main__":
    main()





