#!/usr/bin/env python3
"""
Shared publication plot styling for NoLiMa analysis scripts.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.container import BarContainer


PALETTE = [
    "#ca0020",
    "#f4a582",
    "#92c5de",
    "#0571b0",
]

TICK_FONTSIZE = 22
LABEL_FONTSIZE = 26
LEGEND_FONTSIZE = 22
HEATMAP_NA_COLOR = "#b3b3b3"
HEATMAP_NA_TEXT = "N/A"
HEATMAP_NA_TEXT_COLOR = "#000000"
HEATMAP_CELL_FONTSIZE = max(12, TICK_FONTSIZE - 4)
HEATMAP_NA_FONTSIZE = max(12, TICK_FONTSIZE - 5)

_SHORT_LABELS = {
    "Claude Sonnet 4": "Sonnet-4",
    "Claude Sonnet 4.5": "Sonnet-4.5",
    "Gemini 2.5 Flash": "Gemini-2.5-Fl",
    "Gemini 3 Flash": "Gemini-3-Fl",
    "Answer": "Ans",
    "Answer + Evidence": "Ans+Evid",
    "Factual TN: Answer + Evidence": "Factual TN (A+E)",
    "Contradictory TN: Answer + Evidence": "Contrad. TN (A+E)",
}


def palette_color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def build_color_map(keys: Iterable[str]) -> dict[str, str]:
    return {k: palette_color(i) for i, k in enumerate(keys)}


def _shorten_label(label: str, max_len: int = 14) -> str:
    text = str(label)
    if text in _SHORT_LABELS:
        return _SHORT_LABELS[text]
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _is_obvious_label(label: str) -> bool:
    txt = (label or "").strip().lower()
    return txt in {
        "context length",
        "context length (log scale)",
    }


def _has_bars(ax) -> bool:
    return any(isinstance(container, BarContainer) for container in ax.containers)


def _strip_bar_value_annotations(ax):
    # Remove top/bottom numeric and delta annotations on bar charts.
    for txt in list(ax.texts):
        content = (txt.get_text() or "").strip().lower()
        if not content:
            continue
        if re.search(r"\d", content) or "Δ" in content or "pp" in content or "%" in content:
            txt.remove()


def apply_publication_style():
    matplotlib.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": TICK_FONTSIZE,
            "axes.titlesize": TICK_FONTSIZE,
            "axes.labelsize": LABEL_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "axes.prop_cycle": cycler(color=PALETTE),
        }
    )


def style_figure(fig):
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")

    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    shared_xlabel = next((ax.get_xlabel() for ax in visible_axes if ax.get_xlabel()), "")
    shared_ylabel = next((ax.get_ylabel() for ax in visible_axes if ax.get_ylabel()), "")
    use_shared_labels = len(visible_axes) > 1

    for ax in visible_axes:
        # No titles for final publication figures.
        ax.set_title("")

        # Axis labels and ticks (shared labels for multi-panel figures).
        if use_shared_labels:
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            if _is_obvious_label(ax.get_xlabel()):
                ax.set_xlabel("")
            if _is_obvious_label(ax.get_ylabel()):
                ax.set_ylabel("")

            if ax.get_xlabel():
                ax.xaxis.label.set_size(LABEL_FONTSIZE)
            if ax.get_ylabel():
                ax.yaxis.label.set_size(LABEL_FONTSIZE)

        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

        # Keep vertical-bar charts readable by shortening long category labels.
        xlabels = [tick.get_text() for tick in ax.get_xticklabels()]
        if xlabels and all(lbl != "" for lbl in xlabels) and any(len(lbl) > 14 for lbl in xlabels):
            ax.set_xticklabels([_shorten_label(lbl) for lbl in xlabels], rotation=25, ha="right")

        ylabels = [tick.get_text() for tick in ax.get_yticklabels()]
        if ylabels and all(lbl != "" for lbl in ylabels) and any(len(lbl) > 18 for lbl in ylabels):
            ax.set_yticklabels([_shorten_label(lbl, max_len=18) for lbl in ylabels])

        legend = ax.get_legend()
        if legend:
            for txt in legend.get_texts():
                txt.set_fontsize(LEGEND_FONTSIZE)

        if _has_bars(ax):
            _strip_bar_value_annotations(ax)

    if use_shared_labels:
        if shared_xlabel and not _is_obvious_label(shared_xlabel):
            fig.supxlabel(shared_xlabel, fontsize=LABEL_FONTSIZE)
        if shared_ylabel and not _is_obvious_label(shared_ylabel):
            fig.supylabel(shared_ylabel, fontsize=LABEL_FONTSIZE)


def save_publication_figure(fig, out_dir: Path, name: str, dpi: int = 300) -> Path:
    apply_publication_style()
    style_figure(fig)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = Path(name).with_suffix(".pdf").name
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=max(300, dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
