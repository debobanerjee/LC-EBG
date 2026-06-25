#!/usr/bin/env python3
"""Ablation: Why does Ans-Only vs Ans+Evid delta vary across models?

(1) Conditional accuracy in Ans+Evid runs:
    P(answer_correct | evidence_correct) vs P(answer_correct | evidence_wrong).
    If the gap is large → evidence grounding genuinely helps reasoning.
    If the gap is small → models answer correctly regardless of evidence
                          (suggesting parametric memorisation).

(2) Depth-dependence comparison:
    Mean accuracy as a function of needle depth, for Ans-Only vs Ans+Evid.
    If Ans-Only stays flat across depths while Ans+Evid degrades, the model is
    leaning on parametric knowledge rather than retrieved evidence.

Outputs:
  results/answer_only/plots/ablation_conditional_accuracy.pdf
  results/answer_only/plots/ablation_depth_dependence.pdf
"""

import json
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import (
    TICK_FONTSIZE,
    LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    apply_publication_style,
)

warnings.filterwarnings("ignore", category=FutureWarning)
apply_publication_style()

# ─── Configuration ────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT  = PROJECT_DIR / "evaluation"
SPECIAL_ROOT  = RESULTS_ROOT / "special_experiments"
REASONING    = "commonsense_knowledge"
OUTPUT_DIR = PROJECT_DIR / "results" / "answer_only"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

STRICT_BOOK   = "1"
STRICT_DEPTHS = {0.0, 0.33, 0.67, 1.0}
CANONICAL_TESTS = {
    "0402_T01_C02_onehop","0405_T01_C02_onehop",
    "0402Inv_T01_C02_onehop","0405Inv_T01_C02_onehop",
    "0402_T01_C02_twohop","0405_T01_C02_twohop",
    "0402Inv_T01_C02_twohop","0405Inv_T01_C02_twohop",
}
NEEDLE_CHARACTER = {
    "0402": "Yuki","0402Inv": "Yuki",
    "0405": "Stuart","0405Inv": "Stuart",
}
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mdir(k): return k.replace(".", "-")

def _is_api_error(r):
    if r.get("error") or r.get("error_type"): return True
    if r.get("response") is None and r.get("input_tokens", 0) == 0: return True
    return False


def load_records(base_dir: Path, model_key: str):
    """Yield (depth, answer_metric, evidence_metric, ctx_len) tuples per record."""
    mdir = _mdir(model_key)
    comm = base_dir / REASONING
    if not comm.is_dir(): return
    for ctx_dir in comm.glob("rand_shuffle_*"):
        try: ctx_len = int(ctx_dir.name.split("_")[-1])
        except ValueError: continue
        for test_dir in ctx_dir.iterdir():
            if not test_dir.is_dir(): continue
            tname = test_dir.name
            if tname not in CANONICAL_TESTS: continue
            nid = re.match(r"^(0402Inv|0402|0405Inv|0405)_", tname)
            if not nid: continue
            expected_char = NEEDLE_CHARACTER.get(nid.group(1))
            book_file = test_dir / f"{mdir}_rand_book_{STRICT_BOOK}_{tname}.json"
            if not book_file.exists(): continue
            try: data = json.loads(book_file.read_text())
            except: continue
            for r in data.get("results", []):
                if _is_api_error(r): continue
                depth = (r.get("placement_metadata") or {}).get("depth")
                if depth is None: continue
                d = round(float(depth), 2)
                if d not in STRICT_DEPTHS: continue
                if expected_char and r.get("selected_character","") != expected_char:
                    continue
                a = r.get("answer_metric")
                e = r.get("evidence_metric")
                if a is None: continue
                yield (d, int(a), int(e) if e is not None else None, ctx_len)


def collect_all():
    """Return per-model dict of std and ao record lists."""
    data = {}
    for k in MODEL_ORDER:
        mdir = _mdir(k)
        std_base = RESULTS_ROOT / f"results_{mdir}"
        if not std_base.is_dir():
            alt = RESULTS_ROOT / f"results_{k}"
            if alt.is_dir(): std_base = alt
        ao_base = SPECIAL_ROOT / f"results_{mdir}-answer-only"
        data[k] = {
            "std": list(load_records(std_base, k)),
            "ao":  list(load_records(ao_base,  k)),
        }
    return data


# ─── Analysis (1): Conditional accuracy ───────────────────────────────────────

def conditional_stats(records):
    """Return (p_ans_given_evid_right, p_ans_given_evid_wrong, n_right, n_wrong)."""
    pos, neg = [], []
    for d, a, e, c in records:
        if e is None: continue
        if e == 1: pos.append(a)
        else:      neg.append(a)
    p_pos = (sum(pos)/len(pos))*100 if pos else float("nan")
    p_neg = (sum(neg)/len(neg))*100 if neg else float("nan")
    return p_pos, p_neg, len(pos), len(neg)


def plot_conditional(all_data):
    stats = []
    for k in MODEL_ORDER:
        p_r, p_w, n_r, n_w = conditional_stats(all_data[k]["std"])
        # also overall AO and Std means to compute delta
        std_a = np.mean([a for _,a,_,_ in all_data[k]["std"]])*100
        ao_a  = np.mean([a for _,a,_,_ in all_data[k]["ao"]])*100 if all_data[k]["ao"] else float("nan")
        delta = ao_a - std_a
        stats.append((MODEL_DISPLAY[k], p_r, p_w, n_r, n_w, std_a, ao_a, delta))

    # Sort by delta (most positive first)
    stats.sort(key=lambda x: x[7], reverse=True)
    names = [s[0] for s in stats]
    p_r = [s[1] for s in stats]
    p_w = [s[2] for s in stats]

    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    x = np.arange(len(names))
    w = 0.38
    b1 = ax.bar(x - w/2, p_r, w, label=r"P(Ans correct | Evid correct)",
                color="#0571b0", edgecolor="black", linewidth=0.7)
    b2 = ax.bar(x + w/2, p_w, w, label=r"P(Ans correct | Evid wrong)",
                color="#ca0020", edgecolor="black", linewidth=0.7)

    # annotate bars
    for bar, v in zip(b1, p_r):
        ax.text(bar.get_x()+bar.get_width()/2, v+1.5, f"{v:.1f}",
                ha="center", va="bottom", fontsize=TICK_FONTSIZE-6)
    for bar, v in zip(b2, p_w):
        if not np.isnan(v):
            ax.text(bar.get_x()+bar.get_width()/2, v+1.5, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=TICK_FONTSIZE-6)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=TICK_FONTSIZE-2, rotation=20, ha="right")
    ax.set_ylabel("Answer Accuracy (%)", fontsize=LABEL_FONTSIZE)
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("Conditional Answer Accuracy in Ans+Evid Runs\n(does correct evidence lift answer accuracy?)",
                 fontsize=LABEL_FONTSIZE-2, pad=14)
    # Legend below the x-axis labels, horizontal (both items side by side).
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              fontsize=LEGEND_FONTSIZE-4, framealpha=0.95, frameon=False,
              columnspacing=2.0, handlelength=1.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.subplots_adjust(bottom=0.22)

    out = PLOTS_DIR / "ablation_conditional_accuracy.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved → {out}")
    return stats


# ─── Analysis (2): Depth dependence ───────────────────────────────────────────

def depth_means(records):
    """Return {depth: mean_accuracy_pct, n}."""
    bucket = {}
    for d, a, e, c in records:
        bucket.setdefault(d, []).append(a)
    return {d: (100*sum(v)/len(v), len(v)) for d, v in bucket.items()}


def plot_depth(all_data):
    depths_sorted = sorted(STRICT_DEPTHS)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), dpi=300, sharex=True, sharey=True)
    axes = axes.flatten()

    # Order by delta (positive → negative) for visual clarity
    deltas = {}
    for k in MODEL_ORDER:
        std_a = np.mean([a for _,a,_,_ in all_data[k]["std"]])*100
        ao_a  = np.mean([a for _,a,_,_ in all_data[k]["ao"]])*100 if all_data[k]["ao"] else float("nan")
        deltas[k] = ao_a - std_a
    ordered_keys = sorted(MODEL_ORDER, key=lambda k: -deltas[k])

    summary = {}
    for ax, k in zip(axes, ordered_keys):
        std_m = depth_means(all_data[k]["std"])
        ao_m  = depth_means(all_data[k]["ao"])
        ys_std = [std_m.get(d,(float("nan"),0))[0] for d in depths_sorted]
        ys_ao  = [ao_m.get(d,(float("nan"),0))[0]  for d in depths_sorted]

        ax.plot(depths_sorted, ys_ao,  marker="o", linewidth=2.5, markersize=10,
                color="#0571b0", label="Ans* (Ans-Only)")
        ax.plot(depths_sorted, ys_std, marker="s", linewidth=2.5, markersize=10,
                color="#ca0020", label="Ans (Ans+Evid)")

        ax.set_title(f"{MODEL_DISPLAY[k]}  (Δ={deltas[k]:+.1f} pp)",
                     fontsize=LABEL_FONTSIZE-4, pad=6)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(depths_sorted)
        ax.set_xticklabels([f"{d:.2f}" for d in depths_sorted], fontsize=TICK_FONTSIZE-6)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE-6)

        # slope (range): max - min, lower means flatter
        flat_ao  = np.nanmax(ys_ao)  - np.nanmin(ys_ao)
        flat_std = np.nanmax(ys_std) - np.nanmin(ys_std)
        summary[k] = (flat_ao, flat_std, ys_ao, ys_std)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               fontsize=LEGEND_FONTSIZE-2, frameon=True,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("", fontsize=LABEL_FONTSIZE)
    fig.text(0.5, 0.02, "Needle Depth", ha="center", fontsize=LABEL_FONTSIZE)
    fig.text(0.005, 0.5, "Answer Accuracy (%)", va="center", rotation="vertical",
             fontsize=LABEL_FONTSIZE)
    fig.suptitle("Accuracy vs Needle Depth (Ans* vs Ans)",
                 fontsize=LABEL_FONTSIZE, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0.02, 0.04, 1, 0.95])

    out = PLOTS_DIR / "ablation_depth_dependence.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved → {out}")
    return summary


# ─── Run ──────────────────────────────────────────────────────────────────────

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data …")
    all_data = collect_all()
    for k in MODEL_ORDER:
        print(f"  {MODEL_DISPLAY[k]:<14}  std n={len(all_data[k]['std'])}  ao n={len(all_data[k]['ao'])}")

    print("\n─── Analysis (1): Conditional accuracy in Ans+Evid runs ───")
    stats = plot_conditional(all_data)
    print(f"\n{'Model':<14} {'Δ(AO−Std)':>11} {'P(A|E✓)':>10} {'P(A|E✗)':>10} {'Gap':>8}  {'n(E✓)':>6} {'n(E✗)':>6}")
    print("-"*80)
    for name, p_r, p_w, n_r, n_w, std_a, ao_a, delta in stats:
        gap = (p_r - p_w) if not (np.isnan(p_w) or np.isnan(p_r)) else float("nan")
        print(f"{name:<14} {delta:>+10.1f} {p_r:>9.1f}% {p_w:>9.1f}% {gap:>+7.1f}  {n_r:>6} {n_w:>6}")

    print("\n─── Analysis (2): Depth dependence (flatness = max − min across depths) ───")
    summary = plot_depth(all_data)
    print(f"\n{'Model':<14} {'Δ(AO−Std)':>11} {'AO flat':>10} {'Std flat':>10}")
    print("-"*60)
    for k in MODEL_ORDER:
        flat_ao, flat_std, ys_ao, ys_std = summary[k]
        std_a = np.mean([a for _,a,_,_ in all_data[k]["std"]])*100
        ao_a  = np.mean([a for _,a,_,_ in all_data[k]["ao"]])*100 if all_data[k]["ao"] else float("nan")
        delta = ao_a - std_a
        print(f"{MODEL_DISPLAY[k]:<14} {delta:>+10.1f} {flat_ao:>9.1f}pp {flat_std:>9.1f}pp")


if __name__ == "__main__":
    main()
