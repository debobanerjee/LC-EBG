#!/usr/bin/env python3
"""
NoLiMa Research Analysis – Publication-Quality Evaluation
==========================================================

Produces rigorous, error-filtered analysis of needle-in-a-haystack (NIAH)
results across multiple frontier LLMs.  All API errors are excluded before
computing metrics.

Standard plots use a STRICT canonical setup:
  - Book 1 only
  - T01 template variant only
  - 4 depths (0%, 33%, 67%, 100%)
  - Canonical characters: Yuki for 0402/0402Inv, Stuart for 0405/0405Inv

Comparison plots (needle_family, variant_sensitivity, book_comparison,
character_bias) are exempt and use broader data to enable meaningful contrasts.

Output figures (results/standard/):
  Standard (strict setup):
    1.  accuracy_heatmap.png       – Model × context heatmap (onehop + twohop)
    2.  scaling_curves.png         – Accuracy vs context length with ±1 SE
    3.  onehop_vs_twohop.png       – Reasoning difficulty comparison
    4.  depth_analysis.png         – Accuracy vs needle depth (lost-in-middle)
    5.  model_ranking.png          – Fair ranking (shared context lengths)
    6.  answer_vs_evidence.png     – Per-model answer acc vs evidence acc
    7.  per_model_curves.png       – Onehop/twohop curves per model
    8.  context_stress_test.png    – Short vs long context stress test (dumbbell)
    9.  failure_mode_breakdown.png – Outcome decomposition (short vs long)
  Comparison (exempt):
    10. needle_family_performance.png – Relative needle difficulty + absolute bars
    11. variant_sensitivity.png      – Per-context T04−T01 delta distributions
    12. book_comparison.png          – Random book vs Book 1 performance
    13. character_bias.png           – Accuracy by character (matched setup only)

Usage:
    python research_analysis.py
"""

import json, os, re, sys, warnings
from itertools import combinations
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

from plot_style import (
    PALETTE,
    apply_publication_style,
    build_color_map,
    save_publication_figure,
)

warnings.filterwarnings("ignore", category=FutureWarning)
apply_publication_style()

# ─── Configuration ────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_DIR / "evaluation"
REASONING_TYPE = "commonsense_knowledge"
OUTPUT_DIR = PROJECT_DIR / "results" / "standard"
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"
SKIP_RESULT_DIRS = {"DEPRECATED_results_claude-3-7-sonnet-20250219"}

RANDOM_SEED = 42
MIN_SAMPLES = 8   # min samples per cell (strict setup: 4 needles × 4 depths = 16 per hop-cell)

MODEL_DISPLAY = {
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "gemini-2-5-flash": "Gemini 2.5 Flash",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gpt-4o": "GPT-4o",
    "gpt-4-1": "GPT-4.1",
    "gpt-5-2025-08-07": "GPT-5",
    "o3-mini-2025-01-31": "o3-mini",
}

MODEL_PALETTE = build_color_map(MODEL_DISPLAY.keys())

STRICT_BOOK = "1"
STRICT_DEPTHS_4 = (0.0, 0.33, 0.67, 1.0)
NEEDLE_CHARACTER = {
    "0402": "Yuki", "0402Inv": "Yuki",
    "0405": "Stuart", "0405Inv": "Stuart",
}
CANONICAL_TESTS_T01 = {
    "0402_T01_C02_onehop", "0402_T01_C02_twohop",
    "0405_T01_C02_onehop", "0405_T01_C02_twohop",
    "0402Inv_T01_C02_onehop", "0402Inv_T01_C02_twohop",
    "0405Inv_T01_C02_onehop", "0405Inv_T01_C02_twohop",
}
CANONICAL_TESTS_T01_T04 = CANONICAL_TESTS_T01 | {
    "0402_T04_C02_onehop", "0402_T04_C02_twohop",
    "0405_T04_C02_onehop", "0405_T04_C02_twohop",
    "0402Inv_T04_C02_onehop", "0402Inv_T04_C02_twohop",
    "0405Inv_T04_C02_onehop", "0405Inv_T04_C02_twohop",
}
NEEDLE_ORDER = ["0402", "0402Inv", "0405", "0405Inv"]
VARIANT_ORDER = ["T01", "T04"]
TEST_NAME_RE = re.compile(r"^(0402Inv|0402|0405Inv|0405)_(T\d+)_C\d+_(onehop|twohop)$")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def dname(m: str) -> str:
    return MODEL_DISPLAY.get(m, m)

def fmt_ctx(length: int) -> str:
    if length >= 1_000_000:
        return f"{length / 1_000_000:.0f}M"
    if length >= 1_000:
        return f"{length // 1_000}K"
    return str(length)

def color_of(m: str) -> str:
    return MODEL_PALETTE.get(m, PALETTE[3])

def _savefig(fig, name: str, out_dir: Path, dpi: int = 300):
    """Save figure to out_dir using shared publication style."""
    out = save_publication_figure(fig, out_dir, name, dpi=dpi)
    print(f"  ✓ {out}")

def _sorted_models(df: pd.DataFrame) -> list:
    """Consistent alphabetical model ordering by display name."""
    return sorted(df["model"].unique(), key=lambda m: dname(m))


# ─── Data Loading ─────────────────────────────────────────────────────────────

def _is_api_error(r: dict) -> bool:
    """True when a result entry is a genuine API / infrastructure error."""
    if r.get("error") or r.get("error_type"):
        return True
    if r.get("response") is None and r.get("input_tokens", 0) == 0:
        return True
    return False


def _reparse_response(resp) -> dict:
    """Re-parse a stored response (trailing commas, code fences, etc.)."""
    if resp is None:
        return None
    if isinstance(resp, dict):
        return resp
    if not isinstance(resp, str) or len(resp.strip()) < 3:
        return None
    text = resp
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    # Code block extraction
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    # Direct JSON with "answer" key
    m = re.search(r'\{[^{}]*"answer"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    # Any JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidate = re.sub(r",\s*}", "}", m.group(0))
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _rescore_entry(r: dict, reparsed: dict) -> tuple:
    """Re-evaluate answer/evidence metrics using a re-parsed response."""
    if reparsed is None or "answer" not in reparsed:
        return int(r.get("answer_metric", 0) or 0), int(r.get("evidence_metric", 0) or 0)
    expected_char = r.get("selected_character", "")
    placement = r.get("placement_metadata", {})
    expected_line = placement.get("needle_line_num")
    answer_text = str(reparsed.get("answer", "")).lower()
    answer_ok = expected_char.lower() in answer_text if expected_char else False
    lines = reparsed.get("lines", [])
    if not isinstance(lines, list):
        lines = [lines] if lines is not None else []
    int_lines = []
    for ln in lines:
        try:
            int_lines.append(int(ln))
        except (ValueError, TypeError):
            pass
    evidence_ok = expected_line is not None and expected_line in int_lines
    return int(answer_ok), int(evidence_ok)


def load_raw() -> pd.DataFrame:
    """Load every individual depth result across all models.

    Re-parses string responses, re-scores metrics, marks API errors,
    extracts book number. Returns one row per (model, ctx, test, book, depth).
    """
    rows = []
    stats = {"reparsed": 0, "rescored": 0, "api_errors": 0}

    for model_dir in sorted(RESULTS_ROOT.glob("results_*")):
        if model_dir.name in SKIP_RESULT_DIRS:
            continue
        model_name = model_dir.name.replace("results_", "")
        comm_dir = model_dir / REASONING_TYPE
        if not comm_dir.is_dir():
            continue

        for ctx_dir in sorted(comm_dir.glob("rand_shuffle_*")):
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
                    results = data.get("results", [])
                    if not results:
                        continue
                    bm = re.search(r"rand_book_(\d+)", jf.stem)
                    book = bm.group(1) if bm else "?"

                    for r in results:
                        err = _is_api_error(r)
                        if err:
                            stats["api_errors"] += 1
                            ans, evi = 0, 0
                        else:
                            resp = r.get("response")
                            if isinstance(resp, str) and len(resp) > 5:
                                reparsed = _reparse_response(resp)
                                if reparsed and "answer" in reparsed:
                                    stats["reparsed"] += 1
                                    new_ans, new_evi = _rescore_entry(r, reparsed)
                                    old_ans = int(r.get("answer_metric", 0) or 0)
                                    old_evi = int(r.get("evidence_metric", 0) or 0)
                                    if new_ans != old_ans or new_evi != old_evi:
                                        stats["rescored"] += 1
                                    ans, evi = new_ans, new_evi
                                else:
                                    ans = int(r.get("answer_metric", 0) or 0)
                                    evi = int(r.get("evidence_metric", 0) or 0)
                            else:
                                ans = int(r.get("answer_metric", 0) or 0)
                                evi = int(r.get("evidence_metric", 0) or 0)

                        placement = r.get("placement_metadata", {})
                        depth = placement.get("depth")
                        rows.append({
                            "model": model_name,
                            "context_length": ctx_len,
                            "reasoning_hop": hop,
                            "test_name": tname,
                            "book": book,
                            "depth": float(depth) if depth is not None else np.nan,
                            "selected_character": r.get("selected_character", "Unknown"),
                            "answer_correct": ans,
                            "evidence_correct": evi,
                            "both_correct": 1 if (ans == 1 and evi == 1) else 0,
                            "is_error": err,
                            "input_tokens": r.get("input_tokens", 0),
                            "output_tokens": r.get("output_tokens", 0),
                        })

    if stats["reparsed"]:
        print(f"  📝 Re-parsed {stats['reparsed']} string responses")
    if stats["rescored"]:
        print(f"  ✅ Rescored {stats['rescored']} entries")
    if stats["api_errors"]:
        print(f"  ⚠ {stats['api_errors']} API errors will be filtered")
    if not rows:
        raise RuntimeError("No results found.")
    return pd.DataFrame(rows)


# ─── Dataset Preparation ─────────────────────────────────────────────────────

def _sample_one_book_per_cell(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """For each (model, context_length, test_name), randomly pick one book."""
    rng = np.random.default_rng(seed)
    keys = df[["model", "context_length", "test_name"]].drop_duplicates()
    chosen = []
    for _, row in keys.iterrows():
        sub = df[(df["model"] == row["model"]) & (df["context_length"] == row["context_length"]) & (df["test_name"] == row["test_name"])]
        books = sub["book"].unique().tolist()
        if books:
            chosen.append((row["model"], row["context_length"], row["test_name"], rng.choice(books)))
    if not chosen:
        return df.iloc[0:0].copy()
    m, c, t, b = zip(*chosen)
    sel = pd.DataFrame({"model": m, "context_length": c, "test_name": t, "book": b})
    return df.merge(sel, on=["model", "context_length", "test_name", "book"], how="inner")


def _filter_strict_book1_4depths(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only book 1 and the 4 standard depths (0%, 33%, 67%, 100%)."""
    df = df[df["book"] == STRICT_BOOK].copy()
    df = df.dropna(subset=["depth"])
    df["depth_r2"] = df["depth"].round(2)
    df = df[df["depth_r2"].isin(set(STRICT_DEPTHS_4))].drop(columns=["depth_r2"])
    return df


def _filter_correct_characters(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where selected_character ≠ canonical needle character.

    Canonical mapping: 0402/0402Inv → Yuki, 0405/0405Inv → Stuart.
    """
    if "selected_character" not in df.columns:
        return df
    needle_ids = df["test_name"].str.extract(
        r"^(0402Inv|0402|0405Inv|0405)_", expand=False
    )
    expected = needle_ids.map(NEEDLE_CHARACTER)
    mask = expected.isna() | (df["selected_character"] == expected)
    dropped = (~mask).sum()
    if dropped:
        print(f"  Character filter: dropped {dropped} rows with non-canonical characters")
    return df[mask].copy()


def _build_strict_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical strict dataset for standard plots.

    Filters: Book 1, T01 only, 4 depths, canonical characters (Yuki/Stuart),
    no API errors.
    """
    out = _filter_tests(df, CANONICAL_TESTS_T01)
    out = _filter_strict_book1_4depths(out)
    out = _filter_correct_characters(out)
    out = out[~out["is_error"]].copy()
    return out


def _ctx_weighted_accuracy(df, metric="both_correct"):
    """Accuracy with equal weight per context length."""
    cell = df.groupby(["model", "context_length"])[metric].mean().reset_index(name="cell_acc")
    return cell.groupby("model")["cell_acc"].mean() * 100.0


def _filter_tests(df: pd.DataFrame, allowed_tests: set) -> pd.DataFrame:
    """Keep only rows whose test_name is in the allowed canonical set."""
    return df[df["test_name"].isin(allowed_tests)].copy()


def _parse_test_name(test_name: str) -> dict:
    """Parse canonical test_name into needle/variant metadata."""
    m = TEST_NAME_RE.match(str(test_name))
    if not m:
        return {
            "needle_id": None,
            "variant": None,
            "hop_from_test": None,
            "needle_family": None,
            "is_inverse": None,
        }
    needle_id, variant, hop = m.group(1), m.group(2), m.group(3)
    return {
        "needle_id": needle_id,
        "variant": variant,
        "hop_from_test": hop,
        "needle_family": needle_id.replace("Inv", ""),
        "is_inverse": int("Inv" in needle_id),
    }


def _with_test_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Attach parsed test metadata columns to a dataframe."""
    parsed = pd.DataFrame([_parse_test_name(t) for t in df["test_name"]], index=df.index)
    out = df.copy()
    for c in parsed.columns:
        out[c] = parsed[c]
    return out


# ─── Plot 1: Accuracy Heatmap ────────────────────────────────────────────────

def plot_accuracy_heatmap(df: pd.DataFrame, out_dir: Path):
    """Heatmap: rows=models, cols=context lengths. Two panels: onehop + twohop."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"hspace": 0.35})
    cmap = LinearSegmentedColormap.from_list(
        "pub", [PALETTE[0], PALETTE[1], PALETTE[1], PALETTE[2], PALETTE[3]], N=256)

    for ax_idx, hop in enumerate(["onehop", "twohop"]):
        ax = axes[ax_idx]
        sub = df[df["reasoning_hop"] == hop]
        pivot = sub.pivot_table(index="model", columns="context_length",
                                values="both_correct", aggfunc=lambda x: 100.0 * x.mean())
        counts = sub.pivot_table(index="model", columns="context_length",
                                 values="both_correct", aggfunc="count")

        sorted_models = sorted(pivot.index, key=lambda m: dname(m))
        sorted_ctx = sorted(pivot.columns)
        pivot = pivot.reindex(index=sorted_models, columns=sorted_ctx)
        counts = counts.reindex(index=sorted_models, columns=sorted_ctx)

        data = pivot.values
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=100,
                       interpolation="nearest")

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                n = counts.values[i, j] if not np.isnan(counts.values[i, j]) else 0
                if np.isnan(val) or n < MIN_SAMPLES:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 fill=True, facecolor=PALETTE[1],
                                 edgecolor="white", linewidth=0.5))
                    label = "—" if np.isnan(val) else f"~{val:.0f}\nn={int(n)}"
                    ax.text(j, i, label, ha="center", va="center",
                            fontsize=4, color="#999", style="italic")
                else:
                    tc = "white" if val < 40 or val > 85 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=6, color=tc, fontweight="bold")

        ax.set_xticks(range(len(sorted_ctx)))
        ax.set_xticklabels([fmt_ctx(c) for c in sorted_ctx], fontsize=7,
                           rotation=45, ha="right")
        ax.set_yticks(range(len(sorted_models)))
        ax.set_yticklabels([dname(m) for m in sorted_models], fontsize=9)
        ax.set_title(f"{hop.capitalize()} – Combined Accuracy (%)",
                     fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Accuracy (%)", fontsize=10)
    fig.suptitle("Model Accuracy Across Context Lengths (Error-Filtered)",
                 fontsize=14, fontweight="bold", y=1.01)
    _savefig(fig, "accuracy_heatmap.png", PLOTS_DIR)


# ─── Plot 3: Scaling Curves ──────────────────────────────────────────────────

def plot_scaling_curves(df: pd.DataFrame, out_dir: Path):
    """Combined accuracy vs context length with ±1 SE bands. Two panels."""
    models = _sorted_models(df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax_idx, hop in enumerate(["onehop", "twohop"]):
        ax = axes[ax_idx]
        sub = df[df["reasoning_hop"] == hop]
        for model in models:
            sm = sub[sub["model"] == model]
            agg = sm.groupby("context_length").agg(
                acc=("both_correct", lambda x: 100.0 * x.mean()),
                n=("both_correct", "count"),
                se=("both_correct", lambda x: 100.0 * x.sem() if len(x) > 1 else 0),
            ).reset_index()
            agg = agg[agg["n"] >= MIN_SAMPLES]
            if agg.empty:
                continue
            ax.plot(agg["context_length"], agg["acc"], marker="o", markersize=4,
                    linewidth=2, color=color_of(model), label=dname(model))
            ax.fill_between(agg["context_length"],
                            (agg["acc"] - agg["se"]).clip(0),
                            (agg["acc"] + agg["se"]).clip(upper=100),
                            alpha=0.1, color=color_of(model))

        ax.set_ylim(-2, 105)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: fmt_ctx(int(v))))
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title(f"{hop.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Context Length (log scale)", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Combined Accuracy (%)", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(models), 4),
               fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Accuracy Scaling with Context Length (±1 SE)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig(fig, "scaling_curves.png", PLOTS_DIR)


# ─── Plot 4: Onehop vs Twohop ────────────────────────────────────────────────

def plot_onehop_vs_twohop(df: pd.DataFrame, out_dir: Path):
    """Grouped bar chart comparing onehop vs twohop accuracy."""
    models = _sorted_models(df)
    agg = (
        df.groupby(["model", "reasoning_hop"])
        .agg(
            combined=("both_correct", lambda x: 100.0 * x.mean()),
            n=("both_correct", "count"),
        )
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(models))
    width = 0.35

    for hi, hop in enumerate(["onehop", "twohop"]):
        vals = []
        for m in models:
            row = agg[(agg["model"] == m) & (agg["reasoning_hop"] == hop)]
            vals.append(row["combined"].values[0] if len(row) > 0 else 0)

        ax1.bar(
            x + (-width / 2 + hi * width),
            vals,
            width,
            label=hop.capitalize(),
            color=[PALETTE[3], PALETTE[1]][hi],
            alpha=0.85,
            edgecolor="white",
        )

        for i, v in enumerate(vals):
            ax1.text(
                x[i] + (-width / 2 + hi * width),
                v + 1,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels([dname(m) for m in models], rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Combined Accuracy (%)", fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.set_title("Onehop vs. Twohop Accuracy", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Onehop vs. Twohop Reasoning Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _savefig(fig, "onehop_vs_twohop.png", PLOTS_DIR)


# ─── Plot 5: Depth Analysis (Lost in the Middle) ─────────────────────────────

def plot_depth_analysis(df: pd.DataFrame, out_dir: Path):
    """Line plot: accuracy vs needle depth (%), one line per model, two panels."""
    df_d = df.dropna(subset=["depth"]).copy()
    if df_d.empty:
        print("  No depth data - skipping depth_analysis.png")
        return

    keep = np.zeros(len(df_d), dtype=bool)
    for d in STRICT_DEPTHS_4:
        keep |= np.isclose(df_d["depth"].values, d, atol=0.005)
    df_d = df_d[keep].copy()
    if df_d.empty:
        print("  No data at standard 4 depths - skipping depth_analysis.png")
        return

    models = _sorted_models(df_d)
    df_d["depth_pct"] = df_d["depth"] * 100.0

    # Important: use constrained layout here
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5.8),
        sharey=True,
        constrained_layout=True
    )

    for ax_idx, hop in enumerate(["onehop", "twohop"]):
        ax = axes[ax_idx]
        sub = df_d[df_d["reasoning_hop"] == hop]

        for model in models:
            sm = sub[sub["model"] == model]
            agg = (
                sm.groupby("depth_pct")
                .agg(
                    acc=("both_correct", lambda x: 100.0 * x.mean()),
                    n=("both_correct", "count"),
                )
                .reset_index()
                .sort_values("depth_pct")
            )
            if agg.empty:
                continue

            ax.plot(
                agg["depth_pct"],
                agg["acc"],
                marker="o",
                markersize=5,
                linewidth=2,
                color=color_of(model),
                label=dname(model),
            )

        ax.set_xlim(-2, 102)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Needle Depth (% of Document)", fontsize=10, labelpad=8)
        if ax_idx == 0:
            ax.set_ylabel("Combined Accuracy (%)", fontsize=10)
        ax.set_title(f"{hop.capitalize()}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.17),   # ← small positive y lifts it just above x-ticks
        bbox_transform=fig.transFigure,
        ncols=min(len(models), 4),
        fontsize=12,
    )

    fig.suptitle(
        "Accuracy vs. Needle Placement Depth (All Context Lengths Pooled, Error-Filtered)",
        fontsize=13,
        fontweight="bold",
    )

    fig.subplots_adjust(bottom=0.18, top=0.92, wspace=0.05)

    _savefig(fig, "depth_analysis.png", PLOTS_DIR)


# ─── Plot 6: Model Ranking (Fair, Shared Context Lengths) ────────────────────

def plot_model_ranking(df: pd.DataFrame, out_dir: Path):
    """Fair ranking using only context lengths where ALL models have data."""
    models = _sorted_models(df)

    # Find shared context lengths
    model_ctx = df.groupby(["model", "context_length"]).size().reset_index(name="n")
    shared_ctx = set(df["context_length"].unique())
    for m in models:
        mc = set(model_ctx[(model_ctx["model"] == m) & (model_ctx["n"] >= MIN_SAMPLES)]["context_length"])
        shared_ctx &= mc
    shared_ctx = sorted(shared_ctx)

    if not shared_ctx:
        print("  ⚠ No shared context lengths – skipping model_ranking.png")
        return

    df_shared = df[df["context_length"].isin(shared_ctx)]
    fw_combined = _ctx_weighted_accuracy(df_shared, "both_correct")
    fw_answer = _ctx_weighted_accuracy(df_shared, "answer_correct")
    fw_evidence = _ctx_weighted_accuracy(df_shared, "evidence_correct")

    agg = pd.DataFrame({
        "model": fw_combined.index,
        "combined": fw_combined.values,
        "answer": fw_answer.reindex(fw_combined.index).values,
        "evidence": fw_evidence.reindex(fw_combined.index).values,
    })
    sc = df_shared.groupby("model").size().reset_index(name="n")
    agg = agg.merge(sc, on="model")
    agg["display"] = agg["model"].map(dname)
    agg = agg.sort_values("combined", ascending=True)

    ctx_range = f"{fmt_ctx(min(shared_ctx))}–{fmt_ctx(max(shared_ctx))}"
    print(f"  Fair comparison: {len(shared_ctx)} shared ctx ({ctx_range})")

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(agg))
    ax.barh(y, agg["combined"], color=[color_of(m) for m in agg["model"]],
            edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(agg["display"], fontsize=10)
    ax.set_xlabel("Combined Accuracy (%)", fontsize=10)
    ax.set_title(f"Fair Ranking: {len(shared_ctx)} Shared Context Lengths ({ctx_range})",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 115)
    ax.grid(axis="x", alpha=0.3)
    for i, row in enumerate(agg.itertuples()):
        ax.text(row.combined + 0.5, i,
                f"{row.combined:.1f}%  (n={int(row.n):,})",
                va="center", fontsize=9)

    fig.suptitle("Model Ranking (Shared Context Lengths Only)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _savefig(fig, "model_ranking.png", PLOTS_DIR)


# ─── Plot 6.5: Context Stress Test ────────────────────────────────────────────

def plot_context_stress_test(df: pd.DataFrame, out_dir: Path):
    """Dumbbell chart: short-context vs long-context performance by model and hop.

    Short score: context-weighted mean over contexts <= 100K.
    Long score: context-weighted mean over contexts >= 500K.
    Uses only context cells with at least MIN_SAMPLES rows.
    """
    short_max = 100_000
    long_min = 500_000
    rows = []

    for model in _sorted_models(df):
        for hop in ["onehop", "twohop"]:
            sm = df[(df["model"] == model) & (df["reasoning_hop"] == hop)]
            if sm.empty:
                continue

            cell = sm.groupby("context_length").agg(
                acc=("both_correct", lambda x: 100.0 * x.mean()),
                n=("both_correct", "count"),
            ).reset_index()
            cell = cell[cell["n"] >= MIN_SAMPLES]
            if cell.empty:
                continue

            short_cell = cell[cell["context_length"] <= short_max]
            long_cell = cell[cell["context_length"] >= long_min]
            if short_cell.empty or long_cell.empty:
                continue

            short_acc = short_cell["acc"].mean()
            long_acc = long_cell["acc"].mean()
            rows.append({
                "model": model,
                "hop": hop,
                "short_acc": short_acc,
                "long_acc": long_acc,
                "drop_pp": short_acc - long_acc,
                "n_short_ctx": len(short_cell),
                "n_long_ctx": len(long_cell),
            })

    if not rows:
        print("  ⚠ No data for context stress test – skipping context_stress_test.png")
        return

    metrics = pd.DataFrame(rows)
    metrics["display"] = metrics["model"].map(dname)
    metrics_csv = TABLES_DIR / "context_stress_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)
    print(f"  ✓ Context stress metrics CSV: {metrics_csv} ({len(metrics)} rows)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    for ax_idx, hop in enumerate(["onehop", "twohop"]):
        ax = axes[ax_idx]
        sub = metrics[metrics["hop"] == hop].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="#888")
            ax.set_title(hop.capitalize(), fontsize=12, fontweight="bold")
            continue

        sub = sub.sort_values("drop_pp", ascending=False).reset_index(drop=True)
        y = np.arange(len(sub))

        # Dumbbell connector (short -> long), colored by model.
        for i, row in enumerate(sub.itertuples()):
            c = color_of(row.model)
            ax.plot([row.short_acc, row.long_acc], [i, i], color=c, alpha=0.45, linewidth=2.5)
            ax.scatter(row.short_acc, i, color=c, s=42, marker="o", edgecolor="white", linewidth=0.7, zorder=3)
            ax.scatter(row.long_acc, i, color=c, s=42, marker="s", edgecolor="white", linewidth=0.7, zorder=3)
            ax.text(max(row.short_acc, row.long_acc) + 1.2, i, f"{row.drop_pp:+.1f} pp",
                    va="center", fontsize=8, color="#444")

        ax.set_yticks(y)
        ax.set_yticklabels(sub["display"], fontsize=9)
        ax.set_xlim(0, 102)
        ax.set_xlabel("Combined Accuracy (%)", fontsize=10)
        ax.set_title(f"{hop.capitalize()} (Short <=100K vs Long >=500K)",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Mini legend per panel (marker semantics).
        short_proxy = plt.Line2D([0], [0], marker='o', color='none',
                                 markerfacecolor="#666", markeredgecolor="white",
                                 markersize=7, label="Short")
        long_proxy = plt.Line2D([0], [0], marker='s', color='none',
                                markerfacecolor="#666", markeredgecolor="white",
                                markersize=7, label="Long")
        ax.legend(handles=[short_proxy, long_proxy], fontsize=8, loc="lower right")

    fig.suptitle("Context Stress Test: Performance Degradation from Short to Long Context",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _savefig(fig, "context_stress_test.png", PLOTS_DIR)


# ─── Plot 6.6: Failure-Mode Breakdown ─────────────────────────────────────────

def plot_failure_mode_breakdown(df: pd.DataFrame, out_dir: Path):
    """Stacked bars of outcome modes for short vs long context, by hop and model.

    Modes:
      - both_correct   : answer=1, evidence=1
      - answer_only    : answer=1, evidence=0
      - evidence_only  : answer=0, evidence=1
      - both_wrong     : answer=0, evidence=0
    Aggregation is context-weighted within each bucket/hop/model, with MIN_SAMPLES
    threshold applied per context cell.
    """
    short_max = 100_000
    long_min = 500_000

    work = df.copy()
    work["answer_only"] = ((work["answer_correct"] == 1) & (work["evidence_correct"] == 0)).astype(int)
    work["evidence_only"] = ((work["answer_correct"] == 0) & (work["evidence_correct"] == 1)).astype(int)
    work["both_wrong"] = ((work["answer_correct"] == 0) & (work["evidence_correct"] == 0)).astype(int)
    work["bucket"] = np.where(
        work["context_length"] <= short_max, "short",
        np.where(work["context_length"] >= long_min, "long", "mid")
    )
    work = work[work["bucket"].isin(["short", "long"])].copy()
    if work.empty:
        print("  ⚠ No short/long context data – skipping failure_mode_breakdown.png")
        return

    rows = []
    mode_cols = ["both_correct", "answer_only", "evidence_only", "both_wrong"]
    for model in _sorted_models(work):
        for hop in ["onehop", "twohop"]:
            for bucket in ["short", "long"]:
                sm = work[
                    (work["model"] == model)
                    & (work["reasoning_hop"] == hop)
                    & (work["bucket"] == bucket)
                ]
                if sm.empty:
                    continue

                cell = sm.groupby("context_length").agg(
                    n=("both_correct", "count"),
                    both_correct=("both_correct", "mean"),
                    answer_only=("answer_only", "mean"),
                    evidence_only=("evidence_only", "mean"),
                    both_wrong=("both_wrong", "mean"),
                ).reset_index()
                cell = cell[cell["n"] >= MIN_SAMPLES]
                if cell.empty:
                    continue

                rows.append({
                    "model": model,
                    "display": dname(model),
                    "hop": hop,
                    "bucket": bucket,
                    "n_contexts": len(cell),
                    "both_correct": 100.0 * cell["both_correct"].mean(),
                    "answer_only": 100.0 * cell["answer_only"].mean(),
                    "evidence_only": 100.0 * cell["evidence_only"].mean(),
                    "both_wrong": 100.0 * cell["both_wrong"].mean(),
                })

    if not rows:
        print("  ⚠ No cells pass sample threshold – skipping failure_mode_breakdown.png")
        return

    metrics = pd.DataFrame(rows)
    metrics_csv = TABLES_DIR / "failure_mode_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)
    print(f"  ✓ Failure-mode metrics CSV: {metrics_csv} ({len(metrics)} rows)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    mode_order = ["both_correct", "answer_only", "evidence_only", "both_wrong"]
    mode_label = {
        "both_correct": "Both Correct",
        "answer_only": "Answer-Only",
        "evidence_only": "Evidence-Only",
        "both_wrong": "Both Wrong",
    }
    mode_color = {
        "both_correct": PALETTE[3],
        "answer_only": PALETTE[1],
        "evidence_only": PALETTE[2],
        "both_wrong": PALETTE[0],
    }

    for ax_idx, hop in enumerate(["onehop", "twohop"]):
        ax = axes[ax_idx]
        sub = metrics[metrics["hop"] == hop].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="#888")
            ax.set_title(hop.capitalize(), fontsize=12, fontweight="bold")
            continue

        models = [m for m in _sorted_models(sub) if m in set(sub["model"])]
        x = np.arange(len(models))
        width = 0.36

        for bi, bucket in enumerate(["short", "long"]):
            sb = sub[sub["bucket"] == bucket].set_index("model")
            xpos = x + (-width / 2 if bucket == "short" else width / 2)
            bottom = np.zeros(len(models))
            for mode in mode_order:
                vals = np.array([sb.loc[m, mode] if m in sb.index else 0.0 for m in models])
                bars = ax.bar(
                    xpos, vals, width=width, bottom=bottom,
                    color=mode_color[mode], edgecolor="white", linewidth=0.5,
                    alpha=1.0 if bucket == "short" else 0.72,
                    label=f"{mode_label[mode]} ({bucket})" if ax_idx == 0 else None,
                    hatch=None if bucket == "short" else "//"
                )
                bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([dname(m) for m in models], rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 102)
        ax.set_ylabel("Outcome Share (%)", fontsize=10)
        ax.set_title(f"{hop.capitalize()} (Short <=100K vs Long >=500K)",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    # Legend: one entry per mode + bucket semantics.
    mode_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=mode_color[m], edgecolor="white", label=mode_label[m])
        for m in mode_order
    ]
    short_proxy = plt.Rectangle((0, 0), 1, 1, facecolor="#999", edgecolor="white", label="Short", alpha=1.0)
    long_proxy = plt.Rectangle((0, 0), 1, 1, facecolor="#999", edgecolor="white", label="Long", alpha=0.72, hatch="//")
    fig.legend(
        handles=mode_handles + [short_proxy, long_proxy],
        loc="lower center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.03)
    )

    fig.suptitle("Failure-Mode Breakdown Across Context Buckets",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _savefig(fig, "failure_mode_breakdown.png", PLOTS_DIR)


# ─── Plot 6.7: Needle-Family Performance ─────────────────────────────────────

def plot_needle_family_performance(df: pd.DataFrame, out_dir: Path):
    """Needle performance via relative difficulty and absolute accuracy bars."""
    sub = df[(~df["is_error"]) & (df["test_name"].isin(CANONICAL_TESTS_T01_T04))].copy()
    if sub.empty:
        print("  ⚠ No canonical rows for needle-family plot – skipping needle_family_performance.png")
        return

    sub = _with_test_meta(sub)
    sub = sub[sub["needle_id"].isin(NEEDLE_ORDER)].copy()
    if sub.empty:
        print("  ⚠ No parseable needle IDs – skipping needle_family_performance.png")
        return

    # Context-level performance per needle.
    cell = (sub.groupby(["model", "reasoning_hop", "needle_id", "context_length"])
            .agg(acc=("both_correct", "mean"), n=("both_correct", "count"))
            .reset_index())
    cell = cell[cell["n"] >= MIN_SAMPLES].copy()
    if cell.empty:
        print("  ⚠ No cells pass sample threshold – skipping needle_family_performance.png")
        return
    cell["acc_pp"] = 100.0 * cell["acc"]

    # Baseline within each model/hop/context to remove model+context effects.
    base = (cell.groupby(["model", "reasoning_hop", "context_length"])
            .agg(base_acc_pp=("acc_pp", "mean"))
            .reset_index())
    cell = cell.merge(base, on=["model", "reasoning_hop", "context_length"], how="left")
    cell["delta_pp"] = cell["acc_pp"] - cell["base_acc_pp"]

    # Per-model metrics (for audit) and global summary (for plotting).
    metrics = (cell.groupby(["model", "reasoning_hop", "needle_id"])
               .agg(
                   rel_delta_pp=("delta_pp", "mean"),
                   abs_acc_pp=("acc_pp", "mean"),
                   n_contexts=("context_length", "nunique"),
               )
               .reset_index())
    metrics_csv = TABLES_DIR / "needle_family_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)

    summary = (metrics.groupby(["reasoning_hop", "needle_id"])
               .agg(
                   rel_delta_pp=("rel_delta_pp", "mean"),
                   abs_acc_pp=("abs_acc_pp", "mean"),
                   n_models=("model", "nunique"),
               )
               .reset_index())
    summary_csv = TABLES_DIR / "needle_family_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"  ✓ Needle-family metrics CSV: {metrics_csv} ({len(metrics)} rows)")
    print(f"  ✓ Needle-family summary CSV: {summary_csv} ({len(summary)} rows)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    x = np.arange(len(NEEDLE_ORDER))
    width = 0.36
    hop_colors = {"onehop": PALETTE[3], "twohop": PALETTE[1]}

    # Left: relative difficulty (delta vs per-model/context baseline).
    # for hi, hop in enumerate(["onehop", "twohop"]):
    #     sh = summary[summary["reasoning_hop"] == hop].set_index("needle_id")
    #     vals = np.array([sh.loc[n, "rel_delta_pp"] if n in sh.index else np.nan for n in NEEDLE_ORDER])
    #     xpos = x + (-width / 2 if hi == 0 else width / 2)
    #     ax1.bar(xpos, vals, width=width, color=hop_colors[hop], alpha=0.9,
    #             edgecolor="white", label=hop.capitalize())
    #     for i, v in enumerate(vals):
    #         if np.isnan(v):
    #             continue
    #         ax1.text(xpos[i], v + (0.4 if v >= 0 else -0.6), f"{v:+.1f}",
    #                  ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    # ax1.axhline(0, color="black", linewidth=0.9)
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(NEEDLE_ORDER, fontsize=9)
    # ax1.set_ylabel("Relative Difficulty (pp vs model-context baseline)", fontsize=10)
    # ax1.set_title("Needle Difficulty (Controlled for Model/Context)", fontsize=11, fontweight="bold")
    # ax1.grid(axis="y", alpha=0.3)
    # ax1.legend(fontsize=9)

    # Right: absolute performance by needle.
    for hi, hop in enumerate(["onehop", "twohop"]):
        sh = summary[summary["reasoning_hop"] == hop].set_index("needle_id")
        vals = np.array([sh.loc[n, "abs_acc_pp"] if n in sh.index else np.nan for n in NEEDLE_ORDER])
        xpos = x + (-width / 2 if hi == 0 else width / 2)
        ax2.bar(xpos, vals, width=width, color=hop_colors[hop], alpha=0.9,
                edgecolor="white", label=hop.capitalize())
        for i, v in enumerate(vals):
            if np.isnan(v):
                continue
            ax2.text(xpos[i], v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(NEEDLE_ORDER, fontsize=9)
    ax2.set_ylim(0, 102)
    ax2.set_ylabel("Combined Accuracy (%)", fontsize=10)
    ax2.set_title("Absolute Accuracy by Needle", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=9)

    # Console insight.
    for hop in ["onehop", "twohop"]:
        sh = summary[summary["reasoning_hop"] == hop]
        if sh.empty:
            continue
        hard = sh.sort_values("rel_delta_pp").iloc[0]
        easy = sh.sort_values("rel_delta_pp", ascending=False).iloc[0]
        print(f"  Needle insight ({hop}): hardest={hard['needle_id']} ({hard['rel_delta_pp']:+.1f} pp), "
              f"easiest={easy['needle_id']} ({easy['rel_delta_pp']:+.1f} pp)")

    fig.suptitle("Needle Family Deep Dive", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _savefig(fig, "needle_family_performance.png", PLOTS_DIR)


# ─── Plot 6.8: Variant Sensitivity (T01 vs T04) ──────────────────────────────

def plot_variant_sensitivity(df: pd.DataFrame, out_dir: Path):
    """Per-context variant effect distribution: delta = T04 - T01."""
    sub = df[(~df["is_error"]) & (df["test_name"].isin(CANONICAL_TESTS_T01_T04))].copy()
    if sub.empty:
        print("  ⚠ No canonical rows for variant plot – skipping variant_sensitivity.png")
        return

    sub = _with_test_meta(sub)
    sub = sub[sub["variant"].isin(VARIANT_ORDER)].copy()
    if sub.empty:
        print("  ⚠ No parseable variants – skipping variant_sensitivity.png")
        return

    # Context-level cells per variant.
    cell = (sub.groupby(["model", "reasoning_hop", "variant", "context_length"])
            .agg(acc=("both_correct", "mean"), n=("both_correct", "count"))
            .reset_index())
    cell = cell[cell["n"] >= MIN_SAMPLES].copy()
    if cell.empty:
        print("  ⚠ No cells pass sample threshold – skipping variant_sensitivity.png")
        return

    piv = (cell.pivot_table(index=["model", "reasoning_hop", "context_length"],
                            columns="variant", values="acc", aggfunc="mean")
           .reset_index())
    piv = piv.dropna(subset=["T01", "T04"]).copy()
    if piv.empty:
        print("  ⚠ No shared T01/T04 context cells – skipping variant_sensitivity.png")
        return
    piv["delta_t04_minus_t01"] = 100.0 * (piv["T04"] - piv["T01"])

    context_csv = TABLES_DIR / "variant_sensitivity_context_deltas.csv"
    piv.to_csv(context_csv, index=False)

    summary = (piv.groupby(["model", "reasoning_hop"])
               .agg(
                   mean_delta_pp=("delta_t04_minus_t01", "mean"),
                   median_delta_pp=("delta_t04_minus_t01", "median"),
                   iqr_pp=("delta_t04_minus_t01", lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
                   n_contexts=("context_length", "nunique"),
               )
               .reset_index())
    metrics_csv = TABLES_DIR / "variant_sensitivity_metrics.csv"
    summary.to_csv(metrics_csv, index=False)
    print(f"  ✓ Variant context-delta CSV: {context_csv} ({len(piv)} rows)")
    print(f"  ✓ Variant sensitivity metrics CSV: {metrics_csv} ({len(summary)} rows)")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6), sharey=True)
    rng = np.random.default_rng(RANDOM_SEED)
    for ax_idx, hop in enumerate(["onehop", "twohop"]):
        ax = axes[ax_idx]
        ph = piv[piv["reasoning_hop"] == hop].copy()
        if ph.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="#888")
            ax.set_title(hop.capitalize(), fontsize=12, fontweight="bold")
            continue

        sm = summary[summary["reasoning_hop"] == hop].sort_values("median_delta_pp")
        models = sm["model"].tolist()
        data = [ph[ph["model"] == m]["delta_t04_minus_t01"].values for m in models]
        pos = np.arange(1, len(models) + 1)
        bp = ax.boxplot(data, positions=pos, patch_artist=True, widths=0.6, showfliers=False)
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(color_of(m))
            patch.set_alpha(0.35)
            patch.set_edgecolor(color_of(m))
        for key in ["whiskers", "caps", "medians"]:
            for item in bp[key]:
                item.set_color("#555")
                item.set_linewidth(1.2 if key == "medians" else 1.0)

        # Jittered points for distribution visibility.
        for i, m in enumerate(models, start=1):
            vals = ph[ph["model"] == m]["delta_t04_minus_t01"].values
            if len(vals) == 0:
                continue
            xj = i + rng.normal(0, 0.045, size=len(vals))
            ax.scatter(xj, vals, s=10, color=color_of(m), alpha=0.35, edgecolors="none")

        ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
        ax.set_xticks(pos)
        ax.set_xticklabels([dname(m) for m in models], rotation=30, ha="right", fontsize=8.5)
        ax.set_ylabel("T04 - T01 (pp)", fontsize=10)
        ax.set_title(f"{hop.capitalize()} – Variant Effect Distribution", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    # Console insight.
    for hop in ["onehop", "twohop"]:
        sh = summary[summary["reasoning_hop"] == hop]
        if sh.empty:
            continue
        worst = sh.sort_values("median_delta_pp").iloc[0]
        best = sh.sort_values("median_delta_pp", ascending=False).iloc[0]
        print(f"  Variant insight ({hop}): median T04 drop={dname(worst['model'])} "
              f"({worst['median_delta_pp']:+.1f} pp), median T04 gain={dname(best['model'])} "
              f"({best['median_delta_pp']:+.1f} pp)")

    fig.suptitle("Variant Sensitivity (Per-Context Delta Distribution)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _savefig(fig, "variant_sensitivity.png", PLOTS_DIR)


# ─── Plot 7: Answer vs Evidence ──────────────────────────────────────────────

def plot_answer_vs_evidence(df: pd.DataFrame, out_dir: Path):
    """Per-model line plots comparing answer acc vs evidence acc over context."""
    models = _sorted_models(df)
    n = len(models)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharey=True)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i, model in enumerate(models):
        ax = axes[i // cols][i % cols]
        sub = df[df["model"] == model]
        agg = sub.groupby("context_length").agg(
            ans=("answer_correct", lambda x: 100.0 * x.mean()),
            evi=("evidence_correct", lambda x: 100.0 * x.mean()),
            n=("both_correct", "count"),
        ).reset_index().sort_values("context_length")
        agg = agg[agg["n"] >= MIN_SAMPLES]

        if agg.empty:
            ax.text(0.5, 0.5, "Insufficient\nsamples", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="#CCC", fontweight="bold")
        else:
            ax.plot(agg["context_length"], agg["ans"], color=PALETTE[0],
                    marker="o", linewidth=2.5, label="Answer")
            ax.plot(agg["context_length"], agg["evi"], color=PALETTE[3],
                    marker="s", linestyle="--", linewidth=2, label="Evidence")
            ax.fill_between(agg["context_length"], agg["ans"], agg["evi"],
                            color="gray", alpha=0.1)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, _: fmt_ctx(int(v))))

        ax.set_ylim(-2, 105)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title(dname(model), fontsize=11, fontweight="bold",
                     color=color_of(model))
        if i >= (rows - 1) * cols:
            ax.set_xlabel("Context Length", fontsize=9)
        if i % cols == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=9)

    # Hide unused
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Answer vs. Evidence Accuracy per Model",
                 fontsize=14, fontweight="bold", y=1.07)
    plt.tight_layout()
    _savefig(fig, "answer_vs_evidence.png", PLOTS_DIR)


# ─── Plot 7: Per-Model Accuracy Curves ───────────────────────────────────────

def plot_per_model_curves(df: pd.DataFrame, out_dir: Path):
    """One subplot per model showing onehop and twohop combined accuracy vs context."""
    models = _sorted_models(df)
    if not models:
        print("  ⚠ No models available – skipping per_model_curves.png")
        return

    n = len(models)
    cols = min(2, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.4, rows * 4.6),
                             squeeze=False, sharex=False, sharey=False)

    ctx_all = sorted(df["context_length"].dropna().unique().tolist())
    if not ctx_all:
        print("  ⚠ No context lengths available – skipping per_model_curves.png")
        return
    x_min, x_max = min(ctx_all), max(ctx_all)
    tick_candidates = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    x_ticks = [t for t in tick_candidates if x_min <= t <= x_max]
    if len(x_ticks) < 3:
        x_ticks = sorted(set(ctx_all))

    for idx, model in enumerate(models):
        ax = axes[idx // cols][idx % cols]
        sub = df[df["model"] == model]
        has_data = False

        for hop, ls, marker, alpha in [
            ("onehop", "-", "o", 1.0),
            ("twohop", "--", "s", 0.85),
        ]:
            sh = sub[sub["reasoning_hop"] == hop]
            agg = sh.groupby("context_length").agg(
                acc=("both_correct", lambda x: 100.0 * x.mean()),
                n=("both_correct", "count"),
            ).reset_index()
            agg = agg[agg["n"] >= MIN_SAMPLES].sort_values("context_length")
            if agg.empty:
                continue
            has_data = True
            ax.plot(agg["context_length"], agg["acc"], marker=marker, markersize=3.5,
                    linewidth=2, linestyle=ls, color=color_of(model),
                    label=hop.capitalize(), alpha=alpha)

        ax.set_ylim(-2, 105)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(True, alpha=0.3, which="major")
        ax.set_title(dname(model), fontsize=11, fontweight="bold",
                     color=color_of(model))

        if has_data:
            ax.set_xscale("log")
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(x_ticks)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, _: fmt_ctx(int(v))))
            ax.legend(fontsize=8, loc="lower left")
            ax.tick_params(axis="both", labelsize=9)
        else:
            ax.text(0.5, 0.5, "Insufficient\nsamples", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="#CCC",
                    fontweight="bold")
            ax.set_xticks([])
        ax.set_xlabel("Context Length", fontsize=9)
        ax.set_ylabel("Combined Accuracy (%)", fontsize=9)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(f"Per-Model Accuracy vs Context Length (n >= {MIN_SAMPLES} per context)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig(fig, "per_model_curves.png", PLOTS_DIR)


# ─── Plot 8: Book Comparison (Per-Model) ─────────────────────────────────────

def plot_book_comparison(df_random: pd.DataFrame, df_book1: pd.DataFrame,
                         out_dir: Path):
    """Per-model subplots: random book (solid) vs book 1 (dashed) accuracy."""
    all_models = sorted(
        set(df_random["model"].unique()) | set(df_book1["model"].unique()),
        key=lambda m: dname(m))
    # Gemini runs here have only one book, so random-vs-book1 comparison is not meaningful.
    all_models = [m for m in all_models if not m.startswith("gemini-")]
    if not all_models:
        print("  ⚠ No non-Gemini models available – skipping book_comparison.png")
        return

    n = len(all_models)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4),
                             squeeze=False, sharey=True)

    for idx, model in enumerate(all_models):
        ax = axes[idx // cols][idx % cols]
        has_data = False

        for label, df_src, ls, alpha in [
            ("Random Book", df_random, "-", 1.0),
            ("Book 1", df_book1, "--", 0.6),
        ]:
            sm = df_src[(df_src["model"] == model) & (~df_src["is_error"])]
            agg = sm.groupby("context_length").agg(
                acc=("both_correct", lambda x: 100.0 * x.mean()),
                n=("both_correct", "count"),
            ).reset_index()
            agg = agg[agg["n"] >= 4]  # lower threshold for book-1
            if agg.empty:
                continue
            has_data = True
            ax.plot(agg["context_length"], agg["acc"], marker="o", markersize=3,
                    linewidth=2, linestyle=ls, color=color_of(model),
                    label=label, alpha=alpha)

        ax.set_ylim(-2, 105)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title(dname(model), fontsize=11, fontweight="bold",
                     color=color_of(model))

        if has_data:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, _: fmt_ctx(int(v))))
            ax.legend(fontsize=7, loc="lower left")
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="#CCC",
                    fontweight="bold")
            ax.set_xticks([])

        if idx >= (rows - 1) * cols:
            ax.set_xlabel("Context Length", fontsize=9)
        if idx % cols == 0:
            ax.set_ylabel("Combined Accuracy (%)", fontsize=9)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Book Selection Effect: Random vs. Book 1 (Per Model)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig(fig, "book_comparison.png", PLOTS_DIR)


# ─── Plot 10: Character Bias ─────────────────────────────────────────────────

def plot_character_bias(df: pd.DataFrame, out_dir: Path):
    """Matched character comparison with setup held constant.

    Keeps experiment setup fixed at (model, context, hop, test, depth), then compares
    only setups shared by all selected characters. Books are averaged within each
    setup+character cell to avoid book-composition confounding.
    """
    if "selected_character" not in df.columns:
        return

    df_char = df[(df["selected_character"] != "Unknown") & (~df["is_error"])].copy()
    if df_char.empty:
        return

    # Restrict to canonical publication tests to keep task setup stable.
    df_char = df_char[df_char["test_name"].isin(CANONICAL_TESTS_T01_T04)].copy()
    if df_char.empty:
        print("  ⚠ No canonical test rows for character analysis – skipping character_bias.png")
        return

    # Export raw per-character coverage before matching.
    raw_cov = (df_char.groupby("selected_character")
               .agg(
                   n_points_raw=("both_correct", "count"),
                   n_contexts_raw=("context_length", "nunique"),
                   n_models_raw=("model", "nunique"),
               )
               .reset_index()
               .sort_values("n_points_raw", ascending=False))
    raw_cov_path = TABLES_DIR / "character_bias_raw_coverage.csv"
    raw_cov.to_csv(raw_cov_path, index=False)
    print(f"  ✓ Raw character coverage CSV: {raw_cov_path}")

    # Use rounded depth to align 0.333333... and similar float variants.
    df_char["depth_r6"] = df_char["depth"].round(6)
    setup_cols = ["model", "context_length", "reasoning_hop", "test_name", "depth_r6"]

    # Average across books within the same setup+character cell.
    cell = (df_char.groupby(setup_cols + ["selected_character"])
            .agg(acc=("both_correct", "mean"),
                 n_raw=("both_correct", "count"))
            .reset_index())
    if cell.empty:
        print("  ⚠ Character cell table empty – skipping character_bias.png")
        return

    # Build setup sets per character and choose a matched subset.
    # We auto-search context caps and character subsets, maximizing matched points
    # while keeping at least 3 characters to preserve a meaningful comparison.
    cell["setup_tuple"] = list(map(tuple, cell[setup_cols].itertuples(index=False, name=None)))
    max_chars = min(6, cell["selected_character"].nunique())
    min_chars = min(4, max_chars)
    if max_chars < 2:
        print("  ⚠ Not enough characters for comparison – skipping character_bias.png")
        return

    context_caps = sorted(cell["context_length"].unique())
    best_config = None
    # best_config tuple:
    # (n_shared_setups, n_chars, -cap, cap, chars_list, shared_setup_set)
    for cap in context_caps:
        cell_cap = cell[cell["context_length"] <= cap]
        char_to_setups = {
            ch: set(grp["setup_tuple"].tolist())
            for ch, grp in cell_cap.groupby("selected_character")
        }
        chars = sorted(char_to_setups.keys(), key=lambda c: len(char_to_setups[c]), reverse=True)
        if len(chars) < min_chars:
            continue

        for k in range(max_chars, min_chars - 1, -1):
            if len(chars) < k:
                continue
            for comb in combinations(chars, k):
                shared = set.intersection(*(char_to_setups[c] for c in comb))
                if len(shared) < MIN_SAMPLES:
                    continue
                cand = (len(shared), k, -cap, cap, list(comb), shared)
                if (best_config is None) or (cand[:3] > best_config[:3]):
                    best_config = cand

    if best_config is None:
        print("  ⚠ No matched character subset with sufficient shared setups – skipping character_bias.png")
        return

    _, _, _, selected_cap, best_chars, best_shared = best_config
    matched = cell[
        cell["selected_character"].isin(best_chars)
        & cell["setup_tuple"].isin(best_shared)
        & (cell["context_length"] <= selected_cap)
    ].copy()
    if matched.empty:
        print("  ⚠ Matched character frame empty – skipping character_bias.png")
        return

    char_agg = (matched.groupby("selected_character")
                .agg(
                    acc=("acc", lambda x: 100.0 * x.mean()),
                    n_points=("setup_tuple", "nunique"),
                    n_contexts=("context_length", "nunique"),
                    n_models=("model", "nunique"),
                )
                .reset_index()
                .sort_values("acc"))

    # Coverage report: data points and contexts per character.
    coverage_path = TABLES_DIR / "character_bias_coverage.csv"
    coverage_out = char_agg.rename(columns={"selected_character": "character"}).copy()
    coverage_out.to_csv(coverage_path, index=False)
    chosen_chars_txt = ", ".join(best_chars)
    print(f"  Character matching config: contexts <= {fmt_ctx(int(selected_cap))}, chars=[{chosen_chars_txt}]")
    print("  Character coverage (matched setup):")
    for row in coverage_out.itertuples(index=False):
        print(f"    {row.character:<10} points={int(row.n_points):4d}  contexts={int(row.n_contexts):2d}  models={int(row.n_models):2d}  acc={row.acc:.1f}%")
    print(f"  ✓ Coverage CSV: {coverage_path}")

    fig, ax = plt.subplots(figsize=(max(8, len(char_agg) * 1.0), 6))
    x = np.arange(len(char_agg))
    colors = plt.cm.Set2(np.linspace(0, 1, len(char_agg)))
    ax.bar(x, char_agg["acc"], color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(char_agg["selected_character"], rotation=30, ha="right",
                       fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Combined Accuracy (%)", fontsize=10)
    points_each = int(char_agg["n_points"].min())
    ctx_each = int(char_agg["n_contexts"].min())
    ax.set_title(f"Performance by Character (matched setup, {len(best_chars)} chars, "
                 f"context <= {fmt_ctx(int(selected_cap))}, n={points_each} each, {ctx_each} contexts each)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for i, row in enumerate(char_agg.itertuples()):
        ax.text(i, row.acc + 1, f"{row.acc:.1f}%", ha="center", fontsize=9,
                fontweight="bold")

    fig.suptitle("Character Bias Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _savefig(fig, "character_bias.png", PLOTS_DIR)


# ─── Console Summary ─────────────────────────────────────────────────────────

def print_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                  df_raw_full_for_csv: pd.DataFrame, out_dir: Path):
    """Print key findings and save CSV."""
    models = _sorted_models(df_clean)

    print("\n" + "=" * 80)
    print("  RESEARCH SUMMARY – NoLiMa Long-Context Evaluation")
    print("=" * 80)

    print(f"\n  Metrics are context-weighted (each context length = equal weight)")
    print(f"\n{'Model':<25} {'Hop':<10} {'Combined':>9} {'Answer':>8} {'Evidence':>9} {'N':>6}")
    print("-" * 72)

    for model in models:
        clean_m = df_clean[df_clean["model"] == model]
        for hop in ["onehop", "twohop"]:
            ch = clean_m[clean_m["reasoning_hop"] == hop]
            if ch.empty:
                continue
            cell = ch.groupby("context_length").agg(
                comb=("both_correct", "mean"),
                ans_=("answer_correct", "mean"),
                evi_=("evidence_correct", "mean"),
            )
            comb = 100 * cell["comb"].mean()
            ans = 100 * cell["ans_"].mean()
            evi = 100 * cell["evi_"].mean()
            n = len(ch)
            print(f"  {dname(model):<23} {hop:<10} {comb:>8.1f}% {ans:>7.1f}% {evi:>8.1f}% {n:>6}")

    print("=" * 72)

    # Key findings
    print("\n📊 KEY FINDINGS:")
    overall = _ctx_weighted_accuracy(df_clean, "both_correct").sort_values(ascending=False)
    print(f"  1. Best model (ctx-weighted): {dname(overall.index[0])} ({overall.iloc[0]:.1f}%)")

    hop_agg = df_clean.groupby("reasoning_hop")["both_correct"].mean()
    if "onehop" in hop_agg and "twohop" in hop_agg:
        gap = 100 * (hop_agg["onehop"] - hop_agg["twohop"])
        print(f"  2. Onehop vs Twohop gap: {gap:+.1f} pp")

    short = df_clean[df_clean["context_length"] <= 50_000]["both_correct"].mean()
    long_ = df_clean[df_clean["context_length"] >= 500_000]["both_correct"].mean()
    if long_ > 0:
        print(f"  3. Short (≤50K) vs Long (≥500K): {100*short:.1f}% vs {100*long_:.1f}% "
              f"({100*(short - long_):+.1f} pp)")

    df_d = df_clean.dropna(subset=["depth"])
    if not df_d.empty:
        edge = df_d[(df_d["depth"] < 0.20) | (df_d["depth"] > 0.80)]["both_correct"].mean()
        mid = df_d[(df_d["depth"] >= 0.20) & (df_d["depth"] <= 0.80)]["both_correct"].mean()
        print(f"  4. Lost-in-middle: edge={100*edge:.1f}%, middle={100*mid:.1f}% "
              f"({100*(edge - mid):+.1f} pp)")

    errs = df_raw["is_error"].sum()
    print(f"  5. Data quality: {errs}/{len(df_raw)} API errors ({100*errs/len(df_raw):.1f}%)")
    print()

    # Save CSV – per-needle granular results:
    # one row per (model, context, reasoning_hop, needle_id, depth).
    # 16 rows per (model, context, hop): 4 needles × 4 depths.
    # Strict source: Book 1, T01, canonical characters (Yuki/Stuart).
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []

    # Book 1 only, non-error, canonical tests only, correct characters
    df_full = df_raw_full_for_csv[
        (df_raw_full_for_csv["book"].astype(str) == "1")
        & (~df_raw_full_for_csv["is_error"])
        & (df_raw_full_for_csv["test_name"].isin(CANONICAL_TESTS_T01))
    ].copy()
    df_full = _filter_correct_characters(df_full)

    # Extract needle_id from test_name (e.g. "0402_T01_C02_onehop" → "0402")
    df_full["needle_id"] = df_full["test_name"].str.extract(
        r"^(0402Inv|0402|0405Inv|0405)_", expand=False
    )

    # Snap depths to standard 4
    depths_arr = df_full["depth"].values
    snapped = np.full(len(depths_arr), np.nan)
    for d in STRICT_DEPTHS_4:
        mask = np.abs(depths_arr - d) <= 0.05
        snapped[mask] = d
    df_full["depth_std"] = snapped
    df_full = df_full.dropna(subset=["depth_std"])

    for model in _sorted_models(df_full):
        model_df = df_full[df_full["model"] == model]
        ctx_lengths = sorted(model_df["context_length"].unique())
        for ctx in ctx_lengths:
            ctx_df = model_df[model_df["context_length"] == ctx]
            for hop in ["onehop", "twohop"]:
                hop_df = ctx_df[ctx_df["reasoning_hop"] == hop]
                if hop_df.empty:
                    continue
                for needle in NEEDLE_ORDER:
                    needle_df = hop_df[hop_df["needle_id"] == needle]
                    if needle_df.empty:
                        continue
                    for d, grp in needle_df.groupby("depth_std"):
                        csv_rows.append({
                            "model": dname(model),
                            "context_length": int(ctx),
                            "reasoning_hop": hop,
                            "needle_id": needle,
                            "character": NEEDLE_CHARACTER[needle],
                            "depth": round(d * 100),
                            "combined_accuracy": round(100 * grp["both_correct"].mean(), 2),
                            "answer_accuracy": round(100 * grp["answer_correct"].mean(), 2),
                            "evidence_accuracy": round(100 * grp["evidence_correct"].mean(), 2),
                        })

    csv_path = TABLES_DIR / "results_summary_per_needle.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  ✓ Summary CSV (book1 + T01, per-needle): {csv_path} ({len(csv_rows)} rows)")


      # Save CSV – accuracy averaged across needle types + characters,
    # grouped by (model, context, reasoning_hop, depth).
    # Strict source: full raw data, Book 1 only, T01 only.
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []

    # Book 1 only, non-error, canonical tests only, correct characters
    df_full = df_raw_full_for_csv[
        (df_raw_full_for_csv["book"].astype(str) == "1")
        & (~df_raw_full_for_csv["is_error"])
        & (df_raw_full_for_csv["test_name"].isin(CANONICAL_TESTS_T01))
    ].copy()
    df_full = _filter_correct_characters(df_full)

    # Snap depths to standard 4
    depths_arr = df_full["depth"].values
    snapped = np.full(len(depths_arr), np.nan)
    for d in STRICT_DEPTHS_4:
        mask = np.abs(depths_arr - d) <= 0.05
        snapped[mask] = d
    df_full["depth_std"] = snapped
    df_full = df_full.dropna(subset=["depth_std"])

    for model in _sorted_models(df_full):
        model_df = df_full[df_full["model"] == model]
        ctx_lengths = sorted(model_df["context_length"].unique())
        for ctx in ctx_lengths:
            ctx_df = model_df[model_df["context_length"] == ctx]
            for hop in ["onehop", "twohop"]:
                hop_df = ctx_df[ctx_df["reasoning_hop"] == hop]
                if hop_df.empty:
                    continue
                for d, grp in hop_df.groupby("depth_std"):
                    csv_rows.append({
                        "model": dname(model),
                        "context_length": int(ctx),
                        "reasoning_hop": hop,
                        "depth": round(d * 100),
                        "combined_accuracy": round(100 * grp["both_correct"].mean(), 2),
                        "answer_accuracy": round(100 * grp["answer_correct"].mean(), 2),
                        "evidence_accuracy": round(100 * grp["evidence_correct"].mean(), 2),
                    })

    csv_path = TABLES_DIR / "results_summary_grouped.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  ✓ Summary CSV (book1 + T01 only): {csv_path} ({len(csv_rows)} rows)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = PLOTS_DIR  # legacy alias; per-extension routing handled at write sites

    print("=" * 60)
    print("  NoLiMa Research Analysis")
    print("=" * 60)

    # 1. Load all raw data
    print("\n1. Loading results ...")
    df_raw_full = load_raw()
    n_total = len(df_raw_full)
    n_models = df_raw_full["model"].nunique()
    n_ctx = df_raw_full["context_length"].nunique()
    print(f"   {n_total:,} results across {n_models} models, {n_ctx} context lengths")

    # 2. Build STRICT dataset for standard plots
    #    (Book 1, T01 only, 4 depths, Yuki for 0402/0402Inv, Stuart for 0405/0405Inv, no API errors)
    print("\n2. Building strict dataset (Book 1, T01, 4 depths, Yuki/Stuart) ...")
    df_strict = _build_strict_dataset(df_raw_full)
    print(f"   {len(df_strict):,} valid rows ({df_strict['model'].nunique()} models, "
          f"{df_strict['context_length'].nunique()} contexts)")

    # 3. Build datasets for EXEMPT comparison plots
    #    These plots deliberately compare different configurations, so they
    #    use broader data (T01+T04, random books, multiple characters, etc.)
    print("\n3. Building datasets for comparison plots ...")
    df_raw_full_pub = _filter_tests(df_raw_full, CANONICAL_TESTS_T01_T04)
    df_raw_sampled = _sample_one_book_per_cell(df_raw_full_pub)
    df_clean_sampled = df_raw_sampled[~df_raw_sampled["is_error"]].copy()
    df_raw_book1 = _filter_strict_book1_4depths(df_raw_full_pub)
    n_err = df_raw_sampled["is_error"].sum()
    print(f"   Sampled (T01+T04): {len(df_clean_sampled):,} valid ({n_err:,} errors filtered)")
    print(f"   Book-1 (T01+T04):  {len(df_raw_book1):,} rows")

    # 4. Generate STANDARD plots (strict setup: Book 1, T01, Yuki/Stuart)
    print("\n4. Generating standard plots (strict setup) ...")
    plot_accuracy_heatmap(df_strict, out_dir)
    plot_scaling_curves(df_strict, out_dir)
    plot_onehop_vs_twohop(df_strict, out_dir)
    plot_depth_analysis(df_strict, out_dir)
    plot_model_ranking(df_strict, out_dir)
    plot_context_stress_test(df_strict, out_dir)
    plot_failure_mode_breakdown(df_strict, out_dir)
    plot_answer_vs_evidence(df_strict, out_dir)
    plot_per_model_curves(df_strict, out_dir)

    # 5. Generate EXEMPT plots (deliberate comparison experiments)
    #    - needle_family_performance: needle × character confounded under strict setup
    #    - variant_sensitivity:       T01 vs T04 comparison
    #    - book_comparison:           random book vs book 1
    #    - character_bias:            character comparison
    print("\n5. Generating comparison plots (exempt from strict setup) ...")
    plot_needle_family_performance(df_clean_sampled, out_dir)
    plot_variant_sensitivity(df_clean_sampled, out_dir)
    plot_book_comparison(df_raw_sampled, df_raw_book1, out_dir)
    plot_character_bias(df_raw_full_pub, out_dir)

    # 6. Summary (uses strict dataset for main metrics)
    print("\n6. Summary")
    print_summary(df_raw_sampled, df_strict, df_raw_full_pub, out_dir)

    print(f"\nAll outputs saved to: {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
