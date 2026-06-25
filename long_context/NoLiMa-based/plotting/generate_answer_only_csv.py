#!/usr/bin/env python3
"""
Generate results_summary CSV for the answer-only experiment.

Reads the answer-only results from:
  evaluation/results_claude-sonnet-4-5-20250929-answer-only/

Produces a CSV with the exact same columns and format as the main
research_analysis results_summary.csv:

  model, context_length, reasoning_hop, depth,
  combined_accuracy, answer_accuracy, evidence_accuracy

Filtering logic matches research_analysis.print_summary:
  - Book 1 only
  - T01 canonical tests only (8 test names)
  - Depths snapped to 4 standard positions (0%, 33%, 67%, 100%)
  - API errors excluded
  - Accuracy = mean over all matching entries per (ctx, hop, depth) cell

Usage:
  python generate_answer_only_csv.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "evaluation" / "special_experiments" / "results_claude-sonnet-4-5-20250929-answer-only"
REASONING_TYPE = "commonsense_knowledge"
OUTPUT_DIR = PROJECT_DIR / "results" / "answer_only"
TABLES_DIR = OUTPUT_DIR / "tables"
OUTPUT_CSV = TABLES_DIR / "answer_only_results_summary.csv"

MODEL_DISPLAY_NAME = "Claude Sonnet 4.5 (answer-only)"

CANONICAL_TESTS_T01 = {
    "0402_T01_C02_onehop", "0402_T01_C02_twohop",
    "0405_T01_C02_onehop", "0405_T01_C02_twohop",
    "0402Inv_T01_C02_onehop", "0402Inv_T01_C02_twohop",
    "0405Inv_T01_C02_onehop", "0405Inv_T01_C02_twohop",
}

STRICT_BOOK = "1"
STRICT_DEPTHS_4 = (0.0, 0.33, 0.67, 1.0)


# ─── Helpers (mirrored from research_analysis.py) ────────────────────────────

def _is_api_error(r: dict) -> bool:
    if r.get("error") or r.get("error_type"):
        return True
    if r.get("response") is None and r.get("input_tokens", 0) == 0:
        return True
    return False


def _reparse_response(resp):
    if resp is None:
        return None
    if isinstance(resp, dict):
        return resp
    if not isinstance(resp, str) or len(resp.strip()) < 3:
        return None
    text = re.sub(r",\s*}", "}", resp)
    text = re.sub(r",\s*]", "]", text)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    m = re.search(r'\{[^{}]*"answer"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidate = re.sub(r",\s*}", "}", m.group(0))
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _rescore_entry(r: dict, reparsed: dict) -> tuple:
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


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_answer_only_results() -> pd.DataFrame:
    """Load all depth-level results from the answer-only experiment."""
    comm_dir = RESULTS_DIR / REASONING_TYPE
    if not comm_dir.is_dir():
        print(f"ERROR: Results directory not found: {comm_dir}")
        sys.exit(1)

    rows = []
    stats = {"reparsed": 0, "rescored": 0, "api_errors": 0, "files": 0}

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
                stats["files"] += 1

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
                        "model": "claude-sonnet-4-5-20250929-answer-only",
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
                    })

    print(f"  Loaded {len(rows)} depth-level entries from {stats['files']} files")
    if stats["reparsed"]:
        print(f"  Re-parsed {stats['reparsed']} string responses")
    if stats["rescored"]:
        print(f"  Rescored {stats['rescored']} entries")
    if stats["api_errors"]:
        print(f"  {stats['api_errors']} API errors (will be filtered)")

    if not rows:
        print("ERROR: No results found.")
        sys.exit(1)

    return pd.DataFrame(rows)


# ─── CSV Generation ──────────────────────────────────────────────────────────

def generate_csv(df: pd.DataFrame) -> None:
    """Generate results_summary CSV with the exact same format."""

    # Filter: book 1, non-error, T01 canonical tests only
    df_full = df[
        (df["book"].astype(str) == STRICT_BOOK)
        & (~df["is_error"])
        & (df["test_name"].isin(CANONICAL_TESTS_T01))
    ].copy()

    if df_full.empty:
        print("ERROR: No valid book-1 / T01 entries found after filtering.")
        sys.exit(1)

    print(f"  After filtering (book 1, T01, no errors): {len(df_full)} entries")

    # Snap depths to standard 4
    depths_arr = df_full["depth"].values
    snapped = np.full(len(depths_arr), np.nan)
    for d in STRICT_DEPTHS_4:
        mask = np.abs(depths_arr - d) <= 0.05
        snapped[mask] = d
    df_full["depth_std"] = snapped
    df_full = df_full.dropna(subset=["depth_std"])

    print(f"  After depth snapping: {len(df_full)} entries")

    # Build CSV rows
    csv_rows = []
    ctx_lengths = sorted(df_full["context_length"].unique())

    for ctx in ctx_lengths:
        ctx_df = df_full[df_full["context_length"] == ctx]
        for hop in ["onehop", "twohop"]:
            hop_df = ctx_df[ctx_df["reasoning_hop"] == hop]
            if hop_df.empty:
                continue
            for d, grp in hop_df.groupby("depth_std"):
                csv_rows.append({
                    "model": MODEL_DISPLAY_NAME,
                    "context_length": int(ctx),
                    "reasoning_hop": hop,
                    "depth": round(d * 100),
                    "combined_accuracy": 0.0,
                    "answer_accuracy": round(100 * grp["answer_correct"].mean(), 2),
                    "evidence_accuracy": 0.0,
                })

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(OUTPUT_CSV, index=False)
    print(f"\n  CSV saved: {OUTPUT_CSV} ({len(csv_rows)} rows)")

    # Quick summary
    result_df = pd.DataFrame(csv_rows)
    for hop in ["onehop", "twohop"]:
        sub = result_df[result_df["reasoning_hop"] == hop]
        if not sub.empty:
            print(
                f"  {hop:>8s}: answer_accuracy={sub['answer_accuracy'].mean():.1f}%  "
                f"({len(sub)} cells)"
            )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Answer-Only Experiment — Results Summary CSV")
    print("=" * 60)

    print("\n1. Loading answer-only results ...")
    df = load_answer_only_results()

    print("\n2. Generating CSV ...")
    generate_csv(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
