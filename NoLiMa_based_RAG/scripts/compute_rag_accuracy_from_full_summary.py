#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute per-topk binary-gated RAG accuracy from results_summary_full.csv and needle ranks."
    )
    p.add_argument(
        "--base-csv",
        default="NoLiMa_based_RAG/results/results_summary_full.csv",
        help="Path to results_summary_full.csv",
    )
    p.add_argument(
        "--needle-json",
        default="NoLiMa_based_RAG/results/NeedleRanking/needle_rank_summary.json",
        help="Path to needle_rank_summary.json",
    )
    p.add_argument(
        "--out-csv",
        default="NoLiMa_based_RAG/tables/rag_accuracy_full_binary_from_results_summary_full.csv",
        help="Output CSV path",
    )
    p.add_argument(
        "--missing-report-csv",
        default="NoLiMa_based_RAG/results/missing_rows_results_summary_full_vs_needle.csv",
        help="Output CSV for unmatched base rows",
    )
    return p.parse_args()


def to_i(v: str) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0


def to_f(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def main() -> None:
    args = parse_args()

    base_rows = list(csv.DictReader(open(args.base_csv, newline="")))
    needle_rows = json.load(open(args.needle_json))

    # Key: (context_length, task_id, question_key)
    needle_index = {}
    for n in needle_rows:
        ctx = n["source_file"].split("_")[0]
        key = (ctx, n["task_id"], n["question_key"])
        needle_index[key] = {
            "needle_rank_1based": int(n["needle_rank_1based"]),
            "total_vector_count": int(n.get("total_vector_count") or n.get("num_scored") or 0),
            "q_id": n["q_id"],
            "source_file": n["source_file"],
        }

    out = []
    missing = []

    for r in base_rows:
        ctx = r["context_length"].strip()
        task_id = r["needle_id"].strip()  # needle_id in base corresponds to task_id in needle JSON
        qk = r["reasoning_hop"].strip()   # reasoning_hop in base corresponds to question_key in needle JSON
        key = (ctx, task_id, qk)

        n = needle_index.get(key)
        if n is None:
            m = dict(r)
            m["missing_join_key"] = f"{ctx}|{task_id}|{qk}"
            m["missing_reason"] = "No matching (context_length, task_id, question_key) in needle json"
            missing.append(m)
            continue

        rank = n["needle_rank_1based"]
        total_vec = n["total_vector_count"]
        combined = to_f(r["combined_accuracy"])

        for topk in range(1, total_vec + 1):
            hit = 1 if rank <= topk else 0
            out.append(
                {
                    "model": r["model"],
                    "context_length": ctx,
                    "reasoning_hop": qk,
                    "needle_id": task_id,
                    "character": r["character"],
                    "depth": r["depth"],
                    "topk": topk,
                    "needle_rank_1based": rank,
                    "total_vector_count": total_vec,
                    "retrieval_hit_at_topk": hit,
                    "lc_combined_accuracy": f"{combined:.6f}",
                    "rag_combined_accuracy": f"{(combined * hit):.6f}",
                    "q_id": n["q_id"],
                    "source_file": n["source_file"],
                }
            )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_fields = [
        "model",
        "context_length",
        "reasoning_hop",
        "needle_id",
        "character",
        "depth",
        "topk",
        "needle_rank_1based",
        "total_vector_count",
        "retrieval_hit_at_topk",
        "lc_combined_accuracy",
        "rag_combined_accuracy",
        "q_id",
        "source_file",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(out)

    miss_path = Path(args.missing_report_csv)
    miss_path.parent.mkdir(parents=True, exist_ok=True)
    miss_fields = list(base_rows[0].keys()) + ["missing_join_key", "missing_reason"]
    with miss_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=miss_fields)
        w.writeheader()
        w.writerows(missing)

    print(f"Wrote: {out_path} ({len(out)} rows)")
    print(f"Wrote: {miss_path} ({len(missing)} rows)")


if __name__ == "__main__":
    main()

"""
Standard run (uses defaults — run from EBG_repo/):
python3 NoLiMa_based_RAG/scripts/compute_rag_accuracy_from_full_summary.py

Override all paths explicitly:
python3 NoLiMa_based_RAG/scripts/compute_rag_accuracy_from_full_summary.py \
  --base-csv   NoLiMa_based_RAG/results/results_summary_full.csv \
  --needle-json NoLiMa_based_RAG/results/NeedleRanking/needle_rank_summary.json \
  --out-csv    NoLiMa_based_RAG/tables/rag_accuracy_full_binary_from_results_summary_full.csv \
  --missing-report-csv NoLiMa_based_RAG/results/missing_rows_results_summary_full_vs_needle.csv
"""