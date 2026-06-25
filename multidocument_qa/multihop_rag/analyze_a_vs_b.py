"""
Cross-experiment analysis: join Experiment A (EBG outputs) with Experiment B
(per-query retrieval ranks) and report long-context EBG accuracy stratified
by retrieval difficulty.

For each Experiment A output JSONL we:
  1. join records by ``qid`` with the per-fact ranks in Experiment B's
     ``fact_ranks.jsonl`` for a chosen embedding config;
  2. compute the *worst* gold-fact rank per query (the most-buried hop);
  3. bin queries by worst-rank bands (e.g. 1-10, 11-50, 51-200, 201+) and
     report Ans / Evid-strict / Joint per band.

This isolates whether EBG long-context accuracy holds up on queries where
RAG would have missed at least one hop at standard top-k budgets.
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def worst_rank_by_qid(fact_ranks_path: str) -> dict[int, int]:
    """Return {qid: max(best_rank across facts)}."""
    out: dict[int, int] = {}
    for rec in load_jsonl(fact_ranks_path):
        qid = rec["qid"]
        r = rec["best_rank"]
        out[qid] = max(out.get(qid, 0), r)
    return out


BANDS = [(1, 10), (11, 50), (51, 200), (201, 10**9)]


def band_label(r: int) -> str:
    for lo, hi in BANDS:
        if lo <= r <= hi:
            return f"{lo}-{hi if hi < 10**9 else '+'}"
    return "unknown"


def analyze_one(exp_a_path: str, worst_rank: dict[int, int]) -> None:
    recs = load_jsonl(exp_a_path)
    answer_only = bool(recs[0].get("answer_only")) if recs else False
    name = os.path.basename(exp_a_path)
    print(f"\n=== {name} (mode={'answer-only' if answer_only else 'EBG'}) ===")

    bands: dict[str, list[dict]] = defaultdict(list)
    n_no_rank = 0
    for r in recs:
        if r.get("error"):
            continue
        qid = r["qid"]
        if qid not in worst_rank:
            n_no_rank += 1
            continue
        bands[band_label(worst_rank[qid])].append(r)

    cols = ["band", "n", "Ans", "Evid", "Joint"]
    print(f"  {' '.join(f'{c:>10}' for c in cols)}")
    for lo, hi in BANDS:
        lbl = f"{lo}-{hi if hi < 10**9 else '+'}"
        rs = bands.get(lbl, [])
        if not rs:
            continue
        ans = sum(1 for r in rs if r["answer_correct"]) / len(rs)
        ev = sum(1 for r in rs if r["evidence_strict"]) / len(rs) if not answer_only else float("nan")
        jt = sum(1 for r in rs if r["joint"]) / len(rs)
        ev_str = f"{ev:>10.2f}" if not answer_only else f"{'n/a':>10}"
        print(f"  {lbl:>10}{len(rs):>10}{ans:>10.2f}{ev_str}{jt:>10.2f}")
    if n_no_rank:
        print(f"  ({n_no_rank} queries had no rank record and were skipped)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fact-ranks",
                    default="retrieval_results/text-embedding-3-large_tokens256/fact_ranks.jsonl",
                    help="Path to Experiment B's per-fact rank JSONL.")
    ap.add_argument("--exp-a-paths", nargs="+", required=True,
                    help="Paths to Experiment A output JSONL files.")
    args = ap.parse_args()

    worst = worst_rank_by_qid(args.fact_ranks)
    print(f"Loaded worst-rank for {len(worst)} queries from {args.fact_ranks}")
    for p in args.exp_a_paths:
        analyze_one(p, worst)


if __name__ == "__main__":
    main()
