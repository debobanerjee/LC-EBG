"""
Build per-query RAG haystacks at multiple K values from Experiment B's
retrieval artifacts.

For each query and each K in ``--ks`` we:
  1. score all corpus chunks by cosine similarity against the query embedding;
  2. take the top-K chunks in rank order;
  3. concatenate their text into a sentence-per-line numbered context (same
     line-granularity as ``build_haystack.py``);
  4. locate every gold ``fact`` inside that context to produce new
     ``evidence_lines`` (a fact may be absent if its chunk was not retrieved);
  5. emit one JSONL record per query per K, in the same schema as the
     long-context haystacks plus retrieval-coverage diagnostics.

Coverage diagnostics:

  - ``needles_total``         : number of distinct gold facts for the query.
  - ``needles_in_topk``       : number of those facts located in the top-K chunks
                                (a fact counts if any chunk containing it has
                                rank \\le K).
  - ``needles_evidence_lines``: number of facts that also got a usable line
                                number in the rendered haystack (\\le needles_in_topk;
                                drops below when a fact spans chunk boundaries
                                or is reformatted in a way ``find_fact_lines``
                                cannot recover).

Usage:

    python build_rag_haystack.py \\
        --qids-from experiment_outputs/multihop_rag_gpt-5_20260517_222901.jsonl \\
        --ks 4 10 25 50 100 200 \\
        --retrieval retrieval_results/text-embedding-3-large_tokens256 \\
        --out-dir processed/rag
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_haystack import find_fact_lines, split_into_lines  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def load_chunks(retrieval_dir: str) -> list[dict]:
    return load_jsonl(os.path.join(retrieval_dir, "chunks.jsonl"))


def load_embeddings(retrieval_dir: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    chunk_emb = np.load(os.path.join(retrieval_dir, "chunk_embeddings.npy"))
    query_emb = np.load(os.path.join(retrieval_dir, "query_embeddings.npy"))
    query_ids = json.load(open(os.path.join(retrieval_dir, "query_ids.json")))
    return chunk_emb, query_emb, query_ids


def render_chunks(chunks_top: list[dict]) -> tuple[str, list[str]]:
    """Concatenate retrieved chunks into a sentence-per-line numbered context.

    Each retrieved chunk gets a one-line header ``[CHUNK rank|title]`` so the
    model sees explicit chunk boundaries.
    """
    raw_lines: list[str] = []
    for rank, ch in enumerate(chunks_top, start=1):
        raw_lines.append(f"[CHUNK {rank} | {ch['title']}]")
        raw_lines.extend(split_into_lines(ch["text"]))
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(raw_lines))
    return numbered, raw_lines


def build_per_query(
    qrec: dict,
    qid_to_row: dict[int, int],
    query_emb: np.ndarray,
    chunk_emb: np.ndarray,
    chunks: list[dict],
    ks: Iterable[int],
    fact_to_chunks: dict[tuple[int, int], list[int]],
    qid_facts: dict[int, list[str]],
) -> list[dict]:
    qid = qrec["qid"]
    qrow = qid_to_row.get(qid)
    if qrow is None:
        return []

    sims = chunk_emb @ query_emb[qrow]
    order = np.argsort(-sims)
    rank_of = np.empty_like(order)
    rank_of[order] = np.arange(len(order))

    facts = qid_facts.get(qid, qrec["evidence_facts"])
    fact_chunk_sets = [
        set(fact_to_chunks.get((qid, fi), [])) for fi in range(len(facts))
    ]

    records = []
    for K in ks:
        top_ids = order[:K].tolist()
        top_chunks = [chunks[i] for i in top_ids]
        numbered, raw_lines = render_chunks(top_chunks)

        ev_lines: list[int] = []
        ev_facts: list[str] = []
        n_in_topk = 0
        for fi, fact in enumerate(facts):
            covered = bool(fact_chunk_sets[fi] & set(top_ids))
            if covered:
                n_in_topk += 1
            hits = find_fact_lines(fact, raw_lines)
            if hits:
                ev_lines.append(hits[0])
                ev_facts.append(fact)

        records.append({
            "qid": qid,
            "K": K,
            "query": qrec["query"],
            "answer": qrec["answer"],
            "question_type": qrec["question_type"],
            "num_lines": len(raw_lines),
            "num_chars": len(numbered),
            "haystack": numbered,
            "evidence_lines": sorted(set(ev_lines)),
            "evidence_facts": ev_facts,
            "needles_total": len(facts),
            "needles_in_topk": n_in_topk,
            "needles_evidence_lines": len(ev_facts),
            "topk_chunk_ids": top_ids,
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids-from", required=True,
                    help="JSONL whose 'qid' field defines which queries to build "
                         "(e.g. the existing Experiment A pilot output).")
    ap.add_argument("--lc-queries", default="processed/queries_d20.jsonl",
                    help="Long-context query file (used to recover gold facts).")
    ap.add_argument("--retrieval",
                    default="retrieval_results/text-embedding-3-large_tokens256")
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[4, 10, 25, 50, 100, 200])
    ap.add_argument("--out-dir", default="processed/rag")
    args = ap.parse_args()

    pilot_records = load_jsonl(args.qids_from)
    target_qids = sorted({r["qid"] for r in pilot_records})
    print(f"Building RAG haystacks for {len(target_qids)} qids "
          f"x K={args.ks}")

    lc_recs = load_jsonl(args.lc_queries)
    qid_to_lc = {r["qid"]: r for r in lc_recs}

    chunks = load_chunks(args.retrieval)
    chunk_emb, query_emb, query_ids = load_embeddings(args.retrieval)
    qid_to_row = {qid: i for i, qid in enumerate(query_ids)}

    fact_ranks = load_jsonl(os.path.join(args.retrieval, "fact_ranks.jsonl"))
    fact_to_chunks: dict[tuple[int, int], list[int]] = {}
    qid_facts: dict[int, list[str]] = {}
    for fr in fact_ranks:
        fact_to_chunks[(fr["qid"], fr["fact_idx"])] = fr["gold_chunk_ids"]
        qid_facts.setdefault(fr["qid"], [])
        while len(qid_facts[fr["qid"]]) <= fr["fact_idx"]:
            qid_facts[fr["qid"]].append("")
        qid_facts[fr["qid"]][fr["fact_idx"]] = fr["fact"]

    os.makedirs(args.out_dir, exist_ok=True)
    out_paths: dict[int, str] = {}
    out_files: dict[int, object] = {}
    for K in args.ks:
        p = os.path.join(args.out_dir, f"queries_rag_k{K}.jsonl")
        out_paths[K] = p
        out_files[K] = open(p, "w")

    n_done = 0
    cov_stats = {K: {"covered": 0, "facts_total": 0, "facts_in_topk": 0}
                 for K in args.ks}
    for qid in target_qids:
        if qid not in qid_to_lc:
            continue
        recs = build_per_query(
            qid_to_lc[qid], qid_to_row, query_emb, chunk_emb, chunks,
            args.ks, fact_to_chunks, qid_facts,
        )
        for r in recs:
            out_files[r["K"]].write(json.dumps(r) + "\n")
            cov_stats[r["K"]]["facts_total"] += r["needles_total"]
            cov_stats[r["K"]]["facts_in_topk"] += r["needles_in_topk"]
            if r["needles_in_topk"] == r["needles_total"]:
                cov_stats[r["K"]]["covered"] += 1
        n_done += 1

    for f in out_files.values():
        f.close()

    print(f"Wrote {n_done} queries x {len(args.ks)} K values")
    print(f"\nCoverage (across {n_done} queries):")
    print(f"  {'K':>5}{'all-gold queries':>20}{'fact coverage':>18}")
    for K in args.ks:
        s = cov_stats[K]
        agq = s["covered"] / n_done if n_done else 0
        fc = s["facts_in_topk"] / s["facts_total"] if s["facts_total"] else 0
        print(f"  {K:>5}{agq*100:>19.1f}%{fc*100:>17.1f}%")
    print("\nOutputs:")
    for K, p in out_paths.items():
        print(f"  K={K:>4}: {p}")


if __name__ == "__main__":
    main()
