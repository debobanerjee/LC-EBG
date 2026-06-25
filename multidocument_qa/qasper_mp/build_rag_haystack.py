"""
Build per-query RAG haystacks for QASPER-MP in four variants:

  flat_pool11    top-K paragraphs from the SAME 11-paper LC pool (gold +
                 10 distractors), flat list with ``[CHUNK r | title]``
                 markers. Apples-to-apples corpus match against LC.

  flat_full      top-K paragraphs retrieved from the FULL split corpus
                 (281 papers in dev), flat list. Realistic 'paper search'
                 retrieval problem.

  struct_pool11  top-K paragraphs from the 11-paper pool, grouped by
                 source paper. For every paper that contributes at least
                 one retrieved paragraph we emit a ``<paper>`` block with
                 the title, the abstract, and the retrieved paragraphs of
                 that paper -- preserving the document-level context that
                 flat-RAG strips. Citations use sequential haystack line
                 numbers.

  struct_full    same structured layout but retrieving from the full
                 corpus.

For every record we also report retrieval-coverage diagnostics:
  needles_total          : number of distinct gold evidence paragraphs.
  needles_in_topk        : how many of those gold paragraphs ranked at or
                           below K against the variant's pool.
  needles_evidence_lines : how many gold paragraphs were locatable in the
                           rendered haystack (after re-mapping).

Usage:

    python build_rag_haystack.py \\
      --qids-from experiment_outputs/qasper_mp_gpt-5_<ts>.jsonl \\
      --lc-queries processed/queries_dev_d10.jsonl \\
      --retrieval retrieval_results/dev_large \\
      --ks 1 3 10 25 50 100 \\
      --out-dir processed/rag
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_haystack import find_paragraph, classify_answer  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path)]


def render_flat(top_pars: list[dict]) -> tuple[str, list[str]]:
    """Render a flat list of retrieved paragraphs with [CHUNK r] markers."""
    raw_lines: list[str] = []
    for rank, p in enumerate(top_pars, start=1):
        raw_lines.append(f"[CHUNK {rank} | {p['title']}]")
        raw_lines.append(p["text"])
    numbered = "\n".join(f"{i}: {ln}" for i, ln in enumerate(raw_lines))
    return numbered, raw_lines


def render_structured(top_pars: list[dict],
                      paper_lookup: dict[str, dict]) -> tuple[str, list[str]]:
    """Render top-K paragraphs grouped by source paper.

    For each paper that contributes a paragraph we emit a ``<paper>`` block
    with title + abstract + that paper's retrieved paragraphs (sorted by
    their original position within the paper, so adjacency is visible).
    """
    by_paper: dict[str, list[dict]] = defaultdict(list)
    paper_order: list[str] = []
    for p in top_pars:
        if p["paper_id"] not in by_paper:
            paper_order.append(p["paper_id"])
        by_paper[p["paper_id"]].append(p)

    raw_lines: list[str] = []
    for pid in paper_order:
        paper = paper_lookup[pid]
        raw_lines.append("<paper>")
        raw_lines.append(f"<title>{paper.get('title','').strip()}</title>")
        if paper.get("abstract"):
            raw_lines.append(f"<abstract>{paper['abstract'].strip()}</abstract>")
        pars_sorted = sorted(by_paper[pid], key=lambda x: x["par_idx_in_paper"])
        prev_idx: int | None = None
        for p in pars_sorted:
            cur_idx = p["par_idx_in_paper"]
            # Insert "..." for any gap between the abstract / a previous
            # retrieved paragraph and the current one.
            if prev_idx is None and cur_idx > 0:
                raw_lines.append("...")
            elif prev_idx is not None and cur_idx > prev_idx + 1:
                raw_lines.append("...")
            raw_lines.append(p["text"])
            prev_idx = cur_idx
        # Trailing ellipsis if the paper continues past the last retrieved par.
        n_total = sum(
            1
            for sec in paper.get("full_text", [])
            for par in sec.get("paragraphs", []) if par.strip()
        )
        if prev_idx is not None and n_total and prev_idx < n_total - 1:
            raw_lines.append("...")
        raw_lines.append("</paper>")
    numbered = "\n".join(f"{i}: {ln}" for i, ln in enumerate(raw_lines))
    return numbered, raw_lines


def build_records(
    qrec: dict,
    qid: str,
    qrow: int,
    par_emb: np.ndarray,
    query_emb: np.ndarray,
    par_meta: list[dict],
    pool_paper_ids: set[str],
    full_paper_ids: list[str],
    paper_lookup: dict[str, dict],
    ks: list[int],
) -> list[dict]:
    """Build one record per (variant, K) for this query."""
    sims = par_emb @ query_emb[qrow]

    pool_mask = np.array([par_meta[i]["paper_id"] in pool_paper_ids
                          for i in range(len(par_meta))], dtype=bool)

    pool_order = np.argsort(-sims[pool_mask])
    pool_idx_global = np.where(pool_mask)[0][pool_order]
    full_order = np.argsort(-sims)

    gold_evidence_texts: set[str] = set()
    for ann in qrec["annotations"]:
        for t in ann["evidence_texts"]:
            gold_evidence_texts.add(t)

    records: list[dict] = []
    for variant_name, ordered_idx in (("pool11", pool_idx_global),
                                       ("full",   full_order)):
        for K in ks:
            top_global = ordered_idx[:K].tolist()
            top_pars = [par_meta[i] for i in top_global]
            for p in top_pars:
                p["title"] = paper_lookup[p["paper_id"]].get("title", p["paper_id"])

            top_texts = {p["text"] for p in top_pars}
            n_in_topk = sum(1 for t in gold_evidence_texts if t in top_texts)

            for layout in ("flat", "struct"):
                if layout == "flat":
                    numbered, raw_lines = render_flat(top_pars)
                else:
                    numbered, raw_lines = render_structured(top_pars, paper_lookup)

                # re-map gold evidence paragraphs into this haystack
                new_annotations: list[dict] = []
                for ann in qrec["annotations"]:
                    ev_paragraphs: list[int] = []
                    ev_texts_used: list[str] = []
                    for ev_text in ann["evidence_texts"]:
                        idx = find_paragraph(ev_text, raw_lines,
                                             0, len(raw_lines))
                        if idx is not None:
                            ev_paragraphs.append(idx)
                            ev_texts_used.append(raw_lines[idx])
                    new_annotations.append({
                        "answer": ann["answer"],
                        "evidence_paragraphs": sorted(set(ev_paragraphs)),
                        "evidence_texts": ev_texts_used,
                        "answer_type": ann["answer_type"],
                    })

                records.append({
                    "qid": qid,
                    "variant": f"{layout}_{variant_name}",
                    "K": K,
                    "gold_paper_id": qrec["gold_paper_id"],
                    "query": qrec["query"],
                    "question_type": qrec["question_type"],
                    "n_distractors": qrec["n_distractors"],
                    "num_paragraphs": len(raw_lines),
                    "num_chars": len(numbered),
                    "haystack": numbered,
                    "paragraph_texts": raw_lines,
                    "annotations": new_annotations,
                    "needles_total": len(gold_evidence_texts),
                    "needles_in_topk": n_in_topk,
                    "needles_evidence_lines": sum(
                        1 for a in new_annotations for _ in a["evidence_paragraphs"]
                    ),
                })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids-from", required=True,
                    help="JSONL whose 'qid' set defines which queries to build.")
    ap.add_argument("--lc-queries",
                    default="processed/queries_dev_d10.jsonl",
                    help="LC mini-haystacks (for distractor pool reproduction).")
    ap.add_argument("--split-path",
                    default="dataset/qasper-dev-v0.3.json")
    ap.add_argument("--retrieval", default="retrieval_results/dev_large")
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[1, 3, 10, 25, 50, 100])
    ap.add_argument("--out-dir", default="processed/rag")
    args = ap.parse_args()

    pilot = load_jsonl(args.qids_from)
    qids_wanted = {r["qid"] for r in pilot}
    print(f"Building RAG haystacks for {len(qids_wanted)} pilot qids "
          f"x K={args.ks} x 4 variants")

    lc_recs = load_jsonl(args.lc_queries)
    qid_to_lc = {r["qid"]: r for r in lc_recs}

    src = json.load(open(args.split_path))
    paper_lookup = {pid: paper for pid, paper in src.items()}

    par_meta = load_jsonl(os.path.join(args.retrieval, "paragraphs.jsonl"))
    par_emb = np.load(os.path.join(args.retrieval, "paragraph_emb.npy"))
    q_meta = load_jsonl(os.path.join(args.retrieval, "queries.jsonl"))
    q_emb = np.load(os.path.join(args.retrieval, "query_emb.npy"))
    qid_to_row = {q["qid"]: i for i, q in enumerate(q_meta)}

    os.makedirs(args.out_dir, exist_ok=True)
    out_files: dict[tuple[str, int], object] = {}
    for variant in ("flat_pool11", "flat_full", "struct_pool11", "struct_full"):
        for K in args.ks:
            p = os.path.join(args.out_dir, f"queries_{variant}_k{K}.jsonl")
            out_files[(variant, K)] = open(p, "w")

    full_paper_ids = list(src.keys())
    cov_stats: dict[tuple[str, int], dict] = {
        (v, K): {"covered": 0, "facts_total": 0, "facts_in_topk": 0}
        for v in ("pool11", "full") for K in args.ks
    }

    n_done = 0
    for qid in sorted(qids_wanted):
        if qid not in qid_to_lc or qid not in qid_to_row:
            continue
        qrec = qid_to_lc[qid]
        # Reproduce the LC distractor set: same paper IDs that appear in qrec.
        # LC headers are now ``<title>...</title>`` lines (one per paper).
        title_to_pid = {paper_lookup[p]["title"].strip(): p for p in paper_lookup}
        pool_paper_ids = set()
        for line in qrec["paragraph_texts"]:
            if line.startswith("<title>") and line.endswith("</title>"):
                title = line[len("<title>"):-len("</title>")].strip()
                pid = title_to_pid.get(title)
                if pid:
                    pool_paper_ids.add(pid)
        # Always include the gold paper (in case of title clash).
        pool_paper_ids.add(qrec["gold_paper_id"])

        records = build_records(
            qrec, qid, qid_to_row[qid], par_emb, q_emb, par_meta,
            pool_paper_ids, full_paper_ids, paper_lookup, args.ks,
        )
        seen_cov: set[tuple[str, int]] = set()
        for r in records:
            v_key = "pool11" if r["variant"].endswith("pool11") else "full"
            key = (v_key, r["K"])
            if key not in seen_cov:
                cov_stats[key]["facts_total"] += r["needles_total"]
                cov_stats[key]["facts_in_topk"] += r["needles_in_topk"]
                if r["needles_in_topk"] == r["needles_total"]:
                    cov_stats[key]["covered"] += 1
                seen_cov.add(key)
            out_files[(r["variant"], r["K"])].write(json.dumps(r) + "\n")
        n_done += 1

    for f in out_files.values():
        f.close()

    print(f"\nBuilt {n_done} queries x {len(args.ks)} Ks x 4 variants.")
    print("\nNeedle coverage:")
    print(f"  {'pool':<6}{'K':>5}{'all-needles q%':>20}{'fact recall':>14}")
    for pool in ("pool11", "full"):
        for K in args.ks:
            s = cov_stats[(pool, K)]
            agq = s["covered"] / n_done if n_done else 0.0
            fc = s["facts_in_topk"] / s["facts_total"] if s["facts_total"] else 0.0
            print(f"  {pool:<6}{K:>5}{agq*100:>19.1f}%{fc*100:>13.1f}%")


if __name__ == "__main__":
    main()
