"""
Build per-query multi-paper haystacks for the QASPER benchmark (QASPER-MP).

For each query we construct a haystack consisting of:

  1. the gold paper that the question is anchored to (so the answer is
     answerable);
  2. ``--num-distractors`` additional papers sampled without replacement
     from the same split (so the retriever has to find the gold paper
     among them).

Papers are concatenated paragraph by paragraph in random order, with a
``[PAPER <i> | <title>]`` header line before each paper and an optional
``[SECTION <name>]`` marker before each section. Every text paragraph
becomes one numbered line in the prompt-ready string, using the same
``{lineno}: {text}\\n`` format used elsewhere in the codebase.
``paragraph_texts`` records the raw text of every line, parallel to the
numbered context, so the official Evidence-F1 metric (which operates on
paragraph *texts*) can be reconstructed from predicted paragraph IDs.

Evidence handling follows the official ``text_evidence_only`` protocol:
``FLOAT SELECTED`` evidence (figures/tables) is excluded. Each remaining
gold evidence string is mapped to a paragraph in the gold paper's region
of the rendered haystack via exact text match first, then a substring
fallback. Queries whose every annotator has unmappable evidence are
dropped (~1.4% on dev).

Output (one JSON record per usable query):

    {
      "qid": str,
      "gold_paper_id": str,
      "gold_paper_index": int,
      "gold_paragraph_range": [int, int],
      "query": str,
      "question_type": "extractive" | "abstractive" | "boolean" | "none",
      "n_distractors": int,
      "num_paragraphs": int,
      "num_chars": int,
      "haystack": str,             # prompt-ready numbered context
      "paragraph_texts": [str],    # parallel, indexed by line number
      "annotations": [
        {"answer": str,
         "evidence_paragraphs": [int],
         "evidence_texts": [str],
         "answer_type": str}
      ]
    }

Usage:

    python build_haystack.py --split-path dataset/qasper-dev-v0.3.json \\
        --num-distractors 10 --out processed/queries_dev_d10.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter


def render_paper(idx: int, paper_id: str, paper: dict,
                 raw_lines: list[str]) -> tuple[int, int]:
    """Append ``paper`` to ``raw_lines`` paragraph-by-paragraph.

    Paper boundaries use XML-style ``<paper>``/``</paper>`` tags; the title
    and abstract are emitted as ``[TITLE] ...`` and ``[ABSTRACT] ...`` lines
    so the model can identify each paper without numeric paper indices.
    Returns the (start, end) line-index range covered by this paper.
    """
    start = len(raw_lines)
    title = (paper.get("title") or paper_id).strip()
    raw_lines.append("<paper>")
    raw_lines.append(f"<title>{title}</title>")
    abstract = (paper.get("abstract") or "").strip()
    if abstract:
        raw_lines.append(f"<abstract>{abstract}</abstract>")
    for section in paper.get("full_text", []):
        sec = (section.get("section_name") or "").strip()
        if sec:
            raw_lines.append(f"<section>{sec}</section>")
        for par in section.get("paragraphs", []):
            par = par.strip()
            if par:
                raw_lines.append(par)
    raw_lines.append("</paper>")
    return start, len(raw_lines)


def find_paragraph(ev: str, raw_lines: list[str],
                   start: int, end: int) -> int | None:
    """Locate evidence text within ``raw_lines[start:end]``.

    Tries (in order): exact match, ``ev`` is substring of a paragraph,
    a paragraph is substring of ``ev``, and finally a 80-char prefix match.
    """
    ev_norm = ev.strip()
    if not ev_norm:
        return None
    for i in range(start, end):
        if raw_lines[i] == ev_norm:
            return i
    for i in range(start, end):
        if ev_norm in raw_lines[i]:
            return i
    for i in range(start, end):
        if raw_lines[i] and raw_lines[i] in ev_norm:
            return i
    head = ev_norm[:80]
    for i in range(start, end):
        if raw_lines[i].startswith(head):
            return i
    return None


def classify_answer(ans: dict) -> tuple[str, str]:
    """Return (answer_string, type) following the official QASPER evaluator."""
    if ans.get("unanswerable"):
        return "Unanswerable", "none"
    if ans.get("extractive_spans"):
        return ", ".join(ans["extractive_spans"]), "extractive"
    if ans.get("free_form_answer"):
        return ans["free_form_answer"], "abstractive"
    if ans.get("yes_no") is True:
        return "Yes", "boolean"
    if ans.get("yes_no") is False:
        return "No", "boolean"
    return "", "unknown"


def build(args: argparse.Namespace) -> None:
    src = json.load(open(args.split_path))
    paper_ids = list(src.keys())
    rng_master = random.Random(args.seed)

    n_written = 0
    n_dropped_no_evidence = 0
    n_dropped_short = 0
    n_unanswerable_only = 0
    types_kept: Counter = Counter()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fout:
        for gold_pid, gold_paper in src.items():
            for qa in gold_paper["qas"]:
                qid = qa["question_id"]

                rng = random.Random(rng_master.randrange(2**31))
                pool = [p for p in paper_ids if p != gold_pid]
                rng.shuffle(pool)
                distractor_ids = pool[: args.num_distractors]

                paper_order = distractor_ids + [gold_pid]
                rng.shuffle(paper_order)
                gold_idx = paper_order.index(gold_pid)

                raw_lines: list[str] = []
                gold_range = (0, 0)
                for i, pid in enumerate(paper_order):
                    paper_obj = gold_paper if pid == gold_pid else src[pid]
                    start, end = render_paper(i, pid, paper_obj, raw_lines)
                    if pid == gold_pid:
                        gold_range = (start, end)

                annotations: list[dict] = []
                any_evidence_mapped = False
                for ann in qa["answers"]:
                    ans = ann["answer"]
                    answer_str, answer_type = classify_answer(ans)
                    ev_strings = [e for e in (ans.get("evidence") or [])
                                  if "FLOAT SELECTED" not in e]
                    ev_paragraphs: list[int] = []
                    ev_texts_used: list[str] = []
                    for ev in ev_strings:
                        idx = find_paragraph(ev, raw_lines, *gold_range)
                        if idx is not None:
                            ev_paragraphs.append(idx)
                            ev_texts_used.append(raw_lines[idx])
                    # If the annotator gave evidence but we mapped none, skip.
                    if ev_strings and not ev_paragraphs:
                        continue
                    if ev_paragraphs:
                        any_evidence_mapped = True
                    annotations.append({
                        "answer": answer_str,
                        "evidence_paragraphs": sorted(set(ev_paragraphs)),
                        "evidence_texts": ev_texts_used,
                        "answer_type": answer_type,
                    })

                if not annotations:
                    n_dropped_short += 1
                    continue

                non_none = [a for a in annotations if a["answer_type"] != "none"]
                if non_none and not any_evidence_mapped:
                    n_dropped_no_evidence += 1
                    continue
                if not non_none:
                    n_unanswerable_only += 1
                    if args.skip_unanswerable:
                        continue

                numbered = "\n".join(f"{i}: {ln}" for i, ln in enumerate(raw_lines))
                qtype = Counter(a["answer_type"] for a in annotations).most_common(1)[0][0]
                record = {
                    "qid": qid,
                    "gold_paper_id": gold_pid,
                    "gold_paper_index": gold_idx,
                    "gold_paragraph_range": list(gold_range),
                    "query": qa["question"],
                    "question_type": qtype,
                    "n_distractors": args.num_distractors,
                    "num_paragraphs": len(raw_lines),
                    "num_chars": len(numbered),
                    "haystack": numbered,
                    "paragraph_texts": raw_lines,
                    "annotations": annotations,
                }
                fout.write(json.dumps(record) + "\n")
                n_written += 1
                types_kept[qtype] += 1
                if args.limit and n_written >= args.limit:
                    break
            if args.limit and n_written >= args.limit:
                break

    print(f"Wrote {args.out}")
    print(f"  {n_written} usable queries kept")
    print(f"  dropped (no mappable evidence): {n_dropped_no_evidence}")
    print(f"  dropped (no usable annotation): {n_dropped_short}")
    print(f"  unanswerable-only queries:      {n_unanswerable_only}"
          f"{' (skipped)' if args.skip_unanswerable else ''}")
    print(f"  question-type breakdown: {dict(types_kept)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-path",
                    default="dataset/qasper-dev-v0.3.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-distractors", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many usable queries (smoke test).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-unanswerable", action="store_true")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
