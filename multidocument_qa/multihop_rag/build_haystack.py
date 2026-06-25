"""
Build per-query line-indexed haystacks from the MultiHop-RAG corpus.

For each non-null query we construct a mini-haystack consisting of:

  1. all gold-evidence documents (the ones whose ``title`` appears in the
     query's ``evidence_list``), so the answer is always answerable;
  2. ``--num-distractors`` additional documents sampled (without replacement)
     from the rest of the corpus.

The documents are ordered randomly and rendered in the ``{lineno}: {text}\\n``
format used by the synthetic experiments. The mapping from each gold
``fact`` string to its line number(s) is recomputed inside the assembled
mini-haystack so the evidence labels are aligned with the prompt the model
will see.

Output (one JSON record per usable query):

    {
      "qid": int,
      "query": str,
      "answer": str,
      "question_type": str,
      "num_lines": int,           # haystack size
      "num_chars": int,
      "haystack": str,            # full numbered context, ready to drop into a prompt
      "evidence_lines": [int],    # canonical line per gold-fact
      "evidence_facts": [str],
    }

The corpus has ~600 docs / 6.3M chars, so a 1M-character mini-haystack covers
roughly 90 distractors plus the 2--4 gold docs — comfortably within
long-context budgets for GPT-5 / Gemini-3-Flash, and selectable down to a few
docs for a quick smoke test.

Example:

    python build_haystack.py --num-distractors 20 --out processed_d20.jsonl
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict


SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z0-9\"\'])")


def split_into_lines(body: str) -> list[str]:
    body = body.replace("\r", " ")
    body = re.sub(r"\n{2,}", "\n", body)
    out = []
    for paragraph in body.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for s in SENT_SPLIT_RE.split(paragraph):
            s = s.strip()
            if s:
                out.append(s)
    return out


def find_fact_lines(fact: str, lines: list[str]) -> list[int]:
    fact_norm = " ".join(fact.split())
    if not fact_norm:
        return []
    head = fact_norm[: min(60, len(fact_norm))]
    for i in range(len(lines)):
        if head in lines[i] or head[:40] in lines[i]:
            acc = lines[i]
            hits = [i]
            j = i
            while fact_norm not in " ".join(acc.split()) and j + 1 < len(lines):
                j += 1
                acc = acc + " " + lines[j]
                hits.append(j)
                if len(hits) > 10:
                    break
            return hits
    fact_low = fact_norm.lower()
    for i in range(len(lines)):
        if lines[i].lower()[:50] in fact_low:
            return [i]
    return []


def render_haystack(docs: list[dict]) -> tuple[str, list[str]]:
    """Concatenate ``docs`` into a single sentence-per-line haystack.

    Returns (numbered_text, raw_lines). ``raw_lines[i]`` is the unnumbered
    content of line ``i``; ``numbered_text`` is the prompt-ready string
    where every line is prefixed with ``"{i}: "``.
    """
    raw_lines: list[str] = []
    for doc in docs:
        header = f"[DOC] {doc['title']} | {doc.get('source','')} | {doc.get('published_at','')[:10]}"
        raw_lines.append(header)
        raw_lines.extend(split_into_lines(doc["body"]))
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(raw_lines))
    return numbered, raw_lines


def build(args: argparse.Namespace) -> None:
    corpus = json.load(open(args.corpus))
    queries = json.load(open(args.queries))

    by_title: dict[str, dict] = {}
    for d in corpus:
        # If multiple docs share a title, keep the longest body.
        if d["title"] not in by_title or len(d["body"]) > len(by_title[d["title"]]["body"]):
            by_title[d["title"]] = d
    all_titles = list(by_title.keys())

    n_written = 0
    n_evidence_missing = 0
    n_no_gold_doc = 0
    n_skipped_null = 0

    rng_master = random.Random(args.seed)

    with open(args.out, "w") as fout:
        for qid, q in enumerate(queries):
            if q.get("question_type") == "null_query":
                n_skipped_null += 1
                continue

            gold_titles = []
            for ev in q.get("evidence_list", []):
                t = ev["title"]
                if t in by_title and t not in gold_titles:
                    gold_titles.append(t)
            if not gold_titles:
                n_no_gold_doc += 1
                continue

            # Sample distractor titles deterministically per qid.
            rng = random.Random(rng_master.randrange(2**31))
            pool = [t for t in all_titles if t not in gold_titles]
            rng.shuffle(pool)
            distractors = pool[: args.num_distractors]
            chosen_titles = list(gold_titles) + distractors
            rng.shuffle(chosen_titles)
            docs = [by_title[t] for t in chosen_titles]

            numbered, raw_lines = render_haystack(docs)
            # Find each fact's line in the assembled haystack.
            ev_lines: list[int] = []
            ev_facts: list[str] = []
            ok = True
            for ev in q.get("evidence_list", []):
                hits = find_fact_lines(ev["fact"], raw_lines)
                if not hits:
                    ok = False
                    break
                ev_lines.append(hits[0])
                ev_facts.append(ev["fact"])
            if not ok:
                n_evidence_missing += 1
                continue

            record = {
                "qid": qid,
                "query": q["query"],
                "answer": q["answer"],
                "question_type": q.get("question_type"),
                "num_lines": len(raw_lines),
                "num_chars": len(numbered),
                "haystack": numbered,
                "evidence_lines": sorted(set(ev_lines)),
                "evidence_facts": ev_facts,
            }
            fout.write(json.dumps(record) + "\n")
            n_written += 1
            if args.limit and n_written >= args.limit:
                break

    print(f"Wrote {args.out}")
    print(f"  {n_written} usable queries | "
          f"{n_evidence_missing} dropped (evidence unmappable) | "
          f"{n_no_gold_doc} dropped (no gold doc found) | "
          f"{n_skipped_null} null_query skipped")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="dataset/corpus.json")
    ap.add_argument("--queries", default="dataset/MultiHopRAG.json")
    ap.add_argument("--out", default="processed/queries_d20.jsonl",
                    help="Output JSONL path.")
    ap.add_argument("--num-distractors", type=int, default=20,
                    help="Number of distractor documents to add per query "
                         "(in addition to the 2-4 gold-evidence docs).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many usable queries (smoke test).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    build(args)


if __name__ == "__main__":
    main()
