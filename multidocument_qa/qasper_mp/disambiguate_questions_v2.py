"""Rewrite QASPER pilot questions with REALISTIC, SHORT paper cues.

Two cue types, assigned 50/50 deterministically by qid hash:

  1. AUTHOR cue:  "In Smith et al., ..."  (>=3 surnames)
                  "In Smith and Doe, ..." (2 surnames)
                  "In Smith, ..."          (1 surname)
     -- done programmatically, no LLM needed.

  2. TITLE cue:  2-3 consecutive words copied VERBATIM from the title,
                 e.g. "In the BERTSUM paper, ..." or
                 "In the Winograd Schemas paper, ...".
                 The LLM only chooses the 2-3 word span; we then build
                 the question deterministically.

The substantive question text is otherwise preserved verbatim (with light
pronoun cleanup done by the LLM for the title-cue case).

Output: processed/disambiguated_questions_v2.json
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

from openai import OpenAI  # noqa: E402

DATASET = HERE / "dataset" / "qasper-dev-v0.3.json"
QUERIES = HERE / "processed" / "queries_dev_d10.jsonl"
AUTHORS = HERE / "processed" / "paper_authors.json"
LC_OUTPUT = HERE / "experiment_outputs" / "qasper_mp_gpt-5_20260518_163801.jsonl"
OUT_PATH = HERE / "processed" / "disambiguated_questions_v2.json"

REWRITE_MODEL = "gpt-5"


def get_pilot_qids() -> list[str]:
    return [json.loads(l)["qid"] for l in open(LC_OUTPUT)]


def author_cue(surnames: list[str]) -> str:
    """Return 'Smith', 'Smith and Doe', or 'Smith et al.'."""
    s = [x for x in surnames if x]
    if not s:
        return ""
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        return f"{s[0]} and {s[1]}"
    return f"{s[0]} et al."


SYSTEM_TITLE_CUE = (
    "You pick a SHORT, distinctive cue from a paper title that a researcher "
    "would casually use to refer to that paper. STRICT REQUIREMENTS:\n"
    "  - The cue MUST be a verbatim consecutive substring of the title.\n"
    "  - The cue MUST be 2 or 3 words long (no more, no fewer).\n"
    "  - Pick the most distinctive / content-bearing 2-3 word span; avoid "
    "generic words like 'A Study', 'Neural Network', 'Deep Learning' unless "
    "they are the only specific phrase available.\n"
    "  - Preserve original casing.\n"
    "Output ONLY the chosen 2-3 word phrase, nothing else."
)


SYSTEM_REWRITE = (
    "You rewrite a research-paper question so it is well-posed in a multi-"
    "paper QA setting. You are given a short paper REFERENCE (e.g., 'BERTSUM' "
    "or 'Smith et al.') that you must insert. Prepend exactly:\n"
    "  'In <REFERENCE>, '   if the reference is an author cue (contains "
    "a capitalized surname like 'Smith', 'Smith and Doe', or 'Smith et al.'),\n"
    "  'In the <REFERENCE> paper, '   if the reference is a topic phrase from "
    "the title.\n"
    "Then write the rest of the question. Resolve indexicals ('they', 'the "
    "paper', 'the model', 'the dataset', 'this work') so the question makes "
    "sense on its own. Keep the substantive meaning IDENTICAL: do not add or "
    "remove specifics. Output ONLY the rewritten question, on one line."
)


def pick_title_cue(client: OpenAI, title: str) -> str:
    resp = client.chat.completions.create(
        model=REWRITE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_TITLE_CUE},
            {"role": "user", "content": f"Title: {title}\n2-3 word cue:"},
        ],
    )
    cue = resp.choices[0].message.content.strip().strip('"').strip("'")
    # Validate: must be substring of title (case-insensitive) and 2-3 words.
    words = cue.split()
    if not (2 <= len(words) <= 3) or cue.lower() not in title.lower():
        # Fallback: first 2 content words of title.
        toks = re.findall(r"[A-Za-z0-9-]+", title)
        stop = {"a", "an", "the", "of", "for", "with", "in", "on", "to", "and",
                "or", "using", "via"}
        content = [t for t in toks if t.lower() not in stop]
        cue = " ".join(content[:2]) if content else " ".join(toks[:2])
    return cue


def rewrite_with_cue(client: OpenAI, question: str, title: str,
                     reference: str, is_author: bool) -> str:
    kind = "author cue" if is_author else "title topic cue"
    user = (
        f"Paper title (for your context only, do not paste it): {title}\n"
        f"REFERENCE to use ({kind}): {reference}\n\n"
        f"Original question: {question}\n"
        f"Rewritten question:"
    )
    resp = client.chat.completions.create(
        model=REWRITE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_REWRITE},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def main() -> None:
    if not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=OPENAI_API_KEY)

    dataset = json.load(open(DATASET))
    queries = {json.loads(l)["qid"]: json.loads(l) for l in open(QUERIES)}
    authors = json.load(open(AUTHORS))
    pilot_qids = get_pilot_qids()
    print(f"Pilot qids: {len(pilot_qids)}")

    # Deterministic 50/50 split by qid hash.
    def use_author(qid: str) -> bool:
        h = int(hashlib.md5(qid.encode()).hexdigest(), 16)
        return (h % 2) == 0

    n_auth = sum(use_author(q) for q in pilot_qids)
    print(f"Author cues: {n_auth} ; title cues: {len(pilot_qids) - n_auth}")

    title_cue_cache: dict[str, str] = {}
    # Resume from any partial output.
    out: dict[str, dict] = {}
    if OUT_PATH.exists():
        out = json.load(open(OUT_PATH))
        for v in out.values():
            if v.get("cue_kind") == "title":
                title_cue_cache[v["gold_paper_id"]] = v["reference"]
        print(f"Resuming with {len(out)} existing rewrites cached.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for i, qid in enumerate(pilot_qids, 1):
        if qid in out:
            continue
        q = queries[qid]
        pid = q["gold_paper_id"]
        title = dataset[pid]["title"]
        surnames = authors.get(pid, {}).get("surnames", []) or []
        is_author = use_author(qid) and bool(surnames)

        if is_author:
            ref = author_cue(surnames)
            cue_kind = "author"
        else:
            if pid not in title_cue_cache:
                title_cue_cache[pid] = pick_title_cue(client, title)
            ref = title_cue_cache[pid]
            cue_kind = "title"

        try:
            rewritten = rewrite_with_cue(client, q["query"], title, ref,
                                         is_author=is_author)
        except Exception as exc:
            print(f"  [{i:2d}] {qid[:8]} rewrite failed: {exc}", file=sys.stderr)
            rewritten = q["query"]

        out[qid] = {
            "qid": qid,
            "gold_paper_id": pid,
            "title": title,
            "surnames": surnames,
            "cue_kind": cue_kind,
            "reference": ref,
            "original": q["query"],
            "rewritten": rewritten,
        }
        print(f"  [{i:2d}/{len(pilot_qids)}] {cue_kind:6s} | {ref:30s} | "
              f"{rewritten[:100]}", flush=True)
        json.dump(out, open(OUT_PATH, "w"), indent=2)

    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
