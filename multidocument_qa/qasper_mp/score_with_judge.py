"""LLM-judge scoring for QASPER-MP outputs.

For each record we ask GPT-5 to decide whether the predicted answer is
semantically equivalent to ANY of the annotator gold answers. The judge
returns a score in {0, 0.5, 1}:

    1.0  : prediction means the same thing as a gold answer (or any of them).
    0.5  : partial credit -- conveys some but not all of the gold information,
           or is a defensible but less specific/complete answer.
    0.0  : wrong, irrelevant, or a refusal when the gold is answerable
           (or vice versa).

We cache judgments by content hash so repeated runs are free; cached
results live in ``processed/judge_cache.json``.

Usage:
    python score_with_judge.py --in-dir experiment_outputs_final \\
                               --models gpt-5
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
load_dotenv(ROOT / ".env")

from openai import OpenAI

CACHE_PATH = HERE / "processed" / "judge_cache.json"
JUDGE_MODEL = "gpt-5"

SYSTEM = (
    "You are an exact, dispassionate grader. You will be shown a question, "
    "one or more reference answers written by human annotators, and a "
    "prediction from a model. Decide whether the prediction is semantically "
    "equivalent to AT LEAST ONE reference answer.\n\n"
    "Scoring (return exactly one number):\n"
    "  1.0  : prediction conveys the same content as a reference (paraphrase, "
    "more-specific term such as 'Inception V3' for 'CNN', or a verbatim "
    "match -- all count). \"Unanswerable\" vs \"Unanswerable\" is 1.0.\n"
    "  0.5  : partial credit -- the prediction is on the right track but is "
    "missing key information, or gives one of several required parts.\n"
    "  0.0  : prediction is wrong, irrelevant, contradicts the reference, "
    "or refuses (\"Unanswerable\") when a reference contains a real answer "
    "(or the reverse).\n\n"
    "Be lenient about formatting and elaboration. Be strict about content. "
    "Output ONLY the number: 0, 0.5, or 1. No explanation."
)


def hash_key(question: str, references: list[str], prediction: str) -> str:
    h = hashlib.sha256()
    h.update(question.encode())
    for r in references:
        h.update(b"|R|")
        h.update(r.encode())
    h.update(b"|P|")
    h.update(prediction.encode())
    return h.hexdigest()[:24]


def judge_one(client: OpenAI, question: str, references: list[str],
              prediction: str) -> float:
    if not (prediction or "").strip():
        return 0.0
    refs_block = "\n".join(f"  Reference {i+1}: {r}" for i, r in enumerate(references))
    user = (
        f"Question: {question}\n"
        f"{refs_block}\n"
        f"Prediction: {prediction}\n\n"
        f"Score (0, 0.5, or 1):"
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    # Pull the first number-like token
    for tok in text.replace(",", " ").split():
        tok = tok.strip(".:;").rstrip(")")
        try:
            v = float(tok)
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            continue
    return 0.0


def gold_answers(rec: dict) -> list[str]:
    out: list[str] = []
    for ann in rec.get("annotations", []):
        a = ann.get("answer") if isinstance(ann, dict) else ann
        if isinstance(a, str) and a.strip():
            out.append(a)
    return out


def process_file(client: OpenAI, path: str, cache: dict, parallel: int) -> dict:
    """Score every successful record in ``path``; return summary stats."""
    recs = [json.loads(l) for l in open(path)]
    jobs = []
    for rec in recs:
        if rec.get("error"):
            rec["judge_score"] = None
            continue
        q = rec["query"]
        refs = gold_answers(rec)
        pred = (rec.get("parsed") or {}).get("answer", "") or rec.get("pred_answer", "")
        key = hash_key(q, refs, pred)
        if key in cache:
            rec["judge_score"] = cache[key]
            continue
        jobs.append((rec, q, refs, pred, key))
    if jobs:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = {ex.submit(judge_one, client, q, refs, pred): (rec, key)
                    for rec, q, refs, pred, key in jobs}
            for fut in as_completed(futs):
                rec, key = futs[fut]
                try:
                    score = fut.result()
                except Exception as exc:
                    print(f"  judge error: {exc}", file=sys.stderr)
                    score = 0.0
                rec["judge_score"] = score
                cache[key] = score

    out_path = path.replace(".jsonl", ".judged.jsonl")
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    succ = [r for r in recs if r.get("error") is None]
    if not succ:
        return {"n": 0, "judge": 0.0, "ans_f1": 0.0}
    return {
        "n": len(succ),
        "judge": sum(r.get("judge_score") or 0 for r in succ) / len(succ),
        "ans_f1": sum(r.get("answer_f1") or 0 for r in succ) / len(succ),
        "out_path": out_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="experiment_outputs_final")
    ap.add_argument("--pattern", default="*.jsonl")
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    cache = {}
    if CACHE_PATH.exists() and not args.no_cache:
        cache = json.load(open(CACHE_PATH))
    print(f"Cache: {len(cache)} entries")

    paths = sorted(glob.glob(os.path.join(HERE, args.in_dir, "**",
                                          args.pattern), recursive=True))
    paths = [p for p in paths if not p.endswith(".judged.jsonl")]
    print(f"Scoring {len(paths)} files")
    for p in paths:
        s = process_file(client, p, cache, args.parallel)
        rel = os.path.relpath(p, HERE)
        print(f"  {rel:<80} n={s['n']:>3}  Ans-F1={s['ans_f1']:.3f}  "
              f"Judge={s['judge']:.3f}")
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        json.dump(cache, open(CACHE_PATH, "w"))


if __name__ == "__main__":
    main()
