"""
Run QASPER-MP EBG (or answer-only) over the per-query haystacks produced by
``build_haystack.py``. Scoring uses the official QASPER metrics:
SQuAD-style token-F1 for the answer and paragraph-set F1 (on paragraph
*texts*) for the evidence. As in the official evaluator, per-question
metrics are the max F1 across annotators.

Inputs:
  ``--queries`` JSONL with the schema from ``build_haystack.py``.
Outputs:
  JSONL with one record per query plus a summary printed at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

# The shared engine `experiments.py` lives one level up, in `multidocument_qa/`.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(REPO_ROOT, ".env"))

from experiments import (  # noqa: E402
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_ANSWER_ONLY,
    setup_clients,
    run_model,
    process_answer,
    parse_llm_json_output,
    is_context_window_exceeded_error,
    Model,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qasper_evaluator import token_f1_score, paragraph_f1_score  # noqa: E402


EXAMPLE_FORMAT = """
{
  "answer": "...",
  "paragraphs": [25, 412]
}
"""
EXAMPLE_FORMAT_ANSWER_ONLY = """
{
  "answer": "..."
}
"""

_ANSWER_FORMAT_RULES = (
    "Answer-format rules (STRICT, follow exactly):\n"
    "  - DEFAULT TO ANSWERING when the context contains the relevant "
    "information. Only output \"Unanswerable\" if the context genuinely "
    "does not contain the information the question asks for. Do not refuse "
    "out of caution.\n"
    "  - Yes/no questions: answer with exactly \"Yes\" or \"No\" -- NO "
    "elaboration, no explanation, no follow-up sentence.\n"
    "  - Extractive questions (asking for a name, number, dataset, metric, "
    "model, etc.): answer with the SHORTEST verbatim span from the context "
    "that contains the fact -- no preamble, no full sentence.\n"
    "  - Other questions: answer in at most 20 words. No preamble such as "
    "\"The paper says...\" or \"They...\". No quotes around the answer."
)

_CITATION_RULES = (
    "Citation rules (STRICT):\n"
    "  - Cite ONLY lines that are body paragraphs of a paper. "
    "DO NOT cite lines that are titles, abstracts, section headers, or "
    "structural tags. Concretely: do NOT cite any line that starts with "
    "'<title>', '<abstract>', '<section>', '<paper>', '</paper>', '[CHUNK', "
    "or that is just '...' (ellipsis). These lines are present only as "
    "context labels or gap markers; they are never valid evidence.\n"
    "  - Cite at least one body-paragraph line whenever you give a "
    "non-\"Unanswerable\" answer."
)

SYSTEM_PROMPT_QASPER = (
    "Your job is to answer the question entirely from the context, which is a "
    "set of scientific papers presented paragraph by paragraph with line "
    "numbers. Cite the paragraph line numbers that support your answer.\n\n"
    + _ANSWER_FORMAT_RULES + "\n\n" + _CITATION_RULES
)
SYSTEM_PROMPT_QASPER_ANSWER_ONLY = (
    "Your job is to answer the question entirely from the context, which is a "
    "set of scientific papers.\n\n"
    + _ANSWER_FORMAT_RULES
)


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path)]


def build_prompt(q: dict, answer_only: bool) -> str:
    if answer_only:
        return (
            f"<context>\n{q['haystack']}\n</context>\n\n"
            f"Answer the question based on information only from the context. "
            f"If the question is not answerable from the context, answer "
            f"\"Unanswerable\". Your response should be a short answer in "
            f"JSON format. For example:{EXAMPLE_FORMAT_ANSWER_ONLY}\n\n"
            f"Question: {q['query']}"
        )
    return (
        f"<context>\n{q['haystack']}\n</context>\n\n"
        f"Answer the question based on information only from the context. "
        f"If the question is not answerable from the context, answer "
        f"\"Unanswerable\". Your response should be a short answer and the "
        f"line numbers of the supporting paragraphs in JSON format. For "
        f"example:{EXAMPLE_FORMAT}\n\n"
        f"Question: {q['query']}"
    )


def normalize_paragraphs(ids, paragraph_texts: list[str]) -> list[str]:
    """Convert predicted paragraph IDs to paragraph texts.

    Drops out-of-range / non-integer ids silently so a malformed prediction
    cannot crash the run.
    """
    out: list[str] = []
    for x in ids or []:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(paragraph_texts):
            out.append(paragraph_texts[i])
    return out


def score_record(parsed: dict, q: dict, answer_only: bool) -> dict:
    """Compute per-annotator Answer-F1 and Evidence-F1 and return max-over-annotators."""
    pred_answer = str((parsed or {}).get("answer", "") or "")
    pred_par_ids = (parsed or {}).get("paragraphs") or []
    pred_par_texts = normalize_paragraphs(pred_par_ids, q["paragraph_texts"])

    ann_f1s: list[tuple[float, float, str]] = []
    for ann in q["annotations"]:
        a_f1 = token_f1_score(pred_answer, ann["answer"])
        if answer_only:
            e_f1 = float("nan")
        else:
            e_f1 = paragraph_f1_score(pred_par_texts, ann["evidence_texts"])
        ann_f1s.append((a_f1, e_f1, ann["answer_type"]))

    best_ans = max(ann_f1s, key=lambda x: x[0])
    if answer_only:
        best_ev = (0.0, float("nan"), "")
    else:
        best_ev = max(ann_f1s, key=lambda x: x[1])
    return {
        "pred_answer": pred_answer,
        "pred_paragraph_ids": [int(x) for x in (pred_par_ids or [])
                               if str(x).lstrip("-").isdigit()],
        "pred_paragraph_texts": pred_par_texts,
        "answer_f1": best_ans[0],
        "answer_type_for_best": best_ans[2],
        "evidence_f1": best_ev[1],
    }


def run_one(clients, model, q: dict, output_file: str | None,
            answer_only: bool, lock: Lock) -> dict:
    user_prompt = build_prompt(q, answer_only)
    sys_prompt = (SYSTEM_PROMPT_QASPER_ANSWER_ONLY if answer_only
                  else SYSTEM_PROMPT_QASPER)

    try:
        result = run_model(clients, model, user_prompt, system_prompt=sys_prompt)
    except Exception as e:
        err = str(e)
        if is_context_window_exceeded_error(e):
            err = f"context_window_exceeded: {err}"
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": model.api_name,
            "answer_only": answer_only,
            "qid": q["qid"],
            "question_type": q["question_type"],
            "gold_paper_index": q.get("gold_paper_index"),
            "n_distractors": q.get("n_distractors"),
            "num_chars": q.get("num_chars"),
            "error": err,
        }
        if output_file:
            with lock:
                with open(output_file, "a") as f:
                    f.write(json.dumps(record) + "\n")
        return record

    response, input_tokens = process_answer(result, model)
    if not (response or "").strip():
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": model.api_name,
            "answer_only": answer_only,
            "qid": q["qid"],
            "question_type": q["question_type"],
            "n_distractors": q.get("n_distractors"),
            "input_tokens": input_tokens,
            "error": "empty_response",
        }
        if output_file:
            with lock:
                with open(output_file, "a") as f:
                    f.write(json.dumps(record) + "\n")
        return record

    parsed = parse_llm_json_output(response) or {}
    scores = score_record(parsed, q, answer_only)
    record = {
        "timestamp": datetime.now().isoformat(),
        "model": model.api_name,
        "answer_only": answer_only,
        "qid": q["qid"],
        "question_type": q["question_type"],
        "gold_paper_index": q.get("gold_paper_index"),
        "n_distractors": q.get("n_distractors"),
        "variant": q.get("variant"),
        "K": q.get("K"),
        "needles_total": q.get("needles_total"),
        "needles_in_topk": q.get("needles_in_topk"),
        "num_chars": q.get("num_chars"),
        "input_tokens": input_tokens,
        "query": q["query"],
        "response": response,
        "parsed": parsed,
        **scores,
        "annotations": q["annotations"],
        "error": None,
    }
    if output_file:
        with lock:
            with open(output_file, "a") as f:
                f.write(json.dumps(record) + "\n")
    return record


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """Sample ~uniformly across question_type."""
    by_t: dict[str, list[dict]] = {}
    for r in records:
        by_t.setdefault(r["question_type"], []).append(r)
    rng = random.Random(seed)
    for v in by_t.values():
        rng.shuffle(v)
    types = sorted(by_t)
    base = n // len(types)
    extra = n - base * len(types)
    picks: list[dict] = []
    for i, t in enumerate(types):
        take = base + (1 if i < extra else 0)
        picks.extend(by_t[t][:take])
    rng.shuffle(picks)
    return picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--parallel", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "experiment_outputs"))
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--answer-only", action="store_true")
    ap.add_argument("--stratified", action="store_true",
                    help="Stratify the sample by question_type.")
    args = ap.parse_args()

    model = Model.from_string(args.model)
    queries = load_jsonl(args.queries)
    print(f"Loaded {len(queries)} per-query haystacks; "
          f"median size ~{sorted(q['num_chars'] for q in queries)[len(queries)//2]:,} chars")

    if args.stratified:
        qs = stratified_sample(queries, args.n, args.seed)
    else:
        rng = random.Random(args.seed)
        qs = rng.sample(queries, min(args.n, len(queries)))
    print(f"Sampled {len(qs)} queries "
          f"({{'extractive': {sum(1 for q in qs if q['question_type']=='extractive')}, "
          f"'abstractive': {sum(1 for q in qs if q['question_type']=='abstractive')}, "
          f"'boolean': {sum(1 for q in qs if q['question_type']=='boolean')}, "
          f"'none': {sum(1 for q in qs if q['question_type']=='none')}}})")

    output_file = None
    if not args.no_save:
        os.makedirs(args.out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = model.name.replace("/", "-")
        suffix = "_answer_only" if args.answer_only else ""
        output_file = os.path.join(args.out_dir,
                                   f"qasper_mp_{safe}{suffix}_{ts}.jsonl")
        print(f"Writing outputs to {output_file}")

    clients = setup_clients()
    lock = Lock()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = [ex.submit(run_one, clients, model, q, output_file,
                             args.answer_only, lock)
                   for q in qs]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 5 == 0 or i == len(futures):
                ok = [r for r in results if not r.get("error")]
                af = sum(r["answer_f1"] for r in ok) / max(1, len(ok))
                ef_vals = [r["evidence_f1"] for r in ok
                           if not args.answer_only and r.get("evidence_f1") is not None]
                ef = sum(ef_vals) / max(1, len(ef_vals)) if ef_vals else 0.0
                print(f"  [{i}/{len(futures)}] Ans-F1={af:.3f}"
                      f"{'' if args.answer_only else f' Evid-F1={ef:.3f}'}")

    ok = [r for r in results if not r.get("error")]
    print(f"\n=== Summary ===")
    print(f"Successful API calls: {len(ok)}/{len(results)}")
    if ok:
        af = sum(r["answer_f1"] for r in ok) / len(ok)
        print(f"  Answer F1  : {af:.3f}")
        if not args.answer_only:
            ef = sum(r["evidence_f1"] for r in ok) / len(ok)
            print(f"  Evidence F1: {ef:.3f}")
        by_t: dict[str, list[dict]] = {}
        for r in ok:
            by_t.setdefault(r["question_type"], []).append(r)
        for t, rs in sorted(by_t.items()):
            af = sum(r["answer_f1"] for r in rs) / len(rs)
            ef = (sum(r["evidence_f1"] for r in rs) / len(rs)) if not args.answer_only else float("nan")
            ev_str = "" if args.answer_only else f"  Evid-F1={ef:.3f}"
            print(f"  [{t:>14}]  n={len(rs):>3}  Ans-F1={af:.3f}{ev_str}")


if __name__ == "__main__":
    main()
