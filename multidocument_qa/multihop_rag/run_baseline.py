"""
EBG long-context baseline on MultiHop-RAG.

For each (subsampled) query we feed the full line-indexed haystack plus the
query to an LLM and request a JSON response of the form
``{"lines": [int, ...], "answer": str}`` — i.e. the same EBG protocol used
by the synthetic experiments.

Scoring:
  * ``Ans``  : answer exact-match (case-insensitive substring; Yes/No
               normalized) against the gold answer.
  * ``Evid`` : at least one gold evidence line per gold-evidence-fact
               appears in the model's cited lines (i.e. partial-recall@all
               facts). We also report strict-evidence recall.
  * ``Joint``: Ans AND Evid (both true).

Usage:
    python run_baseline.py --model gpt-5 --n 50
    python run_baseline.py --model sonnet-4-5 --n 50 --question-types inference_query

Outputs land in ``experiment_outputs/`` (relative to this script) as
JSONL, suffixed with the model name and timestamp, matching the convention
in ``experiments.py``.
"""

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
from datetime import datetime
from typing import Any

# Import the shared experiment infrastructure (model defs, clients, error
# handling, JSON parser). The shared engine `experiments.py` lives one level
# up, in `multidocument_qa/`.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO_ROOT)

from experiments import (  # noqa: E402
    MODELS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_ANSWER_ONLY,
    EXAMPLE_FORMAT,
    EXAMPLE_FORMAT_ANSWER_ONLY,
    setup_clients,
    run_model,
    process_answer,
    parse_llm_json_output,
    is_context_window_exceeded_error,
    ContextLengthExceeded,
)

from dotenv import load_dotenv  # noqa: E402
load_dotenv()


def load_queries(path: str) -> list[dict]:
    """Load JSONL records produced by ``build_haystack.py``.

    Each record has its own ``haystack`` string (per-query mini-haystack),
    plus ``query``, ``answer``, ``evidence_lines``, etc.
    """
    return [json.loads(line) for line in open(path)]


def normalize_answer(s: str) -> str:
    s = s.strip().strip(".").strip('"').strip("'").lower()
    # canonicalise yes/no answers
    if s in {"yes", "yes.", "y"}:
        return "yes"
    if s in {"no", "no.", "n"}:
        return "no"
    return s


def answer_match(pred: str, gold: str) -> bool:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not p:
        return False
    if p == g:
        return True
    # short gold (e.g. names, yes/no): require substring containment
    if len(g) <= 30:
        return g in p
    # long gold: rare in MultiHop-RAG; fall back to exact-match prefix
    return p.startswith(g[:40])


def evidence_score(pred_lines: list[int], gold_lines: list[int]) -> tuple[bool, float]:
    """Return (strict_match, recall).

    Strict: every gold line appears in pred.
    Recall: fraction of gold lines present in pred.
    """
    if not gold_lines:
        return False, 0.0
    pred_set = set(int(x) for x in pred_lines)
    hits = sum(1 for g in gold_lines if g in pred_set)
    return hits == len(gold_lines), hits / len(gold_lines)


def run_one(clients, model, q: dict, output_file: str | None,
            answer_only: bool = False) -> dict:
    if answer_only:
        user_prompt = (
            f"<context>\n{q['haystack']}\n</context>\n\n"
            f"Answer the question based on information only from the context. "
            f"If the question is not answerable from the context, answer NA. "
            f"Your response should only contain a short answer (no details) "
            f"in json format. For example:{EXAMPLE_FORMAT_ANSWER_ONLY}\n\n"
            f"Question: {q['query']}"
        )
        system_prompt_used = SYSTEM_PROMPT_ANSWER_ONLY
    else:
        user_prompt = (
            f"<context>\n{q['haystack']}\n</context>\n\n"
            f"Answer the question based on information only from the context. "
            f"If the question is not answerable from the context, answer NA. "
            f"Your response should only contain a short answer (no details) "
            f"and all lines the answer is based on in json format. "
            f"For example:{EXAMPLE_FORMAT}\n\n"
            f"Question: {q['query']}"
        )
        system_prompt_used = SYSTEM_PROMPT

    try:
        result = run_model(clients, model, user_prompt, system_prompt=system_prompt_used)
    except Exception as e:
        err = str(e)
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": model.api_name,
            "qid": q["qid"],
            "question_type": q["question_type"],
            "query": q["query"],
            "num_chars": q.get("num_chars"),
            "num_lines": q.get("num_lines"),
            "gold_answer": q["answer"],
            "gold_lines": q["evidence_lines"],
            "response": None,
            "parsed": None,
            "answer_correct": False,
            "evidence_strict": False,
            "evidence_recall": 0.0,
            "joint": False,
            "input_tokens": 0,
            "error": err,
        }
        if is_context_window_exceeded_error(e):
            record["error_kind"] = "context_length_exceeded"
        if output_file:
            with open(output_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        if is_context_window_exceeded_error(e):
            raise ContextLengthExceeded(err) from e
        return record

    response, input_tokens = process_answer(result, model)
    parsed = parse_llm_json_output(response) or {}
    pred_answer = str(parsed.get("answer", ""))
    pred_lines = parsed.get("lines", []) or []
    ans_ok = answer_match(pred_answer, q["answer"])
    if answer_only:
        # No evidence requested -> joint == answer; evidence metrics undefined.
        ev_strict, ev_recall = False, 0.0
        joint = ans_ok
    else:
        ev_strict, ev_recall = evidence_score(pred_lines, q["evidence_lines"])
        joint = ans_ok and ev_strict
    record = {
        "timestamp": datetime.now().isoformat(),
        "model": model.api_name,
        "answer_only": answer_only,
        "qid": q["qid"],
        "question_type": q["question_type"],
        "query": q["query"],
        "num_chars": q.get("num_chars"),
        "num_lines": q.get("num_lines"),
        "gold_answer": q["answer"],
        "gold_lines": q["evidence_lines"],
        "response": response,
        "parsed": parsed,
        "answer_correct": ans_ok,
        "evidence_strict": ev_strict,
        "evidence_recall": ev_recall,
        "joint": joint,
        "input_tokens": input_tokens,
        "error": None,
    }
    if output_file:
        with open(output_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", required=True, help="see experiments.MODELS")
    ap.add_argument("--queries", default=os.path.join(HERE, "processed", "smoke_d20.jsonl"),
                    help="Per-query JSONL produced by build_haystack.py "
                         "(each record carries its own mini-haystack).")
    ap.add_argument("--n", type=int, default=50, help="number of queries to sample (default: 50)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--question-types", default=None,
                    help="comma-separated list, e.g. 'inference_query,temporal_query'")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "experiment_outputs"))
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--answer-only", action="store_true",
                    help="Run the answer-only baseline (no citation requirement).")
    args = ap.parse_args()

    if args.model not in MODELS:
        print(f"Unknown model {args.model}; pick from: {list(MODELS)[:5]} ...")
        sys.exit(1)
    model = MODELS[args.model]

    qs = load_queries(args.queries)
    if qs:
        import statistics
        mc = int(statistics.median(q["num_chars"] for q in qs))
        print(f"Loaded {len(qs)} per-query haystacks; median size ~{mc:,} chars")
    if args.question_types:
        wanted = set(args.question_types.split(","))
        qs = [q for q in qs if q["question_type"] in wanted]
    rng = random.Random(args.seed)
    rng.shuffle(qs)
    qs = qs[: args.n]
    print(f"Sampled {len(qs)} queries"
          + (f" of types {args.question_types}" if args.question_types else ""))

    clients = setup_clients()
    if model.provider.value not in clients:
        print(f"Provider {model.provider.value} client unavailable; check API key.")
        sys.exit(1)

    output_file = None
    if not args.no_save:
        os.makedirs(args.out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = model.name.replace("/", "-")
        suffix = "_answer_only" if args.answer_only else ""
        output_file = os.path.join(args.out_dir, f"multihop_rag_{safe}{suffix}_{ts}.jsonl")
        print(f"Writing outputs to {output_file}")

    # Run with bounded parallelism.
    results: list[dict] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = [ex.submit(run_one, clients, model, q, output_file, args.answer_only) for q in qs]
            for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                if i % 5 == 0:
                    ok = sum(1 for r in results if r.get("answer_correct"))
                    j = sum(1 for r in results if r.get("joint"))
                    print(f"  [{i}/{len(qs)}] ans={ok}/{len(results)}  joint={j}/{len(results)}")
    except ContextLengthExceeded:
        print("Stopping: prompt exceeded model context window.")

    # Summary.
    good = [r for r in results if not r.get("error")]
    print("\n=== Summary ===")
    print(f"Successful API calls: {len(good)}/{len(results)}")
    if good:
        ans = sum(1 for r in good if r["answer_correct"]) / len(good)
        ev_strict = sum(1 for r in good if r["evidence_strict"]) / len(good)
        ev_recall = sum(r["evidence_recall"] for r in good) / len(good)
        joint = sum(1 for r in good if r["joint"]) / len(good)
        print(f"  Ans            : {ans:.3f}")
        print(f"  Evid (strict)  : {ev_strict:.3f}")
        print(f"  Evid (recall)  : {ev_recall:.3f}")
        print(f"  Ans+Evid       : {joint:.3f}")
        # break out by question type
        by_type: dict[str, list[dict]] = {}
        for r in good:
            by_type.setdefault(r["question_type"], []).append(r)
        for t, rs in by_type.items():
            a = sum(1 for r in rs if r["answer_correct"]) / len(rs)
            j = sum(1 for r in rs if r["joint"]) / len(rs)
            print(f"  [{t:>20}]  n={len(rs):3d}  ans={a:.2f}  joint={j:.2f}")


if __name__ == "__main__":
    main()
