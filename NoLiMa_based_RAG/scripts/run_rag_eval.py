#!/usr/bin/env python3
"""
RAG Top-K Evaluation Script
============================
Loads pre-computed retrieval results from scroll_all, selects top-k chunks,
builds prompts using NoLiMa needle_set templates, calls the LLM, evaluates
the response, and saves results in NoLiMa-compatible format.

Usage:
    python3 NoLiMa_based_RAG/scripts/run_rag_eval.py --config NoLiMa_based_RAG/configs/rag_eval.yaml
    python3 NoLiMa_based_RAG/scripts/run_rag_eval.py --config NoLiMa_based_RAG/configs/rag_eval.yaml --dry-run
"""

import argparse
import asyncio
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# LLM clients (lazy-initialised)
# ---------------------------------------------------------------------------
_oai_client = None
_anthropic_client = None


def _get_openai_client():
    global _oai_client
    if _oai_client is None:
        import openai
        _oai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _oai_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
async def call_llm(
    model_name: str,
    provider: str,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """
    Call the LLM and return {"response_text": str, "input_tokens": int}.
    Uses the OpenAI Responses API for openai provider.
    """
    if provider == "openai":
        client = _get_openai_client()
        result = await client.responses.create(
            model=model_name,
            instructions=system_prompt,
            input=user_prompt,
        )
        response_text = result.output_text
        input_tokens = result.usage.input_tokens if result.usage else 0
    elif provider == "anthropic":
        client = _get_anthropic_client()
        result = await client.messages.create(
            model=model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=1024,
        )
        response_text = result.content[0].text
        input_tokens = result.usage.input_tokens if result.usage else 0
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return {"response_text": response_text, "input_tokens": input_tokens}


async def call_llm_dry_run(
    model_name: str,
    provider: str,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """
    Fake LLM call for dry-run mode.
    Returns a synthetic response with a dummy character and line number.
    """
    fake_response = '```json\n{\n  "lines": [0],\n  "answer": "DRY_RUN_ANSWER"\n}\n```'
    return {"response_text": fake_response, "input_tokens": 0}


# ---------------------------------------------------------------------------
# Auto-discover context lengths
# ---------------------------------------------------------------------------
def discover_context_lengths(scroll_all_dir: str) -> List[int]:
    """
    Scan scroll_all_dir for numeric subdirectories and return them sorted.
    e.g. scroll_all/10000/, scroll_all/200000/ → [10000, 200000, ...]
    """
    base = Path(scroll_all_dir)
    if not base.exists():
        raise FileNotFoundError(f"scroll_all_dir not found: {base}")

    lengths = []
    for child in base.iterdir():
        if child.is_dir():
            try:
                lengths.append(int(child.name))
            except ValueError:
                # skip non-numeric dirs
                continue
    lengths.sort()
    return lengths


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------
def load_scroll_all(scroll_all_dir: str, context_length: int,
                    rand_book: int, q_id: str) -> List[Dict[str, Any]]:
    """Load the pre-computed retrieval JSON for a specific (context_length, rand_book, q_id)."""
    source_file = f"{context_length}_rand_book_{rand_book}"
    filename = f"{source_file}_{q_id}.json"
    filepath = Path(scroll_all_dir) / str(context_length) / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Scroll-all file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def select_top_k(hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Select the first top_k chunks (already sorted by score descending)."""
    return hits[:top_k]


def sort_by_evidence(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort chunks by evidence (line number) in ascending order."""
    return sorted(hits, key=lambda x: int(x["evidence"]))


def format_context(hits: List[Dict[str, Any]]) -> str:
    """Format chunks as 'Line number {evidence}: {text}' joined by newlines."""
    lines = []
    for h in hits:
        evidence = int(h["evidence"])
        text = (h.get("text") or "").strip()
        lines.append(f"Line number {evidence}: {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Character extraction from needle text
# ---------------------------------------------------------------------------
CHARACTER_SET = [
    "Yuki", "Stuart", "Katie", "Veronica", "Gary",
    "Megan", "Calvin", "Mandy", "Diana", "Caleb",
]


def extract_character_from_needle(needle_text: str,
                                  character_set: List[str] = None) -> Optional[str]:
    """Extract the character name from the baked-in needle text."""
    chars = character_set or CHARACTER_SET
    for name in chars:
        if name in needle_text:
            return name
    return None


# ---------------------------------------------------------------------------
# Find needle chunk in top-k hits
# ---------------------------------------------------------------------------
def find_needle_in_hits(hits: List[Dict[str, Any]],
                        needle_item_id: str) -> Optional[Dict[str, Any]]:
    """Find the needle chunk within the selected hits (if present)."""
    for h in hits:
        if h.get("chunk_type") == "needle" and str(h.get("needle_item_id")) == str(needle_item_id):
            return h
    return None


# ---------------------------------------------------------------------------
# Response parsing & evaluation
# ---------------------------------------------------------------------------
def parse_llm_json_output(output_text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON with 'lines' and 'answer' keys from LLM output."""
    # Try backtick-fenced JSON first
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'(\{.*?\})', output_text, re.DOTALL)

    if not json_match:
        return None

    json_str = json_match.group(1)
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        try:
            import json5
            result = json5.loads(json_str)
        except Exception:
            return None

    if isinstance(result, dict) and "answer" in result and "lines" in result:
        return result
    return None


def evaluate_response(
    parsed: Optional[Dict[str, Any]],
    gold_character: str,
    gold_evidence: int,
    evidence_type: str = "line_num",
) -> Tuple[int, int]:
    """
    Evaluate the parsed LLM response against gold standard.
    Returns (answer_metric, evidence_metric) each 0 or 1.
    """
    if parsed is None:
        return 0, 0

    # Answer evaluation
    answer_metric = 0
    llm_answer = str(parsed.get("answer", "")).strip().strip(".")
    if gold_character and gold_character in llm_answer:
        answer_metric = 1

    # Evidence evaluation
    evidence_metric = 0
    llm_lines = parsed.get("lines")
    if llm_lines is not None:
        if evidence_type == "line_num":
            if isinstance(llm_lines, list) and gold_evidence in llm_lines:
                evidence_metric = 1
            elif llm_lines == gold_evidence:
                evidence_metric = 1

    return answer_metric, evidence_metric


# ---------------------------------------------------------------------------
# Async batch helper
# ---------------------------------------------------------------------------
async def gather_in_batches(coros, batch_size=20, batch_pause=1.0):
    out = []
    for i in range(0, len(coros), batch_size):
        batch = coros[i : i + batch_size]
        out.extend(await asyncio.gather(*batch))
        await asyncio.sleep(batch_pause)
    return out


# ---------------------------------------------------------------------------
# Build needle_set index
# ---------------------------------------------------------------------------
def build_needle_set_index(
    needle_set_path: str, task_ids: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Load needle_set and index by task id."""
    with open(needle_set_path, "r", encoding="utf-8") as f:
        needle_set = json.load(f)
    index = {}
    for item in needle_set:
        if item["id"] in task_ids:
            index[item["id"]] = item
    return index


# ---------------------------------------------------------------------------
# Build questions index
# ---------------------------------------------------------------------------
def build_questions_index(questions_path: str) -> Dict[str, Dict[str, Any]]:
    """Load questions.json and index by q_id (without embeddings)."""
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    index = {}
    for q in questions:
        qid = q.get("q_id")
        if qid:
            meta = {k: v for k, v in q.items() if k != "embedding"}
            index[qid] = meta
    return index


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
async def run_evaluation(cfg: Dict[str, Any], dry_run: bool = False) -> None:
    scroll_all_dir = cfg["scroll_all_dir"]
    top_k_values = cfg["top_k_values"]
    rand_books = cfg["rand_books"]
    task_ids = cfg["task_ids"]
    test_key = cfg["test_key"]
    do_sort = cfg.get("sort_by_evidence", True)
    base_seed = cfg.get("base_seed", 42)
    metric = cfg.get("metric", "contains")
    evidence_type = cfg.get("evidence_type", "line_num")
    output_dir = cfg["output_dir"]

    # Resolve models list (support both old single-model and new multi-model config)
    if "models" in cfg and cfg["models"]:
        models = [(m["name"], m["provider"]) for m in cfg["models"]]
    elif "model_name" in cfg:
        # Backward compatibility with single model config
        models = [(cfg["model_name"], cfg["model_provider"])]
    else:
        raise ValueError("No models configured. Provide 'models' list or 'model_name'/'model_provider'.")

    # Resolve context lengths: auto-discover or use explicit list
    if cfg.get("auto_discover_contexts", False):
        context_lengths = discover_context_lengths(scroll_all_dir)
        print(f"Auto-discovered {len(context_lengths)} context lengths from {scroll_all_dir}/")
    else:
        context_lengths = cfg.get("context_lengths") or []
        if not context_lengths:
            raise ValueError("No context_lengths provided and auto_discover_contexts is false.")

    # ---- Apply filters from config ----
    limit_ctx = cfg.get("limit_context_lengths") or []
    if limit_ctx:
        context_lengths = sorted(limit_ctx)
        print(f"Using limit_context_lengths: {context_lengths}")

    limit_topk = cfg.get("limit_top_k_values") or []
    if limit_topk:
        top_k_values = sorted(limit_topk)
        print(f"Using limit_top_k_values: {top_k_values}")

    limit_qids = cfg.get("limit_q_ids") or []

    # Load needle_set and questions indices
    needle_set_index = build_needle_set_index(cfg["needle_set_path"], task_ids)
    questions_index = build_questions_index(cfg["questions_json"])

    # Build the list of q_ids from task_ids x question_keys
    q_id_list = []
    for task_id in task_ids:
        ns = needle_set_index[task_id]
        for question_key in ns["questions"]:
            q_id = f"{task_id}_{question_key}"
            if q_id in questions_index:
                q_id_list.append(q_id)

    # Apply q_id filter
    if limit_qids:
        q_id_list = [q for q in q_id_list if q in limit_qids]
        print(f"Filtered q_ids to: {q_id_list}")

    total_tasks = len(models) * len(context_lengths) * len(rand_books) * len(q_id_list) * len(top_k_values)
    print(f"Total evaluation tasks: {total_tasks}")
    print(f"  Models: {[f'{n} ({p})' for n, p in models]}")
    print(f"  Context lengths ({len(context_lengths)}): {context_lengths}")
    print(f"  Top-k values: {top_k_values}")
    print(f"  Q-IDs: {q_id_list}")
    print(f"  Rand books: {rand_books}")
    print()

    if dry_run:
        print("*** DRY-RUN MODE: No real LLM calls will be made ***\n")

    task_counter = 0

    for model_name, provider in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({provider})")
        print(f"{'='*60}")

        for context_length in context_lengths:
            for rand_book in rand_books:
                source_file = f"{context_length}_rand_book_{rand_book}"

                for q_id in q_id_list:
                    q_meta = questions_index[q_id]
                    task_id = q_meta["task_id"]
                    question_key = q_meta["question_key"]
                    question_text = q_meta["question"]
                    ns = needle_set_index[task_id]

                    system_prompt = ns["system_prompt"]
                    task_template = ns["task_template"]
                    character_set = ns.get("character_set", CHARACTER_SET)

                    # Load pre-computed retrieval results once per (context_length, rand_book, q_id)
                    try:
                        all_hits = load_scroll_all(scroll_all_dir, context_length, rand_book, q_id)
                    except FileNotFoundError as e:
                        print(f"  [SKIP] {e}")
                        continue

                    # Find needle in ALL hits for gold answer (independent of top_k)
                    needle_hit_all = find_needle_in_hits(all_hits, task_id)
                    gold_character = None
                    gold_evidence = None
                    needle_text = None
                    if needle_hit_all:
                        needle_text = needle_hit_all.get("text", "")
                        gold_character = extract_character_from_needle(needle_text, character_set)
                        gold_evidence = int(needle_hit_all["evidence"])

                    # Build one eval item per top_k value
                    eval_items = []
                    for top_k in top_k_values:
                        task_counter += 1

                        top_hits = select_top_k(all_hits, top_k)
                        needle_hit = find_needle_in_hits(top_hits, task_id)

                        if do_sort:
                            display_hits = sort_by_evidence(top_hits)
                        else:
                            display_hits = top_hits

                        context_str = format_context(display_hits)
                        user_prompt = task_template.replace("{haystack}", context_str).replace("{question}", question_text)

                        eval_items.append({
                            "top_k": top_k,
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                            "needle_in_topk": needle_hit is not None,
                            "top_k_count": len(top_hits),
                        })

                    if not eval_items:
                        continue

                    # Choose real or fake LLM call
                    llm_fn = call_llm_dry_run if dry_run else call_llm

                    async_tasks = [
                        llm_fn(model_name, provider, item["system_prompt"], item["user_prompt"])
                        for item in eval_items
                    ]

                    mode_label = "[DRY-RUN] " if dry_run else ""
                    print(f"{mode_label}[{model_name} | ctx={context_length}, book={rand_book}, {q_id}] "
                          f"Calling LLM for {len(async_tasks)} top_k values: {top_k_values}...")

                    responses = await gather_in_batches(async_tasks, batch_size=20, batch_pause=1.0)

                    # Build test_name and eval_name
                    test_name = f"{task_id}_{test_key}_{question_key}"
                    eval_name = f"{model_name}_rand_book_{rand_book}_{test_name}"

                    # Collect all top_k results into one results array
                    results_array = []
                    for item, llm_result in zip(eval_items, responses):
                        response_text = llm_result["response_text"]
                        input_tokens = llm_result["input_tokens"]

                        parsed = parse_llm_json_output(response_text)
                        answer_metric, evidence_metric = evaluate_response(
                            parsed, gold_character, gold_evidence, evidence_type
                        )

                        results_array.append({
                            "top_k": item["top_k"],
                            "selected_character": gold_character,
                            "gold_evidence": gold_evidence,
                            "answer_metric": answer_metric,
                            "evidence_metric": evidence_metric,
                            "needle_in_topk": item["needle_in_topk"],
                            "response": parsed,
                            "raw_response": response_text,
                            "input_tokens": input_tokens,
                            "num_chunks_retrieved": item["top_k_count"],
                        })

                        status = "✓" if answer_metric else "✗"
                        print(f"  {status} top_k={item['top_k']:>3} | answer={answer_metric} evidence={evidence_metric} "
                              f"| needle_in_topk={item['needle_in_topk']}")

                    # Build single result file for this (model, context_length, rand_book, q_id)
                    result_output = {
                        "eval_name": eval_name,
                        "test_name": test_name,
                        "model_name": model_name,
                        "retrieval_question": question_text,
                        "needle": needle_text,
                        "system_prompt": system_prompt,
                        "task_template": task_template,
                        "context_length": context_length,
                        "source_file": source_file,
                        "sort_by_evidence": do_sort,
                        "character_set": character_set,
                        "seed": base_seed,
                        "metric": metric,
                        "evidence_type": evidence_type,
                        "top_k_values": top_k_values,
                        "results": results_array,
                    }

                    # Save result
                    # Path: {output_dir}/results_{model}_w_rag/commonsense_knowledge/task_5p1/rand_shuffle_{ctx}/{test_name}/{eval_name}.json
                    result_dir = (
                        Path(output_dir)
                        / f"results_{model_name}_w_rag"
                        / f"{context_length}"
                    )
                    result_dir.mkdir(parents=True, exist_ok=True)
                    result_path = result_dir / f"{eval_name}.json"

                    with open(result_path, "w", encoding="utf-8") as f:
                        json.dump(result_output, f, indent=4, ensure_ascii=False)

                    print(f"  → Saved {result_path.name}")

    print(f"\nDone. Processed {task_counter} tasks total.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="RAG Top-K Evaluation")
    ap.add_argument("--config", required=True, help="Path to rag_eval.yaml config file")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip real LLM calls; write results with fake responses for testing")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Pull API keys from environment if not in config
    if not cfg.get("openai_api_key"):
        cfg["openai_api_key"] = os.environ.get("OPENAI_API_KEY")
    if not cfg.get("anthropic_api_key"):
        cfg["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY")

    asyncio.run(run_evaluation(cfg, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
