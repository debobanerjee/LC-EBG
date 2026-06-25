#!/usr/bin/env python3
"""
NoLiMa Contradictory Two-Needle Experiment
============================================

Tests whether models trust context over world knowledge when the secondary
needle contains a factually incorrect food/drink definition.

Setup:
  Primary needle:   Contains the character + condition (e.g. "Yuki is lactose
                    intolerant"). Fixed at 50% depth.
  Secondary needle: Contains a FALSE food/drink definition that contradicts
                    world knowledge (e.g. "Black coffee is made with steamed
                    whole milk"). Varies across 4 depths.
  Question:         Requires combining both needles (two-hop reasoning).
                    Correct answer is based ONLY on context, not world knowledge.

  Prompt:      Standard (answer + evidence / line citations)
  Book:        1
  Characters:  Yuki (CTN_0402/0402Inv), Stuart (CTN_0405/0405Inv)
  Depths:      4 levels (0%, 33%, 67%, 100%) for secondary needle
  Primary:     Fixed at 50% depth
  Tests:       4 needle pairs × twohop only = 4 tests per context

Results saved to: evaluation/special_experiments/results_{model}-contradictory-two-needle/

Usage:
  python run_contradictory_two_needle.py --dry-run       # Show what would be run
  python run_contradictory_two_needle.py                  # Run all models
  python run_contradictory_two_needle.py --model gpt-4o   # Run specific model
  python run_contradictory_two_needle.py --parallel       # Run providers in parallel
"""

import os
import sys
import json
import asyncio
import argparse
import hashlib
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Import shared infrastructure from run_full_scale ─────────────────────
from run_full_scale import (
    ModelConfig,
    MODELS,
    CONTEXT_LENGTHS,
    BASE_SEED,
    CHARS_PER_TOKEN,
    DATA_DIR,
    HAYSTACK_DIR,
    RESULTS_DIR,
    DOCUMENT_DEPTH_PERCENT_MIN,
    DOCUMENT_DEPTH_PERCENT_MAX,
    load_haystack,
    insert_needle,
    format_haystack_with_line_numbers,
    call_model,
    parse_json_response,
    evaluate_response,
)

# ============ Experiment Configuration ============

EXPERIMENT_TAG = "contradictory-two-needle"

NEEDLE_SET_PATH = DATA_DIR / "needlesets" / "contradictory_two_needle_set.json"

# Store results under special_experiments alongside the two-needle results
SPECIAL_RESULTS_DIR = RESULTS_DIR / "special_experiments"

NUM_DEPTHS = 4                  # secondary needle: 0%, 33%, 67%, 100%
PRIMARY_DEPTH = 0.50            # primary needle fixed at 50%
DEFAULT_BOOK = 1
MAX_ALLOWED_ERRORS = 1

# Standard prompt (answer + evidence)
SYSTEM_PROMPT = (
    "Your job is to answer the question entirely from the context "
    "and provide a reference. Your answer should cite all lines the "
    "answer is based on."
)

TASK_TEMPLATE = (
    "<context>{haystack}</context>\n\n"
    "Answer the question based on information only from the context. "
    "If the question is not answerable from the context, answer NA. "
    "Your response should comprise only the answer (just the character name) "
    "and all lines the answer is based on in json format. For example:\n"
    "```json\n"
    "{{\n"
    '  "lines": [25, 412],\n'
    '  "answer": "John",\n'
    "}}\n"
    "```\n\n"
    "Question: {question}"
)


# ============ Test Loading ============

def load_contradictory_tests() -> List[Dict]:
    """Load contradictory two-needle test configurations."""
    with open(NEEDLE_SET_PATH, "r") as f:
        needle_set = json.load(f)

    tests = []
    for entry in needle_set:
        character = entry["character"]
        primary = entry["primary_needle"].replace("{CHAR}", character)
        secondary = entry["secondary_needle"]
        question = entry["question"]

        tests.append({
            "test_id": entry["id"],
            "test_name": f"{entry['id']}_twohop",
            "primary_needle": primary,
            "secondary_needle": secondary,
            "question": question,
            "character": character,
            "expected_answer": entry["expected_answer"],
            "seed": BASE_SEED + int(entry["id"].split("_")[1][:4]),
        })

    return tests


# ============ Result Paths ============

def get_result_path(model_name: str, context_length: int, test_name: str, book_num: int) -> Path:
    model_dir = model_name.replace(".", "-").replace("/", "-")
    tag_dir = f"results_{model_dir}-{EXPERIMENT_TAG}"
    return (
        SPECIAL_RESULTS_DIR / tag_dir / "contradictory_two_needle"
        / f"rand_shuffle_{context_length}" / test_name
        / f"{model_dir}_rand_book_{book_num}_{test_name}.json"
    )


def find_existing_result(
    model_name: str, context_length: int, test_name: str, book_num: int
) -> Path:
    result_path = get_result_path(model_name, context_length, test_name, book_num)
    if not result_path.exists():
        return None
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
        results = data.get("results", [])
        if len(results) < NUM_DEPTHS:
            return None
        error_count = sum(
            1 for r in results
            if r.get("error") or r.get("error_type")
            or (r.get("response") is None and r.get("input_tokens", 0) == 0)
        )
        if error_count > MAX_ALLOWED_ERRORS:
            return None
        return result_path
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def result_exists(model_name: str, context_length: int, test_name: str, book_num: int) -> bool:
    return find_existing_result(model_name, context_length, test_name, book_num) is not None


# ============ Two-Needle Insertion ============

def insert_two_needles(
    haystack_lines: List[str],
    primary_needle: str,
    secondary_needle: str,
    primary_depth: float,
    secondary_depth: float,
) -> Tuple[List[str], int, int]:
    """Insert two needles into the haystack.

    Inserts the PRIMARY needle first (at primary_depth), then inserts
    the SECONDARY needle (at secondary_depth, computed on the updated
    haystack length).  Returns the modified haystack and the final
    line numbers of both needles.
    """
    result = haystack_lines.copy()
    primary_pos = int(len(result) * primary_depth)
    primary_pos = max(0, min(primary_pos, len(result)))
    result.insert(primary_pos, primary_needle)

    secondary_pos = int(len(result) * secondary_depth)
    secondary_pos = max(0, min(secondary_pos, len(result)))
    result.insert(secondary_pos, secondary_needle)

    primary_final = None
    secondary_final = None
    for i, line in enumerate(result):
        if line == primary_needle and primary_final is None:
            primary_final = i
        if line == secondary_needle and secondary_final is None:
            secondary_final = i

    return result, primary_final, secondary_final


# ============ Run Single Test ============

async def run_single_test(
    config: ModelConfig,
    test: Dict,
    context_length: int,
    book_num: int,
    haystack_lines: List[str],
    haystack_hash: str,
) -> Dict:
    """Run a single contradictory two-needle test (all secondary depths)."""

    np.random.seed(test["seed"] + book_num)

    result_path = get_result_path(config.name, context_length, test["test_name"], book_num)

    test_hash = hashlib.sha256(
        f"{test['test_name']}_{context_length}_{book_num}_{test['seed']}".encode()
    ).hexdigest()

    outputs = {
        "eval_name": f"{config.name}_rand_book_{book_num}_{test['test_name']}",
        "test_name": test["test_name"],
        "model_name": config.name,
        "question": test["question"],
        "primary_needle": test["primary_needle"],
        "secondary_needle": test["secondary_needle"],
        "character": test["character"],
        "expected_answer": test["expected_answer"],
        "system_prompt": SYSTEM_PROMPT,
        "task_template": TASK_TEMPLATE,
        "haystack_path": str(
            HAYSTACK_DIR / f"rand_shuffle_{context_length}" / f"rand_book_{book_num}.txt"
        ),
        "context_length": context_length,
        "primary_depth": PRIMARY_DEPTH,
        "secondary_depth_intervals": NUM_DEPTHS,
        "experiment_tag": EXPERIMENT_TAG,
        "result_dir": str(result_path.parent),
        "seed": test["seed"] + book_num,
        "results": [],
        "test_hash": test_hash,
    }

    secondary_depths = np.linspace(
        DOCUMENT_DEPTH_PERCENT_MIN, DOCUMENT_DEPTH_PERCENT_MAX, NUM_DEPTHS
    ) / 100

    prompts = []
    metadata = []

    for sec_depth in secondary_depths:
        haystack_with_needles, primary_line, secondary_line = insert_two_needles(
            haystack_lines,
            test["primary_needle"],
            test["secondary_needle"],
            PRIMARY_DEPTH,
            sec_depth,
        )

        formatted_haystack = format_haystack_with_line_numbers(haystack_with_needles)

        user_prompt = TASK_TEMPLATE.format(
            haystack=formatted_haystack, question=test["question"]
        )

        filled_prompt_length = len(SYSTEM_PROMPT) + len(user_prompt)

        prompts.append((SYSTEM_PROMPT, user_prompt))
        metadata.append({
            "character": test["character"],
            "primary_needle_line": primary_line,
            "secondary_needle_line": secondary_line,
            "primary_depth": PRIMARY_DEPTH,
            "secondary_depth": sec_depth,
            "num_lines": len(haystack_with_needles),
            "context_length_w_filled_template": filled_prompt_length,
        })

    # ── Call API in batches ──────────────────────────────────────────
    async def process_batch(batch_prompts):
        tasks = [call_model(config, sp, up) for sp, up in batch_prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    all_responses = [None] * len(prompts)

    for i in range(0, len(prompts), config.batch_size):
        batch_end = min(i + config.batch_size, len(prompts))
        batch_prompts = prompts[i:batch_end]

        responses = await process_batch(batch_prompts)
        for j, resp in enumerate(responses):
            all_responses[i + j] = resp

        if batch_end < len(prompts):
            await asyncio.sleep(config.batch_pause)

    # ── Process responses ────────────────────────────────────────────
    error_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for i, (resp, meta) in enumerate(zip(all_responses, metadata)):
        if isinstance(resp, Exception):
            error_count += 1
            continue

        raw_text = resp.get("response", "")
        if not raw_text or raw_text.strip() == "":
            error_count += 1
            continue

        parsed = parse_json_response(raw_text)
        if not parsed:
            error_count += 1
            continue

        # Answer metric: does response contain the expected character name?
        answer = str(parsed.get("answer", "")).lower()
        answer_correct = test["expected_answer"].lower() in answer

        # Evidence metric: do cited lines include BOTH needle lines?
        lines = parsed.get("lines", [])
        if not isinstance(lines, list):
            lines = [lines] if lines is not None else []
        int_lines = []
        for line in lines:
            try:
                int_lines.append(int(line))
            except (ValueError, TypeError):
                pass
        primary_cited = meta["primary_needle_line"] in int_lines
        secondary_cited = meta["secondary_needle_line"] in int_lines
        evidence_correct = primary_cited and secondary_cited

        result_entry = {
            "character": meta["character"],
            "primary_depth": meta["primary_depth"],
            "secondary_depth": meta["secondary_depth"],
            "placement_metadata": {
                "primary_needle_line": meta["primary_needle_line"],
                "secondary_needle_line": meta["secondary_needle_line"],
                "num_haystack_lines": meta["num_lines"],
            },
            "context_length_w_filled_template": meta["context_length_w_filled_template"],
            "response": parsed,
            "answer_metric": int(answer_correct),
            "evidence_metric": int(evidence_correct),
            "primary_cited": int(primary_cited),
            "secondary_cited": int(secondary_cited),
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp.get("output_tokens", 0),
        }

        outputs["results"].append(result_entry)
        total_input_tokens += resp["input_tokens"]
        total_output_tokens += resp.get("output_tokens", 0)

    success_count = len(outputs["results"])

    if success_count == 0:
        return outputs

    outputs["summary"] = {
        "total_depths": len(metadata),
        "error_count": error_count,
        "success_count": success_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(outputs, f, indent=2)

    return outputs


# ============ Main Runner ============

async def run_model_tests(
    model_name: str,
    context_lengths: List[int] = None,
    dry_run: bool = False,
) -> Dict:
    """Run all contradictory two-needle tests for a specific model."""

    if model_name not in MODELS:
        print(f"Error: Unknown model {model_name}")
        return {}

    config = MODELS[model_name]
    tests = load_contradictory_tests()

    all_context_lengths = context_lengths or CONTEXT_LENGTHS

    context_lengths = [c for c in all_context_lengths if c <= config.max_context_chars]
    skipped = [c for c in all_context_lengths if c > config.max_context_chars]
    if skipped:
        print(f"  Skipping contexts > {config.max_context_chars:,} chars: {skipped}")

    book_num = DEFAULT_BOOK
    total_tests = len(tests) * len(context_lengths)
    completed = 0
    pending = []

    for ctx in context_lengths:
        for test in tests:
            if result_exists(model_name, ctx, test["test_name"], book_num):
                completed += 1
            else:
                pending.append((ctx, test, book_num))

    print(f"\n{'=' * 60}")
    print(f"Contradictory Two-Needle: {model_name}")
    print(f"{'=' * 60}")
    print(f"Prompt:           Standard (answer + evidence)")
    print(f"Book:             {DEFAULT_BOOK}")
    print(f"Primary depth:    {PRIMARY_DEPTH*100:.0f}%")
    print(f"Secondary depths: {NUM_DEPTHS} (0%, 33%, 67%, 100%)")
    print(f"Tests:            {len(tests)} (4 needle pairs × twohop)")
    print(f"Total tests:      {total_tests}")
    print(f"Completed:        {completed}")
    print(f"Pending:          {len(pending)}")

    if dry_run:
        print("\nDry run — would run:")
        for ctx, test, book in pending[:15]:
            print(f"  {ctx:>7,} / {test['test_name']:<30s} / book_{book}")
        if len(pending) > 15:
            print(f"  ... and {len(pending) - 15} more")
        return {"completed": completed, "pending": len(pending)}

    if not pending:
        print("All tests completed!")
        return {"completed": completed, "pending": 0}

    print(f"\nRunning {len(pending)} tests ...")

    by_context = {}
    for ctx, test, book in pending:
        key = (ctx, book)
        if key not in by_context:
            by_context[key] = []
        by_context[key].append(test)

    pbar = tqdm(total=len(pending), desc=f"{model_name} (contradictory-TN)")
    errors = []

    for (context_length, book_num), tests_for_ctx in by_context.items():
        try:
            haystack_lines, haystack_hash = load_haystack(context_length, book_num)
        except FileNotFoundError:
            print(f"  Haystack not found: context {context_length}, book {book_num}")
            pbar.update(len(tests_for_ctx))
            continue

        for test in tests_for_ctx:
            try:
                await run_single_test(
                    config, test, context_length, book_num,
                    haystack_lines, haystack_hash,
                )
            except Exception as e:
                err_msg = str(e)[:80]
                errors.append(f"{context_length}/{test['test_name']}/book_{book_num}: {err_msg}")

            pbar.update(1)

            # Rate limiting
            calls_per_test = NUM_DEPTHS
            min_time_rpm = (calls_per_test / config.rpm_limit) * 60
            min_time = min_time_rpm
            if config.tpm_limit and config.tpm_limit > 0:
                est_input = context_length // CHARS_PER_TOKEN
                est_tokens = NUM_DEPTHS * (est_input + config.max_tokens)
                min_time_tpm = (est_tokens / config.tpm_limit) * 60
                min_time = max(min_time, min_time_tpm)
            batches = (NUM_DEPTHS + config.batch_size - 1) // config.batch_size
            pause_time = (batches - 1) * config.batch_pause
            additional = max(0, min_time - pause_time)
            if additional > 0:
                await asyncio.sleep(additional)

    pbar.close()

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:5]:
            print(f"  {e}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    return {
        "completed": completed + len(pending) - len(errors),
        "errors": len(errors),
    }


async def run_provider_models(
    provider: str,
    models: List[str],
    context_lengths: List[int],
    dry_run: bool,
) -> Dict[str, Dict]:
    if provider == "google" and len(models) > 1:
        tasks = [run_model_tests(m, context_lengths, dry_run) for m in models]
        model_results = await asyncio.gather(*tasks, return_exceptions=True)
        results = {}
        for model, res in zip(models, model_results):
            if isinstance(res, Exception):
                print(f"\nError running {model}: {res}")
                results[model] = {"errors": 1, "completed": 0}
            else:
                results[model] = res
        return results
    results = {}
    for model in models:
        results[model] = await run_model_tests(model, context_lengths, dry_run)
    return results


async def run_all_models(
    models: List[str] = None,
    context_lengths: List[int] = None,
    dry_run: bool = False,
    parallel: bool = False,
):
    models = models or list(MODELS.keys())
    tests = load_contradictory_tests()

    print("=" * 60)
    print("NoLiMa Contradictory Two-Needle Experiment")
    print("=" * 60)
    print(f"Models:           {models}")
    print(f"Prompt:           Standard (answer + evidence)")
    print(f"Primary depth:    {PRIMARY_DEPTH*100:.0f}%")
    print(f"Secondary depths: {NUM_DEPTHS} (0%, 33%, 67%, 100%)")
    print(f"Book:             {DEFAULT_BOOK}")
    print(f"Tests/model:      {len(tests)} tests × 1 book × up to "
          f"{len(context_lengths or CONTEXT_LENGTHS)} contexts × {NUM_DEPTHS} depths")
    print(f"Parallel:         {parallel}")

    if parallel:
        by_provider: Dict[str, List[str]] = {}
        for model in models:
            if model in MODELS:
                provider = MODELS[model].provider
                by_provider.setdefault(provider, []).append(model)

        print(f"\nRunning {len(by_provider)} providers in parallel:")
        for provider, provider_models in by_provider.items():
            print(f"  {provider}: {provider_models}")

        tasks = [
            run_provider_models(provider, provider_models, context_lengths, dry_run)
            for provider, provider_models in by_provider.items()
        ]
        provider_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for i, (provider, _) in enumerate(by_provider.items()):
            if isinstance(provider_results[i], Exception):
                print(f"\nError running {provider}: {provider_results[i]}")
            else:
                results.update(provider_results[i])
    else:
        results = {}
        for model in models:
            results[model] = await run_model_tests(model, context_lengths, dry_run)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model, stats in results.items():
        print(f"  {model}: completed={stats.get('completed', 0)}, "
              f"pending={stats.get('pending', 0)}, errors={stats.get('errors', 0)}")


# ============ CLI ============

def main():
    parser = argparse.ArgumentParser(
        description="NoLiMa Contradictory Two-Needle Experiment (all models)"
    )
    parser.add_argument("--model", help="Run specific model only")
    parser.add_argument("--context-length", type=int, help="Run specific context length only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--parallel", action="store_true",
                        help="Run different providers in parallel")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, config in MODELS.items():
            print(f"  {name} ({config.provider})")
        return

    if args.model:
        if args.model not in MODELS:
            print(f"Error: Unknown model '{args.model}'")
            print("Available:", list(MODELS.keys()))
            return
        models = [args.model]
    else:
        models = list(MODELS.keys())

    if not args.dry_run:
        providers_needed = set(MODELS[m].provider for m in models)
        missing_keys = []
        if "openai" in providers_needed and not os.getenv("OPENAI_API_KEY"):
            missing_keys.append("OPENAI_API_KEY")
        if "anthropic" in providers_needed and not os.getenv("ANTHROPIC_API_KEY"):
            missing_keys.append("ANTHROPIC_API_KEY")
        if "google" in providers_needed and not os.getenv("GOOGLE_API_KEY"):
            missing_keys.append("GOOGLE_API_KEY")
        if missing_keys:
            print(f"Error: Missing API keys: {missing_keys}")
            print("Set them with: export KEY_NAME='...'")
            return

    context_lengths = [args.context_length] if args.context_length else None

    asyncio.run(run_all_models(
        models=models,
        context_lengths=context_lengths,
        dry_run=args.dry_run,
        parallel=args.parallel,
    ))


if __name__ == "__main__":
    main()
