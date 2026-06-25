#!/usr/bin/env python3
"""
NoLiMa Answer-Only Experiment
==============================

Tests whether removing the evidence (line-citation) requirement from the
prompt improves LLM answer accuracy on needle-in-a-haystack tasks.

Setup:
  Model:       claude-sonnet-4-5-20250929 (Claude Sonnet 4.5)
  Prompt:      Answer-only (no line citation requested)
  Books:       2 per context length (book 1 + 1 random, seed 42)
  Characters:  Yuki (0402/0402Inv), Stuart (0405/0405Inv)
  Depths:      4 levels (0%, 33%, 67%, 100%)
  Tests:       T01 only (1 variant per needle × 4 needles × 2 hops = 8 tests)

Results saved to: evaluation/special_experiments/results_claude-sonnet-4-5-20250929-answer-only/
Result JSON format is identical to run_full_scale.py for cross-experiment
comparison.  evidence_metric will be 0 for all entries since line citations
are not requested.

Usage:
  python run_answer_only.py                          # Run the experiment
  python run_answer_only.py --dry-run                # Show what would be run
  python run_answer_only.py --context-length 100000  # Specific context length
"""

import os
import sys
import json
import asyncio
import argparse
import hashlib
import random
import re
import time
from pathlib import Path
from copy import copy
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Import shared logic from run_full_scale ──────────────────────────────
from run_full_scale import (
    ModelConfig,
    MODELS,
    CONTEXT_LENGTHS,
    ALLOWED_NEEDLE_IDS,
    REASONING_TYPE,
    BASE_SEED,
    CHARS_PER_TOKEN,
    CONTEXT_SAFETY_FACTOR,
    DATA_DIR,
    NEEDLE_SET_PATH,
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

MODEL_NAME = "claude-sonnet-4-5-20250929"
EXPERIMENT_TAG = "answer-only"
SPECIAL_RESULTS_DIR = RESULTS_DIR / "special_experiments"

# Reuse the existing model config (provider, batch_size, rpm, etc.)
MODEL_CONFIG = MODELS[MODEL_NAME]

NUM_DEPTHS = 4          # 0%, 33%, 67%, 100%
NUM_BOOKS = 2           # book 1 + 1 random per context length
AVAILABLE_BOOKS = 5     # rand_book_1.txt .. rand_book_5.txt
MAX_TESTS_PER_NEEDLE = 1  # T01 only
BOOK_SEED = 42          # Fixed seed for reproducible random book selection

# Fixed character assignment (same as all other experiments)
NEEDLE_CHARACTER = {
    "0402": "Yuki",
    "0402Inv": "Yuki",
    "0405": "Stuart",
    "0405Inv": "Stuart",
}

# ── Answer-Only Prompts ──────────────────────────────────────────────────
#
# Compared to the standard experiment, these prompts:
#   1. Remove "provide a reference" / "cite all lines" from the system prompt.
#   2. Remove the "lines" field from the JSON example in the task template.
#
# Everything else (context wrapping, question phrasing) stays identical.

ANSWER_ONLY_SYSTEM_PROMPT = (
    "Your job is to answer the question entirely from the context."
)

ANSWER_ONLY_TASK_TEMPLATE = (
    "<context>{haystack}</context>\n\n"
    "Answer the question based on information only from the context. "
    "If the question is not answerable from the context, answer NA. "
    "Your response should comprise only the answer (just the character name) "
    "in json format. For example:\n"
    "```json\n"
    "{{\n"
    '  "answer": "John"\n'
    "}}\n"
    "```\n\n"
    "Question: {question}"
)

MAX_ALLOWED_ERRORS = 1  # Results with more than 1 error are considered incomplete


# ============ Book Selection ============

def get_books_for_context(
    context_lengths: List[int], seed: int = BOOK_SEED
) -> Dict[int, List[int]]:
    """Return [book_1, random_book] per context length.

    Book 1 is always included.  The random book (1-5) is selected
    deterministically per context length using the given seed.
    The random book CAN be book 1 (giving two book-1 runs).
    """
    rng = random.Random(seed)
    books_map = {}
    for ctx in context_lengths:
        random_book = rng.randint(1, AVAILABLE_BOOKS)
        books_map[ctx] = sorted(set([1, random_book]))
    return books_map


# ============ Test Loading (Answer-Only Prompts) ============

def load_tests_answer_only() -> List[Dict]:
    """Load test configurations with answer-only prompts.

    Same needle/question content as the standard experiments, but uses
    modified system prompt and task template that only ask for the answer
    (no evidence / line citations).
    """
    with open(NEEDLE_SET_PATH, 'r') as f:
        needle_set = json.load(f)

    tests = []
    for exp_config in needle_set:
        if exp_config.get("reasoning_type") != REASONING_TYPE:
            continue

        exp_id = exp_config["id"]
        if exp_id not in ALLOWED_NEEDLE_IDS:
            continue

        character_set = exp_config.get("character_set", [])

        # Use fixed character for this needle type
        fixed_char = NEEDLE_CHARACTER.get(exp_id)
        limited_character_set = [fixed_char] if fixed_char else character_set[:1]

        for question_type, question in exp_config["questions"].items():
            test_count = 0
            for test_id, test_cfg in exp_config["tests"].items():
                if test_count >= MAX_TESTS_PER_NEEDLE:
                    break
                test_count += 1

                full_needle = exp_config["needle"]
                full_question = copy(question)

                # Fill in input args
                for arg_no, arg in enumerate(test_cfg["input_args"]):
                    placeholder = "{" + str(arg_no + 1) + "}"
                    full_needle = full_needle.replace(placeholder, arg)
                    full_question = full_question.replace(placeholder, arg)

                tests.append({
                    "test_name": f"{exp_id}_{test_id}_{question_type}",
                    "system_prompt": ANSWER_ONLY_SYSTEM_PROMPT,
                    "task_template": ANSWER_ONLY_TASK_TEMPLATE,
                    "needle": full_needle,
                    "retrieval_question": full_question,
                    "character_set": limited_character_set,
                    "seed": BASE_SEED + int(exp_id[:4]),
                })

    return tests


# ============ Result Paths ============

def get_result_path(context_length: int, test_name: str, book_num: int) -> Path:
    """Result file path under the answer-only results directory."""
    model_dir = MODEL_NAME.replace(".", "-").replace("/", "-")
    tag_dir = f"results_{model_dir}-{EXPERIMENT_TAG}"
    return (
        SPECIAL_RESULTS_DIR / tag_dir / REASONING_TYPE
        / f"rand_shuffle_{context_length}" / test_name
        / f"{model_dir}_rand_book_{book_num}_{test_name}.json"
    )


def find_existing_result(
    context_length: int, test_name: str, book_num: int
) -> Path:
    """Check if a valid result already exists."""
    result_path = get_result_path(context_length, test_name, book_num)
    if not result_path.exists():
        return None

    try:
        with open(result_path, 'r') as f:
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


def result_exists(context_length: int, test_name: str, book_num: int) -> bool:
    """Check if result file already exists and is valid."""
    return find_existing_result(context_length, test_name, book_num) is not None


# ============ Run Single Test ============

async def run_single_test(
    config: ModelConfig,
    test: Dict,
    context_length: int,
    book_num: int,
    haystack_lines: List[str],
    haystack_hash: str,
) -> Dict:
    """Run a single test (all depths) for one test/book combination.

    Identical to run_full_scale.run_single_test except:
      - Uses answer-only prompts (already embedded in `test` dict)
      - Adds experiment_tag to output JSON
    """

    # Initialize random state
    np.random.seed(test["seed"] + book_num)

    result_path = get_result_path(context_length, test["test_name"], book_num)

    test_hash = hashlib.sha256(
        f"{test['test_name']}_{context_length}_{book_num}_{test['seed']}".encode()
    ).hexdigest()

    outputs = {
        "eval_name": f"{config.name}_rand_book_{book_num}_{test['test_name']}",
        "test_name": test["test_name"],
        "model_name": config.name,
        "retrieval_question": test["retrieval_question"],
        "needle": test["needle"],
        "gold_answers": "",
        "system_prompt": test["system_prompt"],
        "use_default_system_prompt": False,
        "task_template": test["task_template"],
        "haystack_path": str(
            HAYSTACK_DIR / f"rand_shuffle_{context_length}" / f"rand_book_{book_num}.txt"
        ),
        "context_length": context_length,
        "character_set": test["character_set"],
        "document_depth_percent_min": DOCUMENT_DEPTH_PERCENT_MIN,
        "document_depth_percent_max": DOCUMENT_DEPTH_PERCENT_MAX,
        "document_depth_percent_intervals": NUM_DEPTHS,
        "shift": 0,
        "static_depth": -1,
        "metric": "contains",
        "result_dir": str(result_path.parent),
        "seed": test["seed"] + book_num,
        "experiment_tag": EXPERIMENT_TAG,
        "results": [],
        "test_hash": test_hash,
    }

    # Generate depths: 0%, 33%, 67%, 100%
    depths = np.linspace(
        DOCUMENT_DEPTH_PERCENT_MIN, DOCUMENT_DEPTH_PERCENT_MAX, NUM_DEPTHS
    ) / 100

    prompts = []
    metadata = []

    for depth in depths:
        # Select character (fixed per needle type)
        if "{CHAR}" in test["needle"]:
            selected_character = test["character_set"][0]
            needle = test["needle"].replace("{CHAR}", selected_character)
            question = test["retrieval_question"].replace("{CHAR}", selected_character)
        else:
            selected_character = ""
            needle = test["needle"]
            question = test["retrieval_question"]

        # Insert needle at depth
        haystack_with_needle, needle_position = insert_needle(
            haystack_lines, needle, depth
        )

        # Format with line numbers
        formatted_haystack = format_haystack_with_line_numbers(haystack_with_needle)

        # Fill template
        user_prompt = test["task_template"].format(
            haystack=formatted_haystack, question=question
        )

        filled_prompt_length = len(test["system_prompt"]) + len(user_prompt)

        prompts.append((test["system_prompt"], user_prompt))
        metadata.append({
            "selected_character": selected_character,
            "needle": needle,
            "needle_position": needle_position,
            "depth": depth,
            "num_lines": len(haystack_with_needle),
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
    ctx_label = (
        f"{context_length // 1000}K"
        if context_length < 1_000_000
        else f"{context_length // 1_000_000}M"
    )

    for i, (resp, meta) in enumerate(zip(all_responses, metadata)):
        depth_pct = f"{meta['depth'] * 100:.0f}%"

        if isinstance(resp, Exception):
            error_count += 1
            print(
                f"    ✗ depth={depth_pct} char={meta['selected_character']}"
                f"  ERROR: {str(resp)[:120]}"
            )
            continue

        raw_text = resp.get("response", "")
        if not raw_text or raw_text.strip() == "":
            error_count += 1
            print(
                f"    ✗ depth={depth_pct} char={meta['selected_character']}"
                f"  EMPTY RESPONSE (skipped)"
            )
            continue

        parsed = parse_json_response(raw_text)
        if not parsed:
            error_count += 1
            print(
                f"    ✗ depth={depth_pct} char={meta['selected_character']}"
                f"  UNPARSEABLE: {raw_text[:100]}"
            )
            continue

        answer_metric, evidence_metric = evaluate_response(
            parsed, meta["selected_character"], meta["needle_position"], meta["needle"]
        )

        result_entry = {
            "selected_character": meta["selected_character"],
            "context_length_w_filled_templated": meta["context_length_w_filled_template"],
            "placement_metadata": {
                "needle": meta["needle"],
                "needle_line_num": meta["needle_position"],
                "depth": meta["depth"],
                "num_haystack_lines_w_needle": meta["num_lines"],
            },
            "context_length_w_filled_template": meta["context_length_w_filled_template"],
            "response": parsed,
            "answer_metric": answer_metric,
            "evidence_metric": evidence_metric,
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp.get("output_tokens", 0),
        }

        outputs["results"].append(result_entry)
        total_input_tokens += resp["input_tokens"]
        total_output_tokens += resp.get("output_tokens", 0)

        ans_icon = "✓" if answer_metric else "✗"
        answer_text = str(parsed.get("answer", ""))[:40]
        print(
            f"    {ans_icon}  depth={depth_pct} char={meta['selected_character']}"
            f'  answer="{answer_text}"'
        )

    success_count = len(outputs["results"])

    if success_count == 0:
        print(f"  ⚠ ALL {len(metadata)} depths failed — result NOT saved")
        return outputs

    # Add summary stats
    outputs["summary"] = {
        "total_depths": len(metadata),
        "error_count": error_count,
        "success_count": success_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save results
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(outputs, f, indent=2)

    if error_count > 0:
        print(
            f"  Saved {success_count}/{len(metadata)} depths"
            f" ({error_count} failed depths skipped)"
        )

    return outputs


# ============ Main Runner ============

async def run_experiment(
    context_lengths: List[int] = None,
    dry_run: bool = False,
) -> Dict:
    """Run the answer-only experiment for all pending tests."""

    config = MODEL_CONFIG
    tests = load_tests_answer_only()

    all_context_lengths = context_lengths or CONTEXT_LENGTHS

    # Filter to model's max context
    context_lengths = [c for c in all_context_lengths if c <= config.max_context_chars]
    skipped = [c for c in all_context_lengths if c > config.max_context_chars]
    if skipped:
        print(f"  Skipping contexts > {config.max_context_chars:,} chars: {skipped}")

    # Book assignments: book 1 + 1 random per context length
    books_map = get_books_for_context(context_lengths)

    # Count tasks
    total_tests = sum(len(tests) * len(books_map[ctx]) for ctx in context_lengths)
    completed = 0
    pending = []

    for ctx in context_lengths:
        for test in tests:
            for book_num in books_map[ctx]:
                if result_exists(ctx, test["test_name"], book_num):
                    completed += 1
                else:
                    pending.append((ctx, test, book_num))

    print(f"\n{'=' * 60}")
    print(f"Answer-Only Experiment: {MODEL_NAME}")
    print(f"{'=' * 60}")
    print(f"Prompt:      Answer only (no evidence/line citations)")
    print(f"Books:       2 per context (book 1 + random, seed {BOOK_SEED})")
    print(f"Characters:  Yuki (0402*), Stuart (0405*)")
    print(f"Depths:      {NUM_DEPTHS}")
    print(f"Tests:       {len(tests)} (T01 × 4 needles × 2 hops)")

    # Show book assignments sample
    sample = list(books_map.items())[:8]
    assignments = ", ".join(f"{ctx // 1000}k→{bks}" for ctx, bks in sample)
    suffix = f", ... ({len(books_map)} total)" if len(books_map) > 8 else ""
    print(f"Book map:    {assignments}{suffix}")

    print(f"Total tests: {total_tests}")
    print(f"Completed:   {completed}")
    print(f"Pending:     {len(pending)}")

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

    # Group by (context_length, book) for efficient haystack loading
    by_context = {}
    for ctx, test, book in pending:
        key = (ctx, book)
        if key not in by_context:
            by_context[key] = []
        by_context[key].append(test)

    pbar = tqdm(total=len(pending), desc=f"{MODEL_NAME} (answer-only)")
    errors = []

    for (context_length, book_num), tests_for_ctx in by_context.items():
        ctx_label = (
            f"{context_length // 1000}K"
            if context_length < 1_000_000
            else f"{context_length // 1_000_000}M"
        )
        print(f"\n{'─' * 60}")
        print(
            f"  Context: {ctx_label} ({context_length:,} chars)"
            f" | Book: {book_num} | Tests: {len(tests_for_ctx)}"
        )
        print(f"{'─' * 60}")

        try:
            haystack_lines, haystack_hash = load_haystack(context_length, book_num)
        except FileNotFoundError:
            print(f"  ⚠ Haystack not found for context {context_length}, book {book_num}")
            pbar.update(len(tests_for_ctx))
            continue

        for test in tests_for_ctx:
            print(f"\n  Test: {test['test_name']}")
            try:
                await run_single_test(
                    config, test, context_length, book_num,
                    haystack_lines, haystack_hash,
                )
            except Exception as e:
                err_msg = str(e)[:80]
                errors.append(
                    f"{ctx_label}/{test['test_name']}/book_{book_num}: {err_msg}"
                )
                print(f"  ✗ FAILED: {err_msg}")

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


# ============ CLI ============

def main():
    parser = argparse.ArgumentParser(
        description="NoLiMa Answer-Only Experiment (Claude Sonnet 4.5)"
    )
    parser.add_argument(
        "--context-length", type=int,
        help="Run specific context length only",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be run without making API calls",
    )
    args = parser.parse_args()

    # Check API key
    if not args.dry_run and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        print("Set it with: export ANTHROPIC_API_KEY='...'")
        return

    context_lengths = [args.context_length] if args.context_length else None

    print("=" * 60)
    print("NoLiMa Answer-Only Experiment")
    print("=" * 60)
    print(f"Model:       {MODEL_NAME}")
    print(f"Experiment:  {EXPERIMENT_TAG}")
    print(f"Prompt:      Answer only (no evidence/line citations)")

    asyncio.run(run_experiment(
        context_lengths=context_lengths,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
