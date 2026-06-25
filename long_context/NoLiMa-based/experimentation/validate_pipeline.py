#!/usr/bin/env python3
"""
NoLiMa Pipeline Validator

Runs one (or more) random needle-in-haystack tests and writes a detailed
human-readable report so you can manually verify prompts, responses,
and evaluation correctness.

Usage:
    python validate_pipeline.py --model gemini-2.5-flash
    python validate_pipeline.py --model gemini-2.5-flash --context-length 10000 --seed 42
    python validate_pipeline.py --model gemini-2.5-flash --num-tests 3 --show-full-haystack
"""

import os
import sys
import asyncio
import argparse
import random
import time
from pathlib import Path
from copy import copy
from datetime import datetime

# ── Import shared logic from run_full_scale ──────────────────────────────
from run_full_scale import (
    MODELS,
    CONTEXT_LENGTHS,
    ALLOWED_NEEDLE_IDS,
    NUM_DEPTHS,
    NUM_CHARACTERS,
    BASE_SEED,
    REASONING_TYPE,
    CHARS_PER_TOKEN,
    PROJECT_DIR,
    DATA_DIR,
    NEEDLE_SET_PATH,
    HAYSTACK_DIR,
    RESULTS_DIR,
    DOCUMENT_DEPTH_PERCENT_MIN,
    DOCUMENT_DEPTH_PERCENT_MAX,
    load_tests,
    load_haystack,
    insert_needle,
    format_haystack_with_line_numbers,
    call_model,
    parse_json_response,
    evaluate_response,
    ModelConfig,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


REPORT_DIR = PROJECT_DIR / "validation_reports"


def pick_random_params(rng, args, config):
    """Pick random test parameters, respecting CLI overrides."""

    tests = load_tests()
    if not tests:
        print("ERROR: No tests loaded. Check needle set file.")
        sys.exit(1)

    test = rng.choice(tests)

    # Context length — filter to what the model supports
    available_contexts = [c for c in CONTEXT_LENGTHS if c <= config.max_context_chars]
    if args.context_length:
        if args.context_length not in available_contexts:
            print(f"WARNING: context length {args.context_length} not in available set. Using anyway.")
        context_length = args.context_length
    else:
        context_length = rng.choice(available_contexts)

    # Depth (0.0 – 1.0)
    if args.depth is not None:
        depth = args.depth / 100.0
    else:
        depths = [d / 100.0 for d in range(
            DOCUMENT_DEPTH_PERCENT_MIN,
            DOCUMENT_DEPTH_PERCENT_MAX + 1,
            max(1, (DOCUMENT_DEPTH_PERCENT_MAX - DOCUMENT_DEPTH_PERCENT_MIN) // (NUM_DEPTHS - 1))
        )]
        depth = rng.choice(depths)

    book_num = args.book_num or 1

    return test, context_length, depth, book_num


async def run_validation_test(config, test, context_length, depth, book_num, rng,
                               show_full_haystack=False):
    """Run one test and return a detailed report dict."""

    report = {}
    report["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report["model"] = config.name
    report["provider"] = config.provider

    # ── Test metadata ────────────────────────────────────────────────
    report["test_name"] = test["test_name"]
    report["context_length"] = context_length
    report["depth_percent"] = f"{depth * 100:.1f}%"
    report["book_num"] = book_num

    # ── Character selection ──────────────────────────────────────────
    if "{CHAR}" in test["needle"]:
        selected_character = str(rng.choice(test["character_set"]))
        needle = test["needle"].replace("{CHAR}", selected_character)
        question = test["retrieval_question"].replace("{CHAR}", selected_character)
    else:
        selected_character = ""
        needle = test["needle"]
        question = test["retrieval_question"]

    report["selected_character"] = selected_character
    report["needle_text"] = needle
    report["question"] = question

    # ── Load haystack & insert needle ────────────────────────────────
    try:
        haystack_lines, haystack_hash = load_haystack(context_length, book_num)
    except FileNotFoundError as e:
        report["error"] = f"Haystack file not found: {e}"
        return report

    report["haystack_total_lines"] = len(haystack_lines)

    haystack_with_needle, needle_position = insert_needle(haystack_lines, needle, depth)
    report["needle_line_number"] = needle_position
    report["haystack_total_lines_with_needle"] = len(haystack_with_needle)

    # ── Haystack excerpt around needle ───────────────────────────────
    excerpt_radius = 5
    start = max(0, needle_position - excerpt_radius)
    end = min(len(haystack_with_needle), needle_position + excerpt_radius + 1)
    excerpt_lines = []
    for i in range(start, end):
        marker = "  <<<< NEEDLE" if i == needle_position else ""
        excerpt_lines.append(f"  {i}: {haystack_with_needle[i]}{marker}")
    report["haystack_excerpt"] = "\n".join(excerpt_lines)

    # ── Build prompts (exactly as run_full_scale does) ───────────────
    formatted_haystack = format_haystack_with_line_numbers(haystack_with_needle)
    user_prompt = test["task_template"].format(haystack=formatted_haystack, question=question)
    system_prompt = test["system_prompt"]

    report["system_prompt"] = system_prompt
    report["user_prompt_length_chars"] = len(user_prompt)
    report["user_prompt_length_est_tokens"] = len(user_prompt) // CHARS_PER_TOKEN

    if show_full_haystack:
        report["user_prompt_full"] = user_prompt
    else:
        # Show first 500 + last 500 chars of the user prompt
        if len(user_prompt) > 1200:
            report["user_prompt_truncated"] = (
                user_prompt[:500]
                + f"\n\n... [{len(user_prompt) - 1000:,} chars omitted] ...\n\n"
                + user_prompt[-500:]
            )
        else:
            report["user_prompt_full"] = user_prompt

    # ── Call the model ───────────────────────────────────────────────
    print(f"  📡 Calling {config.name} ({report['user_prompt_length_est_tokens']:,} est. tokens)...")
    t0 = time.time()
    try:
        api_result = await call_model(config, system_prompt, user_prompt)
    except Exception as e:
        report["error"] = f"API call failed: {e}"
        return report
    elapsed = time.time() - t0

    raw_response = api_result.get("response", "")
    report["api_latency_seconds"] = round(elapsed, 2)
    report["input_tokens"] = api_result.get("input_tokens", 0)
    report["output_tokens"] = api_result.get("output_tokens", 0)
    report["raw_response"] = raw_response

    # ── Parse response ───────────────────────────────────────────────
    parsed = parse_json_response(raw_response)
    report["parsed_response"] = parsed

    # ── Evaluate ─────────────────────────────────────────────────────
    answer_metric, evidence_metric = evaluate_response(
        parsed, selected_character, needle_position, needle
    )
    report["answer_metric"] = answer_metric
    report["evidence_metric"] = evidence_metric

    # Expected values
    report["expected_answer"] = selected_character
    report["expected_line"] = needle_position
    if parsed:
        report["model_answer"] = parsed.get("answer", "N/A")
        report["model_lines"] = parsed.get("lines", [])
    else:
        report["model_answer"] = "UNPARSEABLE"
        report["model_lines"] = []

    # Verdict
    if answer_metric and evidence_metric:
        report["verdict"] = "✅ FULL PASS (answer + evidence correct)"
    elif answer_metric:
        report["verdict"] = "⚠️  PARTIAL PASS (answer correct, evidence wrong)"
    elif evidence_metric:
        report["verdict"] = "⚠️  PARTIAL PASS (answer wrong, evidence correct)"
    else:
        report["verdict"] = "❌ FAIL (both answer and evidence wrong)"

    return report


def format_report(report, test_num=1, total_tests=1):
    """Format report dict into a human-readable text string."""

    lines = []
    sep = "=" * 80

    lines.append(sep)
    lines.append(f"  NoLiMa PIPELINE VALIDATION REPORT  [{test_num}/{total_tests}]")
    lines.append(f"  Generated: {report.get('timestamp', 'N/A')}")
    lines.append(sep)

    # ── Metadata ─────────────────────────────────────────────────────
    lines.append("")
    lines.append("▸ TEST METADATA")
    lines.append(f"  Model:           {report.get('model', 'N/A')}")
    lines.append(f"  Provider:        {report.get('provider', 'N/A')}")
    lines.append(f"  Test name:       {report.get('test_name', 'N/A')}")
    lines.append(f"  Context length:  {report.get('context_length', 'N/A'):,} chars")
    lines.append(f"  Depth:           {report.get('depth_percent', 'N/A')}")
    lines.append(f"  Book:            {report.get('book_num', 'N/A')}")
    lines.append(f"  Character:       {report.get('selected_character', 'N/A')}")

    if "error" in report:
        lines.append("")
        lines.append(f"  ❌ ERROR: {report['error']}")
        lines.append(sep)
        return "\n".join(lines)

    # ── Needle ───────────────────────────────────────────────────────
    lines.append("")
    lines.append("▸ NEEDLE")
    lines.append(f"  Text:      \"{report.get('needle_text', 'N/A')}\"")
    lines.append(f"  Inserted at line: {report.get('needle_line_number', 'N/A')}")
    lines.append(f"  Haystack lines (original): {report.get('haystack_total_lines', 'N/A')}")
    lines.append(f"  Haystack lines (w/ needle): {report.get('haystack_total_lines_with_needle', 'N/A')}")

    # ── Question ─────────────────────────────────────────────────────
    lines.append("")
    lines.append("▸ QUESTION")
    lines.append(f"  {report.get('question', 'N/A')}")

    # ── Haystack excerpt ─────────────────────────────────────────────
    lines.append("")
    lines.append("▸ HAYSTACK EXCERPT (around needle)")
    lines.append(report.get("haystack_excerpt", "  N/A"))

    # ── System prompt ────────────────────────────────────────────────
    lines.append("")
    lines.append("▸ SYSTEM PROMPT")
    lines.append(f"  {report.get('system_prompt', 'N/A')}")

    # ── User prompt ──────────────────────────────────────────────────
    lines.append("")
    lines.append("▸ USER PROMPT")
    lines.append(f"  Length: {report.get('user_prompt_length_chars', 0):,} chars "
                 f"(~{report.get('user_prompt_length_est_tokens', 0):,} tokens)")
    if "user_prompt_full" in report:
        lines.append("  [Full prompt included below]")
        lines.append("-" * 40)
        lines.append(report["user_prompt_full"])
        lines.append("-" * 40)
    elif "user_prompt_truncated" in report:
        lines.append("  [Truncated — use --show-full-haystack to include full prompt]")
        lines.append("-" * 40)
        lines.append(report["user_prompt_truncated"])
        lines.append("-" * 40)

    # ── API response ─────────────────────────────────────────────────
    lines.append("")
    lines.append("▸ API RESPONSE")
    lines.append(f"  Latency:       {report.get('api_latency_seconds', 'N/A')}s")
    lines.append(f"  Input tokens:  {report.get('input_tokens', 'N/A'):,}")
    lines.append(f"  Output tokens: {report.get('output_tokens', 'N/A'):,}")
    lines.append("")
    lines.append("  Raw response:")
    lines.append("-" * 40)
    lines.append(f"  {report.get('raw_response', 'N/A')}")
    lines.append("-" * 40)
    lines.append("")
    lines.append(f"  Parsed JSON: {report.get('parsed_response', 'N/A')}")

    # ── Evaluation ───────────────────────────────────────────────────
    lines.append("")
    lines.append("▸ EVALUATION")
    lines.append(f"  Expected answer:     \"{report.get('expected_answer', 'N/A')}\"")
    lines.append(f"  Model answer:        \"{report.get('model_answer', 'N/A')}\"")
    lines.append(f"  Answer metric:       {report.get('answer_metric', 'N/A')} (1=correct)")
    lines.append(f"  Expected line:       {report.get('expected_line', 'N/A')}")
    lines.append(f"  Model lines:         {report.get('model_lines', 'N/A')}")
    lines.append(f"  Evidence metric:     {report.get('evidence_metric', 'N/A')} (1=correct)")

    # ── Verdict ──────────────────────────────────────────────────────
    lines.append("")
    lines.append(f"  ▸ VERDICT: {report.get('verdict', 'N/A')}")
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(
        description="NoLiMa Pipeline Validator — run random tests and inspect everything",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_pipeline.py --model gemini-2.5-flash
  python validate_pipeline.py --model gemini-2.5-flash --context-length 10000 --seed 42
  python validate_pipeline.py --model gemini-2.5-flash --num-tests 3 --depth 50
  python validate_pipeline.py --model gemini-2.5-flash --show-full-haystack
        """
    )
    parser.add_argument("--model", default=None, help="Model name (e.g. gemini-2.5-flash)")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Specific context length (default: random)")
    parser.add_argument("--depth", type=float, default=None,
                        help="Needle depth 0–100 (default: random)")
    parser.add_argument("--book-num", type=int, default=1,
                        help="Book number (default: 1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility (default: random)")
    parser.add_argument("--num-tests", type=int, default=1,
                        help="Number of random tests to run (default: 1)")
    parser.add_argument("--show-full-haystack", action="store_true",
                        help="Include the entire haystack in the report")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, cfg in MODELS.items():
            print(f"  {name} ({cfg.provider})")
        return

    if not args.model:
        parser.error("--model is required (use --list-models to see options)")

    if args.model not in MODELS:
        print(f"ERROR: Unknown model '{args.model}'. Use --list-models to see options.")
        sys.exit(1)

    config = MODELS[args.model]
    seed = args.seed if args.seed is not None else random.randint(0, 999999)
    rng = random.Random(seed)

    print(f"{'=' * 60}")
    print(f"  NoLiMa Pipeline Validator")
    print(f"{'=' * 60}")
    print(f"  Model:      {config.name}")
    print(f"  Seed:       {seed}")
    print(f"  Num tests:  {args.num_tests}")
    print()

    # Run tests
    all_reports = []
    for i in range(args.num_tests):
        test, context_length, depth, book_num = pick_random_params(rng, args, config)

        ctx_label = f"{context_length // 1000}K" if context_length < 1_000_000 else f"{context_length // 1_000_000}M"
        print(f"[{i+1}/{args.num_tests}] Test: {test['test_name']}  "
              f"Context: {ctx_label}  Depth: {depth*100:.0f}%  Book: {book_num}")

        report = await run_validation_test(
            config, test, context_length, depth, book_num, rng,
            show_full_haystack=args.show_full_haystack
        )
        all_reports.append(report)

        # Print quick verdict to console
        verdict = report.get("verdict", report.get("error", "UNKNOWN"))
        print(f"  → {verdict}")
        print()

    # ── Write report file ────────────────────────────────────────────
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = config.name.replace(".", "-").replace("/", "-")
    report_filename = f"{model_slug}_seed{seed}_{timestamp}.txt"
    report_path = REPORT_DIR / report_filename

    with open(report_path, "w") as f:
        for i, report in enumerate(all_reports):
            f.write(format_report(report, test_num=i + 1, total_tests=len(all_reports)))
            f.write("\n\n")

    print(f"{'=' * 60}")
    print(f"  Report saved to: {report_path}")
    print(f"{'=' * 60}")

    # Quick summary
    if len(all_reports) > 1:
        pass_count = sum(1 for r in all_reports if r.get("answer_metric", 0) == 1)
        evidence_count = sum(1 for r in all_reports if r.get("evidence_metric", 0) == 1)
        print(f"  Answer correct:   {pass_count}/{len(all_reports)}")
        print(f"  Evidence correct: {evidence_count}/{len(all_reports)}")


if __name__ == "__main__":
    asyncio.run(main())
