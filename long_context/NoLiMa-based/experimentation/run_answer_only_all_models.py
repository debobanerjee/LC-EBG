#!/usr/bin/env python3
"""
NoLiMa Answer-Only Experiment — All Models
==========================================

Runs the answer-only prompt experiment across all standard models.
Provider groups (Anthropic, OpenAI, Google) execute in parallel since they
use separate API keys and rate limits. Models within the same provider run
sequentially to stay within shared rate limits.

Setup:
  Prompt:      Answer-only (no line citation requested)
  Books:       Book 1 only
  Characters:  Yuki (0402/0402Inv), Stuart (0405/0405Inv)
  Depths:      4 levels (0%, 33%, 67%, 100%)
  Tests:       T01 only (4 needles × 2 hops = 8 tests)

Results saved to:
  evaluation/special_experiments/results_{model}-answer-only/

Usage:
  python run_answer_only_all_models.py                          # Run all models
  python run_answer_only_all_models.py --dry-run                # Show pending work
  python run_answer_only_all_models.py --model gpt-4.1          # One model only
  python run_answer_only_all_models.py --context-length 100000  # One context length
"""

import os
import sys
import json
import asyncio
import argparse
import hashlib
import time
from copy import copy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from run_full_scale import (
    ModelConfig,
    MODELS,
    CONTEXT_LENGTHS,
    ALLOWED_NEEDLE_IDS,
    REASONING_TYPE,
    BASE_SEED,
    CHARS_PER_TOKEN,
    NEEDLE_CHARACTER,
    MAX_TESTS_PER_NEEDLE,
    NUM_DEPTHS,
    RESULTS_DIR,
    NEEDLE_SET_PATH,
    HAYSTACK_DIR,
    DOCUMENT_DEPTH_PERCENT_MIN,
    DOCUMENT_DEPTH_PERCENT_MAX,
    load_haystack,
    insert_needle,
    format_haystack_with_line_numbers,
    call_model,
    parse_json_response,
    evaluate_response,
)

from run_answer_only import (
    ANSWER_ONLY_SYSTEM_PROMPT,
    ANSWER_ONLY_TASK_TEMPLATE,
    MAX_ALLOWED_ERRORS,
)

# ─── Configuration ────────────────────────────────────────────────────────────

EXPERIMENT_TAG = "answer-only"
SPECIAL_ROOT   = RESULTS_DIR / "special_experiments"
BOOK           = 1  # Book 1 only for all models

MODEL_DISPLAY = {
    "claude-sonnet-4-20250514":   "Claude Sonnet 4",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "gemini-2.5-flash":           "Gemini 2.5 Flash",
    "gemini-3-flash-preview":     "Gemini 3 Flash",
    "gpt-4o":                     "GPT-4o",
    "gpt-4.1":                    "GPT-4.1",
    "gpt-5-2025-08-07":           "GPT-5",
    "o3-mini-2025-01-31":         "o3-mini",
}

ALL_MODELS = list(MODEL_DISPLAY.keys())


# ─── Paths ────────────────────────────────────────────────────────────────────

def _mdir(model_name: str) -> str:
    return model_name.replace(".", "-").replace("/", "-")


def get_result_path(model_name: str, context_length: int, test_name: str) -> Path:
    return (
        SPECIAL_ROOT
        / f"results_{_mdir(model_name)}-{EXPERIMENT_TAG}"
        / REASONING_TYPE
        / f"rand_shuffle_{context_length}"
        / test_name
        / f"{_mdir(model_name)}_rand_book_{BOOK}_{test_name}.json"
    )


def result_exists(model_name: str, context_length: int, test_name: str) -> bool:
    path = get_result_path(model_name, context_length, test_name)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        results = data.get("results", [])
        if len(results) < NUM_DEPTHS:
            return False
        error_count = sum(
            1 for r in results
            if r.get("error") or r.get("error_type")
            or (r.get("response") is None and r.get("input_tokens", 0) == 0)
        )
        return error_count <= MAX_ALLOWED_ERRORS
    except Exception:
        return False


# ─── Test Loading ─────────────────────────────────────────────────────────────

def load_tests() -> List[Dict]:
    """Load T01 test configurations with answer-only prompts."""
    with open(NEEDLE_SET_PATH) as f:
        needle_set = json.load(f)

    tests = []
    for exp_config in needle_set:
        if exp_config.get("reasoning_type") != REASONING_TYPE:
            continue
        exp_id = exp_config["id"]
        if exp_id not in ALLOWED_NEEDLE_IDS:
            continue

        fixed_char = NEEDLE_CHARACTER.get(exp_id)
        limited_chars = [fixed_char] if fixed_char else exp_config.get("character_set", [])[:1]

        for question_type, question in exp_config["questions"].items():
            test_count = 0
            for test_id, test_cfg in exp_config["tests"].items():
                if test_count >= MAX_TESTS_PER_NEEDLE:
                    break
                test_count += 1

                full_needle   = exp_config["needle"]
                full_question = copy(question)
                for arg_no, arg in enumerate(test_cfg["input_args"]):
                    ph = "{" + str(arg_no + 1) + "}"
                    full_needle   = full_needle.replace(ph, arg)
                    full_question = full_question.replace(ph, arg)

                tests.append({
                    "test_name":          f"{exp_id}_{test_id}_{question_type}",
                    "system_prompt":      ANSWER_ONLY_SYSTEM_PROMPT,
                    "task_template":      ANSWER_ONLY_TASK_TEMPLATE,
                    "needle":             full_needle,
                    "retrieval_question": full_question,
                    "character_set":      limited_chars,
                    "seed":               BASE_SEED + int(exp_id[:4]),
                })
    return tests


# ─── Single-Test Runner ───────────────────────────────────────────────────────

async def run_single_test(
    config: ModelConfig,
    test: Dict,
    context_length: int,
    haystack_lines: List[str],
    haystack_hash: str,
    prefix: str = "",
) -> Dict:
    """Run all 4 depths for one (model, context, test) combination."""
    np.random.seed(test["seed"] + BOOK)
    result_path = get_result_path(config.name, context_length, test["test_name"])

    outputs = {
        "eval_name":    f"{_mdir(config.name)}_rand_book_{BOOK}_{test['test_name']}",
        "test_name":    test["test_name"],
        "model_name":   config.name,
        "retrieval_question": test["retrieval_question"],
        "needle":       test["needle"],
        "gold_answers": "",
        "system_prompt":           test["system_prompt"],
        "use_default_system_prompt": False,
        "task_template": test["task_template"],
        "haystack_path": str(HAYSTACK_DIR / f"rand_shuffle_{context_length}" / f"rand_book_{BOOK}.txt"),
        "context_length": context_length,
        "character_set": test["character_set"],
        "document_depth_percent_min":       DOCUMENT_DEPTH_PERCENT_MIN,
        "document_depth_percent_max":       DOCUMENT_DEPTH_PERCENT_MAX,
        "document_depth_percent_intervals": NUM_DEPTHS,
        "shift": 0, "static_depth": -1, "metric": "contains",
        "result_dir": str(result_path.parent),
        "seed": test["seed"] + BOOK,
        "experiment_tag": EXPERIMENT_TAG,
        "results": [],
        "test_hash": hashlib.sha256(
            f"{test['test_name']}_{context_length}_{BOOK}_{test['seed']}".encode()
        ).hexdigest(),
    }

    depths = np.linspace(DOCUMENT_DEPTH_PERCENT_MIN, DOCUMENT_DEPTH_PERCENT_MAX, NUM_DEPTHS) / 100
    prompts  = []
    metadata = []

    for depth in depths:
        if "{CHAR}" in test["needle"]:
            char     = test["character_set"][0]
            needle   = test["needle"].replace("{CHAR}", char)
            question = test["retrieval_question"].replace("{CHAR}", char)
        else:
            char     = ""
            needle   = test["needle"]
            question = test["retrieval_question"]

        haystack_with_needle, needle_pos = insert_needle(haystack_lines, needle, depth)
        formatted = format_haystack_with_line_numbers(haystack_with_needle)
        user_prompt = test["task_template"].format(haystack=formatted, question=question)

        prompts.append((test["system_prompt"], user_prompt))
        metadata.append({
            "selected_character": char,
            "needle": needle,
            "needle_position": needle_pos,
            "depth": depth,
            "num_lines": len(haystack_with_needle),
            "context_length_w_filled_template": len(test["system_prompt"]) + len(user_prompt),
        })

    # Call API in batches
    all_responses: List = [None] * len(prompts)
    for i in range(0, len(prompts), config.batch_size):
        end = min(i + config.batch_size, len(prompts))
        responses = await asyncio.gather(
            *[call_model(config, sp, up) for sp, up in prompts[i:end]],
            return_exceptions=True,
        )
        for j, resp in enumerate(responses):
            all_responses[i + j] = resp
        if end < len(prompts):
            await asyncio.sleep(config.batch_pause)

    # Process responses
    error_count = 0
    total_in_tokens = total_out_tokens = 0

    for resp, meta in zip(all_responses, metadata):
        depth_pct = f"{meta['depth'] * 100:.0f}%"

        if isinstance(resp, Exception):
            error_count += 1
            print(f"{prefix}    ✗ depth={depth_pct}  ERROR: {str(resp)[:100]}")
            continue

        raw = resp.get("response", "")
        if not raw or not raw.strip():
            error_count += 1
            print(f"{prefix}    ✗ depth={depth_pct}  EMPTY RESPONSE")
            continue

        parsed = parse_json_response(raw)
        if not parsed:
            error_count += 1
            print(f"{prefix}    ✗ depth={depth_pct}  UNPARSEABLE: {raw[:80]}")
            continue

        ans_metric, evi_metric = evaluate_response(
            parsed, meta["selected_character"], meta["needle_position"], meta["needle"]
        )
        outputs["results"].append({
            "selected_character": meta["selected_character"],
            "context_length_w_filled_templated": meta["context_length_w_filled_template"],
            "placement_metadata": {
                "needle":                    meta["needle"],
                "needle_line_num":           meta["needle_position"],
                "depth":                     meta["depth"],
                "num_haystack_lines_w_needle": meta["num_lines"],
            },
            "context_length_w_filled_template": meta["context_length_w_filled_template"],
            "response":        parsed,
            "answer_metric":   ans_metric,
            "evidence_metric": evi_metric,
            "input_tokens":    resp["input_tokens"],
            "output_tokens":   resp.get("output_tokens", 0),
        })
        total_in_tokens  += resp["input_tokens"]
        total_out_tokens += resp.get("output_tokens", 0)

        icon = "✓" if ans_metric else "✗"
        print(f"{prefix}    {icon}  depth={depth_pct}  char={meta['selected_character']}"
              f"  answer={str(parsed.get('answer', ''))[:40]!r}")

    if not outputs["results"]:
        print(f"{prefix}  ⚠ ALL depths failed — result NOT saved")
        return outputs

    outputs["summary"] = {
        "total_depths":       len(metadata),
        "error_count":        error_count,
        "success_count":      len(outputs["results"]),
        "total_input_tokens": total_in_tokens,
        "total_output_tokens": total_out_tokens,
        "timestamp":          time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(outputs, indent=2))
    if error_count:
        print(f"{prefix}  Saved {len(outputs['results'])}/{len(metadata)} depths "
              f"({error_count} failed)")
    return outputs


# ─── Per-Model Runner ─────────────────────────────────────────────────────────

async def run_model(
    model_name: str,
    tests: List[Dict],
    context_lengths: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict:
    """Run the answer-only experiment for one model."""
    config   = MODELS[model_name]
    label    = MODEL_DISPLAY.get(model_name, model_name)
    prefix   = f"[{label}] "
    all_ctx  = context_lengths or CONTEXT_LENGTHS
    ctx_list = [c for c in all_ctx if c <= config.max_context_chars]
    skipped  = [c for c in all_ctx if c > config.max_context_chars]
    if skipped:
        print(f"{prefix}Skipping {len(skipped)} contexts above max "
              f"({config.max_context_chars:,} chars)")

    # Inventory pending work
    pending: List[tuple] = []
    completed = 0
    for ctx in ctx_list:
        for test in tests:
            if result_exists(model_name, ctx, test["test_name"]):
                completed += 1
            else:
                pending.append((ctx, test))

    total = completed + len(pending)
    print(f"{prefix}{completed}/{total} complete, {len(pending)} pending")

    if dry_run:
        for ctx, test in pending[:5]:
            ctx_k = f"{ctx // 1000}K" if ctx < 1_000_000 else f"{ctx // 1_000_000}M"
            print(f"{prefix}  would run: {ctx_k} / {test['test_name']}")
        if len(pending) > 5:
            print(f"{prefix}  ... and {len(pending) - 5} more")
        return {"model": model_name, "completed": completed, "pending": len(pending), "errors": 0}

    if not pending:
        print(f"{prefix}All tests complete — nothing to do.")
        return {"model": model_name, "completed": completed, "pending": 0, "errors": 0}

    # Group by context for efficient haystack loading
    by_ctx: Dict[int, List[Dict]] = {}
    for ctx, test in pending:
        by_ctx.setdefault(ctx, []).append(test)

    errors: List[str] = []
    pbar = tqdm(total=len(pending), desc=label, leave=True)

    for context_length in sorted(by_ctx):
        ctx_label = (f"{context_length // 1000}K"
                     if context_length < 1_000_000
                     else f"{context_length // 1_000_000}M")
        print(f"\n{prefix}{'─' * 50}")
        print(f"{prefix}{ctx_label} | {len(by_ctx[context_length])} tests")
        print(f"{prefix}{'─' * 50}")

        try:
            haystack_lines, haystack_hash = load_haystack(context_length, BOOK)
        except FileNotFoundError:
            print(f"{prefix}⚠ Haystack not found for ctx={context_length}, book={BOOK}")
            pbar.update(len(by_ctx[context_length]))
            continue

        for test in by_ctx[context_length]:
            print(f"{prefix}Test: {test['test_name']}")
            try:
                await run_single_test(
                    config, test, context_length,
                    haystack_lines, haystack_hash,
                    prefix=prefix,
                )
            except Exception as e:
                err = str(e)[:80]
                errors.append(f"{ctx_label}/{test['test_name']}: {err}")
                print(f"{prefix}✗ FAILED: {err}")

            pbar.update(1)

            # Rate limiting
            calls     = NUM_DEPTHS
            min_time  = (calls / config.rpm_limit) * 60
            if config.tpm_limit > 0:
                est_tok  = context_length // CHARS_PER_TOKEN
                min_time = max(min_time, (calls * (est_tok + config.max_tokens) / config.tpm_limit) * 60)
            batches  = (calls + config.batch_size - 1) // config.batch_size
            extra    = max(0.0, min_time - (batches - 1) * config.batch_pause)
            if extra > 0:
                await asyncio.sleep(extra)

    pbar.close()
    if errors:
        print(f"\n{prefix}Errors ({len(errors)}):")
        for e in errors[:5]:
            print(f"{prefix}  {e}")
        if len(errors) > 5:
            print(f"{prefix}  ... and {len(errors) - 5} more")

    return {
        "model":     model_name,
        "completed": completed + len(pending) - len(errors),
        "pending":   0,
        "errors":    len(errors),
    }


# ─── Parallel Provider Orchestration ─────────────────────────────────────────

async def run_all(
    models_to_run: List[str],
    context_lengths: Optional[List[int]] = None,
    dry_run: bool = False,
) -> None:
    tests = load_tests()
    print(f"  Tests per model: {len(tests)} (T01 × 4 needles × 2 hops)")

    # Group by provider
    provider_groups: Dict[str, List[str]] = {}
    for m in models_to_run:
        provider_groups.setdefault(MODELS[m].provider, []).append(m)

    print("\n  Execution plan (providers run in parallel, models within provider sequential):")
    for provider, models in sorted(provider_groups.items()):
        names = [MODEL_DISPLAY.get(m, m) for m in models]
        print(f"    {provider:12s}: {names}")

    # Each provider group runs as one async task; within a group, models are sequential
    async def run_provider(provider: str, models: List[str]) -> List[Dict]:
        results = []
        for model in models:
            r = await run_model(model, tests, context_lengths, dry_run)
            results.append(r)
        return results

    all_results = await asyncio.gather(
        *[run_provider(p, ms) for p, ms in sorted(provider_groups.items())]
    )

    flat = [r for group in all_results for r in group]

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    total_done = total_err = 0
    for r in sorted(flat, key=lambda x: MODEL_DISPLAY.get(x["model"], x["model"])):
        name  = MODEL_DISPLAY.get(r["model"], r["model"])
        done  = r["completed"]
        errs  = r.get("errors", 0)
        pend  = r.get("pending", 0)
        total_done += done
        total_err  += errs
        status = "✓" if errs == 0 and pend == 0 else ("⚠" if errs else "…")
        print(f"  {status}  {name:<25}  done={done:4d}  errors={errs:3d}  pending={pend:4d}")
    print(f"\n  Total completed: {total_done}  |  Total errors: {total_err}")
    print("=" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NoLiMa Answer-Only Experiment — All Models"
    )
    parser.add_argument(
        "--model", type=str,
        help=f"Run a single model. Choices: {list(MODEL_DISPLAY.keys())}",
    )
    parser.add_argument(
        "--context-length", type=int,
        help="Run a specific context length only",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show pending work without making any API calls",
    )
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else ALL_MODELS

    # Validate model names
    for m in models_to_run:
        if m not in MODELS:
            print(f"ERROR: Unknown model '{m}'.")
            print(f"       Available: {list(MODELS.keys())}")
            sys.exit(1)

    # Check required API keys
    if not args.dry_run:
        needed = {MODELS[m].provider for m in models_to_run}
        missing = []
        if "anthropic" in needed and not os.getenv("ANTHROPIC_API_KEY"):
            missing.append("ANTHROPIC_API_KEY")
        if "openai"    in needed and not os.getenv("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")
        if "google"    in needed and not os.getenv("GOOGLE_API_KEY"):
            missing.append("GOOGLE_API_KEY")
        if missing:
            print(f"Error: Missing API key(s): {', '.join(missing)}")
            sys.exit(1)

    context_lengths = [args.context_length] if args.context_length else None

    print("=" * 60)
    print("  NoLiMa Answer-Only — All Models")
    print("=" * 60)
    print(f"  Models:   {[MODEL_DISPLAY.get(m, m) for m in models_to_run]}")
    print(f"  Book:     {BOOK} only")
    print(f"  Depths:   {NUM_DEPTHS}  (0%, 33%, 67%, 100%)")
    print(f"  Prompt:   Answer-only (no evidence/line citations)")
    print(f"  Output:   {SPECIAL_ROOT}/results_{{model}}-{EXPERIMENT_TAG}/")
    if args.dry_run:
        print("  Mode:     DRY RUN")

    asyncio.run(run_all(models_to_run, context_lengths, args.dry_run))


if __name__ == "__main__":
    main()
