#!/usr/bin/env python3
"""
NoLiMa Character Fix — In-Place Merge
=======================================

Checks existing result files in results_{model}/ for book 1, T01 tests,
at 4 target depths (0%, 33%, 67%, 100%).  If the correct character
(Yuki for 0402/0402Inv, Stuart for 0405/0405Inv) is NOT present at those
depths, re-runs only those depths and merges them into the existing JSON.

Existing depth entries (e.g., from 26-depth full-scale runs) are preserved.
Only the 4 target depth entries are replaced/added.

Usage:
  python run_controlled_standard.py --dry-run             # Show what would be fixed
  python run_controlled_standard.py                       # Fix all models
  python run_controlled_standard.py --model gpt-4o        # Fix specific model
  python run_controlled_standard.py --parallel            # Fix providers in parallel
"""

import os
import sys
import json
import asyncio
import argparse
import hashlib
import re
import time
import tempfile
from pathlib import Path
from copy import copy
from typing import List, Dict, Tuple, Optional, Set
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
    ALLOWED_NEEDLE_IDS,
    REASONING_TYPE,
    BASE_SEED,
    CHARS_PER_TOKEN,
    DATA_DIR,
    NEEDLE_SET_PATH,
    HAYSTACK_DIR,
    RESULTS_DIR,
    DOCUMENT_DEPTH_PERCENT_MIN,
    DOCUMENT_DEPTH_PERCENT_MAX,
    DEFAULT_BOOK,
    MAX_TESTS_PER_NEEDLE,
    load_haystack,
    insert_needle,
    format_haystack_with_line_numbers,
    call_model,
    parse_json_response,
    evaluate_response,
    get_result_path,
)

# ============ Configuration ============

NEEDLE_CHARACTER = {
    "0402": "Yuki",
    "0402Inv": "Yuki",
    "0405": "Stuart",
    "0405Inv": "Stuart",
}

# The 4 target depths we care about (as fractions)
TARGET_DEPTHS = np.linspace(
    DOCUMENT_DEPTH_PERCENT_MIN, DOCUMENT_DEPTH_PERCENT_MAX, 4
) / 100  # [0.0, 0.3333, 0.6667, 1.0]

DEPTH_TOLERANCE = 0.005  # for matching existing depth entries


# ============ Helpers ============

def get_needle_id_from_test_name(test_name: str) -> Optional[str]:
    for nid in ["0402Inv", "0405Inv", "0402", "0405"]:
        if test_name.startswith(nid + "_"):
            return nid
    return None


def _load_tests_fixed_char() -> List[Dict]:
    """Load T01 tests with fixed characters (Yuki/Stuart)."""
    with open(NEEDLE_SET_PATH, "r") as f:
        needle_set = json.load(f)

    tests = []
    for exp_config in needle_set:
        if exp_config.get("reasoning_type") != REASONING_TYPE:
            continue
        exp_id = exp_config["id"]
        if exp_id not in ALLOWED_NEEDLE_IDS:
            continue

        system_prompt = exp_config["system_prompt"]
        task_template = exp_config.get("task_template", "")
        fixed_char = NEEDLE_CHARACTER.get(exp_id)
        limited_character_set = [fixed_char] if fixed_char else exp_config.get("character_set", [])[:1]

        for question_type, question in exp_config["questions"].items():
            test_count = 0
            for test_id, test_cfg in exp_config["tests"].items():
                if MAX_TESTS_PER_NEEDLE and test_count >= MAX_TESTS_PER_NEEDLE:
                    break
                test_count += 1

                full_needle = exp_config["needle"]
                full_question = copy(question)
                for arg_no, arg in enumerate(test_cfg["input_args"]):
                    placeholder = "{" + str(arg_no + 1) + "}"
                    full_needle = full_needle.replace(placeholder, arg)
                    full_question = full_question.replace(placeholder, arg)

                tests.append({
                    "test_name": f"{exp_id}_{test_id}_{question_type}",
                    "system_prompt": system_prompt,
                    "task_template": task_template,
                    "needle": full_needle,
                    "retrieval_question": full_question,
                    "character_set": limited_character_set,
                    "seed": BASE_SEED + int(exp_id[:4]),
                })

    return tests


def _get_test_by_name(test_name: str) -> Optional[Dict]:
    """Look up a test by name, with fallback to loading all test variants."""
    for t in _load_tests_fixed_char():
        if t["test_name"] == test_name:
            return t

    # Fallback: load without MAX_TESTS_PER_NEEDLE limit
    with open(NEEDLE_SET_PATH, "r") as f:
        needle_set = json.load(f)

    for exp_config in needle_set:
        if exp_config.get("reasoning_type") != REASONING_TYPE:
            continue
        exp_id = exp_config["id"]
        if exp_id not in ALLOWED_NEEDLE_IDS:
            continue

        system_prompt = exp_config["system_prompt"]
        task_template = exp_config.get("task_template", "")
        fixed_char = NEEDLE_CHARACTER.get(exp_id)
        char_set = [fixed_char] if fixed_char else exp_config.get("character_set", [])[:1]

        for question_type, question in exp_config["questions"].items():
            for test_id, test_cfg in exp_config["tests"].items():
                candidate = f"{exp_id}_{test_id}_{question_type}"
                if candidate != test_name:
                    continue

                full_needle = exp_config["needle"]
                full_question = copy(question)
                for arg_no, arg in enumerate(test_cfg["input_args"]):
                    placeholder = "{" + str(arg_no + 1) + "}"
                    full_needle = full_needle.replace(placeholder, arg)
                    full_question = full_question.replace(placeholder, arg)

                return {
                    "test_name": candidate,
                    "system_prompt": system_prompt,
                    "task_template": task_template,
                    "needle": full_needle,
                    "retrieval_question": full_question,
                    "character_set": char_set,
                    "seed": BASE_SEED + int(exp_id[:4]),
                }
    return None


# ============ Check which depths need fixing ============

def check_file_depths(filepath: str, expected_char: str) -> List[float]:
    """Check an existing result file for our 4 target depths.

    Returns list of target depths that are MISSING or have the WRONG character.
    """
    depths_needing_fix = []

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return list(TARGET_DEPTHS)

    results = data.get("results", [])

    # Build map: depth_value -> selected_character
    depth_char_map = {}
    for r in results:
        d = r.get("placement_metadata", {}).get("depth")
        c = r.get("selected_character", "")
        if d is not None:
            depth_char_map[round(d, 6)] = c

    for target in TARGET_DEPTHS:
        target_r = round(target, 6)

        # Find exact or closest match within tolerance
        matched = False
        for existing_depth, char in depth_char_map.items():
            if abs(existing_depth - target_r) < DEPTH_TOLERANCE:
                if char == expected_char:
                    matched = True
                break

        if not matched:
            depths_needing_fix.append(target)

    return depths_needing_fix


def scan_model(model_name: str) -> Tuple[List[Dict], List[Dict]]:
    """Scan book-1 T01 results for a model.

    Returns (ok_list, needs_fix_list) where each item is a dict with:
      filepath, test_name, context_length, needle_id, expected_char, depths_to_fix
    """
    config = MODELS[model_name]
    context_lengths = [c for c in CONTEXT_LENGTHS if c <= config.max_context_chars]
    tests = _load_tests_fixed_char()

    ok_list = []
    fix_list = []

    for ctx in context_lengths:
        for test in tests:
            needle_id = get_needle_id_from_test_name(test["test_name"])
            if not needle_id:
                continue
            expected_char = NEEDLE_CHARACTER[needle_id]

            filepath = get_result_path(model_name, ctx, test["test_name"], DEFAULT_BOOK)

            if filepath.exists():
                depths_to_fix = check_file_depths(str(filepath), expected_char)
            else:
                depths_to_fix = list(TARGET_DEPTHS)

            entry = {
                "filepath": str(filepath),
                "test_name": test["test_name"],
                "context_length": ctx,
                "needle_id": needle_id,
                "expected_char": expected_char,
                "depths_to_fix": depths_to_fix,
                "file_exists": filepath.exists(),
            }

            if len(depths_to_fix) == 0:
                ok_list.append(entry)
            else:
                fix_list.append(entry)

    return ok_list, fix_list


# ============ Run & Merge ============

async def run_and_merge(
    config: ModelConfig,
    entry: Dict,
    haystack_lines: List[str],
    haystack_hash: str,
) -> bool:
    """Run the missing depths with the correct character and merge into the file.

    Returns True on success.
    """
    test = _get_test_by_name(entry["test_name"])
    if test is None:
        raise ValueError(f"Test config not found: {entry['test_name']}")

    context_length = entry["context_length"]
    book_num = DEFAULT_BOOK
    result_path = Path(entry["filepath"])
    depths_to_run = entry["depths_to_fix"]

    np.random.seed(test["seed"] + book_num)

    # Resolve character
    selected_character = test["character_set"][0]
    needle = test["needle"].replace("{CHAR}", selected_character) if "{CHAR}" in test["needle"] else test["needle"]
    question = test["retrieval_question"].replace("{CHAR}", selected_character) if "{CHAR}" in test["retrieval_question"] else test["retrieval_question"]

    # Build prompts for the depths we need
    prompts = []
    metadata = []

    for depth in depths_to_run:
        depth_native = float(depth)  # convert numpy.float64 → Python float
        haystack_with_needle, needle_position = insert_needle(
            haystack_lines, needle, depth_native
        )
        formatted = format_haystack_with_line_numbers(haystack_with_needle)
        user_prompt = test["task_template"].format(haystack=formatted, question=question)
        filled_len = len(test["system_prompt"]) + len(user_prompt)

        prompts.append((test["system_prompt"], user_prompt))
        metadata.append({
            "selected_character": selected_character,
            "needle": needle,
            "needle_position": needle_position,
            "depth": depth_native,
            "num_lines": len(haystack_with_needle),
            "context_length_w_filled_template": filled_len,
        })

    # Call API
    async def process_batch(batch_prompts):
        tasks = [call_model(config, sp, up) for sp, up in batch_prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    all_responses = [None] * len(prompts)
    for i in range(0, len(prompts), config.batch_size):
        batch_end = min(i + config.batch_size, len(prompts))
        responses = await process_batch(prompts[i:batch_end])
        for j, resp in enumerate(responses):
            all_responses[i + j] = resp
        if batch_end < len(prompts):
            await asyncio.sleep(config.batch_pause)

    # Process responses into result entries
    new_entries = []
    for resp, meta in zip(all_responses, metadata):
        depth_pct = f"{meta['depth']*100:.0f}%"
        if isinstance(resp, Exception):
            print(f"    ✗ depth={depth_pct} char={meta['selected_character']}  ERROR: {str(resp)[:120]}")
            continue
        raw_text = resp.get("response", "")
        if not raw_text or raw_text.strip() == "":
            print(f"    ✗ depth={depth_pct} char={meta['selected_character']}  EMPTY RESPONSE (skipped)")
            continue
        parsed = parse_json_response(raw_text)
        if not parsed:
            print(f"    ✗ depth={depth_pct} char={meta['selected_character']}  UNPARSEABLE: {raw_text[:100]}")
            continue

        answer_metric, evidence_metric = evaluate_response(
            parsed, meta["selected_character"], meta["needle_position"], meta["needle"]
        )

        new_entries.append({
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
        })

    if not new_entries:
        return False

    # ── Merge into existing file ─────────────────────────────────────
    if result_path.exists():
        try:
            with open(result_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {}
    else:
        # Create new file structure matching run_full_scale format
        test_hash = hashlib.sha256(
            f"{test['test_name']}_{context_length}_{book_num}_{test['seed']}".encode()
        ).hexdigest()
        data = {
            "eval_name": f"{config.name}_rand_book_{book_num}_{test['test_name']}",
            "test_name": test["test_name"],
            "model_name": config.name,
            "retrieval_question": question,
            "needle": needle,
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
            "document_depth_percent_intervals": 4,
            "shift": 0,
            "static_depth": -1,
            "metric": "contains",
            "result_dir": str(result_path.parent),
            "seed": test["seed"] + book_num,
            "results": [],
            "test_hash": test_hash,
        }

    existing_results = data.get("results", [])

    # Build set of new depth values (rounded for comparison)
    new_depth_set = set()
    for e in new_entries:
        new_depth_set.add(round(e["placement_metadata"]["depth"], 6))

    # Remove existing entries at the depths we're replacing
    kept = []
    for r in existing_results:
        d = round(r.get("placement_metadata", {}).get("depth", -1), 6)
        # Keep if it's NOT one of the depths we're replacing
        if not any(abs(d - nd) < DEPTH_TOLERANCE for nd in new_depth_set):
            kept.append(r)

    # Add new entries and sort by depth
    merged = kept + new_entries
    merged.sort(key=lambda r: r.get("placement_metadata", {}).get("depth", 0))

    data["results"] = merged

    # Update summary
    total_input = sum(r.get("input_tokens", 0) for r in merged)
    total_output = sum(r.get("output_tokens", 0) for r in merged)
    data["summary"] = {
        "total_depths": len(merged),
        "error_count": 0,
        "success_count": len(merged),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save atomically: write to temp file then rename to prevent corruption on interrupt
    result_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(result_path.parent), suffix=".tmp", prefix=".merge_"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, str(result_path))  # atomic on same filesystem
    except BaseException:
        # Clean up temp file on any error (including KeyboardInterrupt)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return True


# ============ Main Runner ============

async def run_model_fixes(
    model_name: str,
    context_lengths: List[int] = None,
    dry_run: bool = False,
) -> Dict:
    if model_name not in MODELS:
        print(f"Error: Unknown model {model_name}")
        return {}

    config = MODELS[model_name]
    ok_list, fix_list = scan_model(model_name)

    if context_lengths:
        fix_list = [e for e in fix_list if e["context_length"] in context_lengths]

    total_api_calls = sum(len(e["depths_to_fix"]) for e in fix_list)

    print(f"\n{'=' * 60}")
    print(f"Character Fix: {model_name}")
    print(f"{'=' * 60}")
    print(f"Book 1, T01 only, 4 target depths (0%, 33%, 67%, 100%)")
    print(f"Already correct:    {len(ok_list)}")
    print(f"Need fix/create:    {len(fix_list)}")
    print(f"API calls needed:   {total_api_calls}")

    if dry_run:
        print("\nDry run — would fix:")
        for e in fix_list[:20]:
            ctx = e["context_length"]
            n = len(e["depths_to_fix"])
            tag = "FIX" if e["file_exists"] else "NEW"
            depths_str = ", ".join(f"{d*100:.0f}%" for d in e["depths_to_fix"])
            print(f"  [{tag}] {ctx:>7,} / {e['test_name']:<30s} → {n} depths ({depths_str})")
        if len(fix_list) > 20:
            print(f"  ... and {len(fix_list) - 20} more")
        return {"ok": len(ok_list), "to_fix": len(fix_list),
                "api_calls": total_api_calls}

    if not fix_list:
        print("All book-1 T01 results already have correct characters!")
        return {"ok": len(ok_list), "fixed": 0, "errors": 0}

    # Group by context_length for efficient haystack loading
    by_ctx = {}
    for entry in fix_list:
        ctx = entry["context_length"]
        if ctx not in by_ctx:
            by_ctx[ctx] = []
        by_ctx[ctx].append(entry)

    pbar = tqdm(total=len(fix_list), desc=f"{model_name} (char fix)")
    errors = []

    for context_length, entries in by_ctx.items():
        try:
            haystack_lines, haystack_hash = load_haystack(context_length, DEFAULT_BOOK)
        except FileNotFoundError:
            print(f"  Haystack not found: {context_length}")
            pbar.update(len(entries))
            continue

        for entry in entries:
            try:
                ok = await run_and_merge(config, entry, haystack_lines, haystack_hash)
                if not ok:
                    errors.append(f"{context_length}/{entry['test_name']}: all API calls failed")
            except Exception as e:
                errors.append(f"{context_length}/{entry['test_name']}: {str(e)[:80]}")

            pbar.update(1)

            # Rate limiting
            calls = len(entry["depths_to_fix"])
            min_time_rpm = (calls / config.rpm_limit) * 60
            min_time = min_time_rpm
            if config.tpm_limit and config.tpm_limit > 0:
                est_input = context_length // CHARS_PER_TOKEN
                est_tokens = calls * (est_input + config.max_tokens)
                min_time_tpm = (est_tokens / config.tpm_limit) * 60
                min_time = max(min_time, min_time_tpm)
            batches = (calls + config.batch_size - 1) // config.batch_size
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

    return {"ok": len(ok_list), "fixed": len(fix_list) - len(errors), "errors": len(errors)}


async def run_provider_models(provider, models, context_lengths, dry_run):
    if provider == "google" and len(models) > 1:
        tasks = [run_model_fixes(m, context_lengths, dry_run) for m in models]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        results = {}
        for model, res in zip(models, results_list):
            results[model] = {"errors": 1} if isinstance(res, Exception) else res
        return results
    results = {}
    for model in models:
        results[model] = await run_model_fixes(model, context_lengths, dry_run)
    return results


async def run_all_models(models=None, context_lengths=None, dry_run=False, parallel=False):
    models = models or list(MODELS.keys())

    print("=" * 60)
    print("NoLiMa Character Fix — In-Place Merge")
    print("=" * 60)
    print(f"Models:     {models}")
    print(f"Scope:      Book 1, T01, 4 depths (0%, 33%, 67%, 100%)")
    print(f"Characters: Yuki (0402*), Stuart (0405*)")
    print(f"Action:     Check & merge correct chars into existing results")
    print(f"Parallel:   {parallel}")

    if parallel:
        by_provider = {}
        for model in models:
            if model in MODELS:
                p = MODELS[model].provider
                by_provider.setdefault(p, []).append(model)

        print(f"\nRunning {len(by_provider)} providers in parallel:")
        for p, ms in by_provider.items():
            print(f"  {p}: {ms}")

        tasks = [run_provider_models(p, ms, context_lengths, dry_run)
                 for p, ms in by_provider.items()]
        provider_results = await asyncio.gather(*tasks, return_exceptions=True)
        results = {}
        for i, (p, _) in enumerate(by_provider.items()):
            if isinstance(provider_results[i], Exception):
                print(f"\nError running {p}: {provider_results[i]}")
            else:
                results.update(provider_results[i])
    else:
        results = {}
        for model in models:
            results[model] = await run_model_fixes(model, context_lengths, dry_run)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model, stats in results.items():
        parts = [f"{k}={v}" for k, v in stats.items()]
        print(f"  {model}: {', '.join(parts)}")


# ============ CLI ============

def main():
    parser = argparse.ArgumentParser(
        description="NoLiMa Character Fix — merge Yuki/Stuart into existing results"
    )
    parser.add_argument("--model", help="Fix specific model only")
    parser.add_argument("--context-length", type=int, help="Fix specific context length only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--parallel", action="store_true",
                        help="Fix different providers in parallel")
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
