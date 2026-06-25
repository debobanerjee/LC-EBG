#!/usr/bin/env python3
"""
NoLiMa Long Context Testing (Lite Mode)

Runs needle-in-haystack evaluations for multiple LLMs at reduced scale.
Lite defaults: 1 book, 4 depths, 1 test per needle, 1 character.
Old full-scale results (26 depths, 5 books) are still recognized as complete.

Features:
- Checks existing results and skips completed tests
- Respects API rate limits with adaptive batching
- Automatic retry with exponential backoff
- Progress tracking and checkpointing
- Supports OpenAI, Anthropic, and Google models

Usage:
  python run_full_scale.py                          # Run all models (lite defaults)
  python run_full_scale.py --model gpt-4o           # Run specific model
  python run_full_scale.py --context-length 100000  # Run specific context length
  python run_full_scale.py --dry-run                # Show what would be run
  python run_full_scale.py --gemini-only            # Run only Gemini models (in parallel)
  python run_full_scale.py --character Yuki          # Force specific character
  python run_full_scale.py --book-num 3             # Use book 3 only
  python run_full_scale.py --num-depths 8           # Override depth count
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
from copy import copy
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============ Configuration ============
#
# Context limit conversion (tokens -> characters):
# - English text: ~4 characters per token (OpenAI, Anthropic, Google docs).
# - We use 0.85 safety factor to leave room for system prompt, task template,
#   line-number prefixes, and tokenizer variance. So: max_chars = tokens * 4 * 0.85.
#
# Official context windows (tokens): GPT-4o 128k, GPT-4.1 1M, o3-mini 200k,
# GPT-5 400k, Claude Sonnet 4/4.5 200k, Gemini 2.5 Flash 1M, Gemini 3 Flash 1M.

CHARS_PER_TOKEN = 4
CONTEXT_SAFETY_FACTOR = 0.99  # reserve 1% for system/template/overhead


def _context_tokens_to_max_chars(context_tokens: int, cap_chars: int = 1_000_000) -> int:
    """Convert model context window (tokens) to a safe max character limit."""
    return min(int(context_tokens * CHARS_PER_TOKEN * CONTEXT_SAFETY_FACTOR), cap_chars)


@dataclass
class ModelConfig:
    name: str
    provider: str  # "openai", "anthropic", "google"
    max_tokens: int = 2048
    batch_size: int = 20  # requests per batch
    batch_pause: float = 1.0  # seconds between batches
    rpm_limit: int = 60  # requests per minute limit
    tpm_limit: int = 0  # tokens per minute limit (0 = not enforced); used for large-context throttling
    max_context_chars: int = 1_000_000  # max input context in characters (from token limit)

MODELS = {
    # OpenAI: https://platform.openai.com/docs/models
    "gpt-4o": ModelConfig("gpt-4o", "openai", batch_size=20, rpm_limit=500,
                          max_context_chars=_context_tokens_to_max_chars(128_000)),
    "gpt-4.1": ModelConfig("gpt-4.1", "openai", batch_size=20, rpm_limit=500,
                            max_context_chars=_context_tokens_to_max_chars(1_000_000)),
    "o3-mini-2025-01-31": ModelConfig("o3-mini-2025-01-31", "openai", batch_size=20, rpm_limit=500,
                                      max_context_chars=_context_tokens_to_max_chars(200_000)),
    "gpt-5-2025-08-07": ModelConfig("gpt-5-2025-08-07", "openai", max_tokens=1024, batch_size=10, rpm_limit=100,
                                    max_context_chars=_context_tokens_to_max_chars(400_000)),
    # Anthropic: https://docs.anthropic.com/en/docs/build-with-claude/context-windows
    #"claude-3-7-sonnet-20250219": ModelConfig(..., max_context_chars=_context_tokens_to_max_chars(200_000)),
    "claude-sonnet-4-20250514": ModelConfig("claude-sonnet-4-20250514", "anthropic", batch_size=20, rpm_limit=50,
                                            max_context_chars=_context_tokens_to_max_chars(200_000)),
    "claude-sonnet-4-5-20250929": ModelConfig("claude-sonnet-4-5-20250929", "anthropic", batch_size=20, rpm_limit=50,
                                              max_context_chars=_context_tokens_to_max_chars(200_000)),
    # Google: Gemini 2.5 Flash — 60 RPM, 1M TPM
    "gemini-2.5-flash": ModelConfig("gemini-2.5-flash", "google", batch_size=3, rpm_limit=60, batch_pause=1.0,
                                    tpm_limit=1_000_000, max_context_chars=_context_tokens_to_max_chars(1_048_576)),
    # Google: Gemini 3 Flash — 1K RPM, 2M TPM, 10K RPD
    "gemini-3-flash-preview": ModelConfig("gemini-3-flash-preview", "google", batch_size=20, rpm_limit=1000, batch_pause=0.2,
                                          tpm_limit=2_000_000, max_context_chars=_context_tokens_to_max_chars(1_048_576)),
}

# Context lengths: 10k to 1000k
CONTEXT_LENGTHS = [
    10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000,
    150_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000,
    600_000, 650_000, 700_000, 750_000, 800_000, 850_000, 900_000, 950_000, 1_000_000
]
# [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 600000, 700000, 800000, 900000, 1000000]

AVAILABLE_BOOKS = 5    # Total book files available (rand_book_1.txt .. rand_book_5.txt)
NUM_BOOKS = 1          # 1 book per context length
NUM_DEPTHS = 4         # 4 depth positions (0%, 33%, 67%, 100%)
MAX_TESTS_PER_NEEDLE = 1  # 1 = only T01 test variant per needle
NUM_CHARACTERS = 1     # 1 character per needle (fixed: Yuki for 0402*, Stuart for 0405*)

# Fixed character assignment per needle type
NEEDLE_CHARACTER = {
    "0402": "Yuki",
    "0402Inv": "Yuki",
    "0405": "Stuart",
    "0405Inv": "Stuart",
}
DEFAULT_BOOK = 1       # Always use book 1 for consistency
DOCUMENT_DEPTH_PERCENT_MIN = 0
DOCUMENT_DEPTH_PERCENT_MAX = 100

BASE_SEED = 42
REASONING_TYPE = "commonsense_knowledge"
# Standardized path structure (no task version in path)

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
DATA_DIR = REPO_ROOT / "datasets" / "NoLiMa"
NEEDLE_SET_PATH = DATA_DIR / "needlesets" / "needle_set.json"
HAYSTACK_DIR = DATA_DIR / "haystacks"
RESULTS_DIR = PROJECT_DIR / "evaluation"

# ============ API Clients ============

_openai_client = None
_anthropic_client = None
_google_configured = False

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client

def get_google_model(model_name):
    global _google_configured
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import google.generativeai as genai
    if not _google_configured:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        _google_configured = True
    return genai.GenerativeModel(model_name), genai

# ============ API Call Functions ============

def is_retryable_error(e: Exception) -> bool:
    """
    Check if an error is retryable (rate limit, transient network/server issues).

    Important: quota / billing / resource exhaustion errors are treated as
    NON-retryable to avoid hammering an exhausted account. For example, Gemini
    errors like:

      "429 You exceeded your current quota, please check your plan and billing..."

    should be surfaced immediately instead of retried.
    """
    error_str = str(e).lower()

    # Explicitly non-retryable patterns (permanent failures)
    non_retryable_patterns = [
        "billing",              # "check your plan and billing details"
        "insufficient funds",
        "payment required",
        "access denied",        # permission issues
    ]
    if any(p in error_str for p in non_retryable_patterns):
        return False
    
    # Quota / resource exhaustion — retryable with backoff (often resolves in 30-60s)
    quota_patterns = ["quota", "resourceexhausted"]
    if any(p in error_str for p in quota_patterns):
        return True

    # Transient / retryable patterns: rate limits, timeouts, 5xx, etc.
    retryable_patterns = [
        "rate limit", "rate_limit",
        "429 too many requests",
        "503", "502", "504",
        "timeout", "timed out",
        "connection", "overloaded",
        "temporarily unavailable",
        "server error", "internal error",
    ]
    return any(p in error_str for p in retryable_patterns)

async def call_openai(config: ModelConfig, system_prompt: str, user_prompt: str) -> Dict:
    """Call OpenAI API with retry logic."""
    client = get_openai_client()
    
    kwargs = {
        "model": config.name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    
    # GPT-5 and o-series use max_completion_tokens
    if "gpt-5" in config.name or "o1" in config.name or "o3" in config.name or "o4" in config.name:
        kwargs["max_completion_tokens"] = config.max_tokens
    else:
        kwargs["max_tokens"] = config.max_tokens
    
    last_error = None
    for attempt in range(3):
        try:
            result = await client.chat.completions.create(**kwargs)
            return {
                "response": result.choices[0].message.content,
                "input_tokens": result.usage.prompt_tokens,
                "output_tokens": result.usage.completion_tokens
            }
        except Exception as e:
            last_error = e
            if attempt < 2 and is_retryable_error(e):
                wait_time = (attempt + 1) * 5
                await asyncio.sleep(wait_time)
            elif not is_retryable_error(e):
                # Non-retryable error, fail immediately
                raise
    raise last_error

async def call_anthropic(config: ModelConfig, system_prompt: str, user_prompt: str) -> Dict:
    """Call Anthropic API with retry logic."""
    client = get_anthropic_client()
    
    last_error = None
    for attempt in range(3):
        try:
            result = await client.messages.create(
                model=config.name,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=config.max_tokens
            )
            return {
                "response": result.content[0].text,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens
            }
        except Exception as e:
            last_error = e
            if attempt < 2 and is_retryable_error(e):
                wait_time = (attempt + 1) * 5
                await asyncio.sleep(wait_time)
            elif not is_retryable_error(e):
                raise
    raise last_error

async def call_google(config: ModelConfig, system_prompt: str, user_prompt: str) -> Dict:
    """Call Google Gemini API with robust retry logic.
    
    Retries up to 5 times with exponential backoff (10s, 20s, 40s, 60s, 60s)
    for rate-limit (429) and transient errors.  Non-retryable errors fail immediately.
    """
    model, genai = get_google_model(config.name)
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    loop = asyncio.get_event_loop()
    
    gen_config = genai.types.GenerationConfig(
        max_output_tokens=config.max_tokens,
        temperature=0.0,
    )
    
    def sync_generate():
        return model.generate_content(full_prompt, generation_config=gen_config)
    
    max_retries = 5
    last_error = None
    for attempt in range(max_retries):
        try:
            result = await loop.run_in_executor(None, sync_generate)
            
            # Extract response text safely
            response_text = ""
            try:
                if hasattr(result, 'text') and result.text:
                    response_text = result.text
                elif result.candidates:
                    candidate = result.candidates[0]
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason.name == "RECITATION":
                        raise ValueError("Response blocked due to RECITATION filter")
                    if candidate.content and candidate.content.parts:
                        response_text = candidate.content.parts[0].text
            except ValueError:
                raise
            except Exception:
                pass
            
            return {
                "response": response_text,
                "input_tokens": result.usage_metadata.prompt_token_count if hasattr(result, 'usage_metadata') else 0,
                "output_tokens": result.usage_metadata.candidates_token_count if hasattr(result, 'usage_metadata') else 0
            }
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1 and is_retryable_error(e):
                wait_time = min(60, 10 * (2 ** attempt))  # 10s, 20s, 40s, 60s
                print(f"      ⏳ Retry {attempt+1}/{max_retries-1} in {wait_time}s: {str(e)[:80]}")
                await asyncio.sleep(wait_time)
            elif not is_retryable_error(e):
                raise
    raise last_error

async def call_model(config: ModelConfig, system_prompt: str, user_prompt: str) -> Dict:
    """Call the appropriate API based on model provider."""
    if config.provider == "openai":
        return await call_openai(config, system_prompt, user_prompt)
    elif config.provider == "anthropic":
        return await call_anthropic(config, system_prompt, user_prompt)
    elif config.provider == "google":
        return await call_google(config, system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")

# ============ Haystack & Needle Functions ============

def load_haystack(context_length: int, book_num: int) -> Tuple[List[str], str]:
    """Load haystack file and return lines + hash."""
    filepath = HAYSTACK_DIR / f"rand_shuffle_{context_length}" / f"rand_book_{book_num}.txt"
    with open(filepath, 'r') as f:
        content = f.read()
    lines = content.strip().split('\n')
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    return lines, content_hash

def insert_needle(haystack_lines: List[str], needle: str, depth: float, shift: int = 0) -> Tuple[List[str], int]:
    """Insert needle at specified depth position."""
    position = int(len(haystack_lines) * depth) + shift
    position = max(0, min(position, len(haystack_lines)))
    result = haystack_lines.copy()
    result.insert(position, needle)
    return result, position

def format_haystack_with_line_numbers(lines: List[str]) -> str:
    """Format haystack with line numbers."""
    return '\n'.join(f"{i}: {line}" for i, line in enumerate(lines))

def _extract_brace_balanced(text: str, start: int) -> str:
    """Extract a {...} or [...] span from start using brace counting. Handles truncation."""
    if start >= len(text) or text[start] not in '{[':
        return ""
    open_b, close_b = ('{', '}') if text[start] == '{' else ('[', ']')
    depth = 0
    i = start
    in_string = None
    escape = False
    while i < len(text):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == in_string:
                in_string = None
            i += 1
            continue
        if c in '"\'':
            in_string = c
            i += 1
            continue
        if c == open_b:
            depth += 1
        elif c == close_b:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    return ""  # truncated: no closing brace

def parse_json_response(text: str) -> Dict:
    """Parse JSON from LLM response.

    Handles common LLM output quirks:
      - Markdown code fences (```json ... ```)
      - Truncated code blocks (no closing ```): extract object by brace balance
      - Trailing commas ({"answer": "X",})
      - Direct JSON objects
    """
    if not text:
        return None

    # Fix trailing commas before } or ] (common LLM quirk)
    cleaned = re.sub(r',\s*}', '}', text)
    cleaned = re.sub(r',\s*]', ']', cleaned)

    # Try code block (full or truncated): find ```json then brace-balanced {...}
    code_start = re.search(r'```(?:json)?\s*', cleaned, re.IGNORECASE)
    if code_start:
        start = code_start.end()
        if start < len(cleaned) and cleaned[start] == '{':
            obj = _extract_brace_balanced(cleaned, start)
            if obj:
                try:
                    return json.loads(obj)
                except (json.JSONDecodeError, ValueError):
                    pass
    # Full code block with closing ```
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Try brace-balanced object from first { (handles raw or truncated JSON)
    first_brace = cleaned.find('{')
    if first_brace != -1:
        obj = _extract_brace_balanced(cleaned, first_brace)
        if obj:
            try:
                return json.loads(obj)
            except (json.JSONDecodeError, ValueError):
                pass

    # Try direct JSON with "answer" key (simple, no nested braces)
    match = re.search(r'\{[^{}]*"answer"[^{}]*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    # Salvage truncated JSON: model cut off before closing (e.g. "answer" or "answer": "Yu")
    first_brace = cleaned.find('{')
    if first_brace != -1:
        tail = cleaned[first_brace:].rstrip()
        for suffix in ('": null}', '"}', '}'):
            try:
                return json.loads(tail + suffix)
            except (json.JSONDecodeError, ValueError):
                continue

    return None

def evaluate_response(parsed: Dict, expected_character: str, expected_line: int, needle: str) -> Tuple[int, int]:
    """Evaluate answer and evidence metrics."""
    if not parsed:
        return 0, 0
    
    # Answer metric
    answer = str(parsed.get("answer", "")).lower()
    answer_correct = expected_character.lower() in answer if expected_character else True
    
    # Evidence metric - normalize lines to integers for comparison
    lines = parsed.get("lines", [])
    if not isinstance(lines, list):
        lines = [lines] if lines is not None else []
    
    # Convert all line values to int for comparison
    int_lines = []
    for line in lines:
        try:
            int_lines.append(int(line))
        except (ValueError, TypeError):
            pass
    
    line_correct = expected_line in int_lines
    
    return int(answer_correct), int(line_correct)

# ============ Test Configuration ============

# Only use these needle IDs (commonsense_knowledge)
ALLOWED_NEEDLE_IDS = {"0402", "0402Inv", "0405", "0405Inv"}

def load_tests() -> List[Dict]:
    """Load test configurations from needle set.
    
    Filters to ALLOWED_NEEDLE_IDS only (0402, 0402Inv, 0405, 0405Inv).
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
        system_prompt = exp_config["system_prompt"]
        task_template = exp_config.get("task_template", "")
        character_set = exp_config.get("character_set", [])
        
        # Use fixed character for this needle type
        fixed_char = NEEDLE_CHARACTER.get(exp_id)
        if fixed_char:
            limited_character_set = [fixed_char]
        else:
            limited_character_set = character_set[:NUM_CHARACTERS] if NUM_CHARACTERS else character_set
        
        for question_type, question in exp_config["questions"].items():
            test_count = 0
            for test_id, test in exp_config["tests"].items():
                if MAX_TESTS_PER_NEEDLE and test_count >= MAX_TESTS_PER_NEEDLE:
                    break
                test_count += 1
                
                full_needle = exp_config["needle"]
                full_question = copy(question)
                
                # Fill in input args
                for arg_no, arg in enumerate(test["input_args"]):
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
                    "seed": BASE_SEED + int(exp_id[:4])
                })
    
    return tests

def get_result_path(model_name: str, context_length: int, test_name: str, book_num: int) -> Path:
    """Get the standardized result file path (no task version in path)."""
    # Normalize model name for directory
    model_dir = model_name.replace(".", "-").replace("/", "-")
    return (RESULTS_DIR / f"results_{model_dir}" / REASONING_TYPE / 
            f"rand_shuffle_{context_length}" / test_name / 
            f"{model_dir}_rand_book_{book_num}_{test_name}.json")

MAX_ALLOWED_ERRORS = 1  # Results with more than 1 error are considered incomplete (re-run)

def find_existing_result(model_name: str, context_length: int, test_name: str, book_num: int) -> Path:
    """Find existing result file with standardized path structure."""
    # Try multiple model name normalizations
    model_name_variants = [
        model_name,
        model_name.replace(".", "-").replace("/", "-"),
        model_name.replace("/", "-"),
    ]
    
    for model_variant in model_name_variants:
        # Check standardized results directory
        result_dir = RESULTS_DIR / f"results_{model_variant}" / REASONING_TYPE / f"rand_shuffle_{context_length}" / test_name
        
        if not result_dir.exists():
            continue
        
        # Look for any matching result file
        for filename in result_dir.iterdir():
            if filename.suffix == '.json' and f"_rand_book_{book_num}_" in filename.name:
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                    results = data.get("results", [])
                    
                    # Accept results with at least NUM_DEPTHS evaluations
                    # (allows old full-scale results with 26 depths to still pass)
                    if len(results) < NUM_DEPTHS:
                        continue
                    
                    # Check error count - treat high-error files as incomplete
                    error_count = sum(
                        1 for r in results
                        if r.get("error") or r.get("error_type")
                        or (r.get("response") is None and r.get("input_tokens", 0) == 0)
                    )
                    if error_count > MAX_ALLOWED_ERRORS:
                        continue
                    
                    return filename
                except (json.JSONDecodeError, IOError, KeyError):
                    continue
    
    return None

def result_exists(model_name: str, context_length: int, test_name: str, book_num: int) -> bool:
    """Check if result file already exists and is valid."""
    return find_existing_result(model_name, context_length, test_name, book_num) is not None

# ============ Run Single Test ============

async def run_single_test(
    config: ModelConfig,
    test: Dict,
    context_length: int,
    book_num: int,
    haystack_lines: List[str],
    haystack_hash: str
) -> Dict:
    """Run a single test (all depths) for one model/test/book combination."""
    
    # Initialize random state
    np.random.seed(test["seed"] + book_num)
    
    # Prepare output structure with standardized path
    result_path = get_result_path(config.name, context_length, test["test_name"], book_num)
    
    # Generate test hash for consistency with existing format
    test_hash = hashlib.sha256(
        f"{test['test_name']}_{context_length}_{book_num}_{test['seed']}".encode()
    ).hexdigest()
    
    outputs = {
        "eval_name": f"{config.name}_rand_book_{book_num}_{test['test_name']}",
        "test_name": test["test_name"],
        "model_name": config.name,
        "retrieval_question": test["retrieval_question"],
        "needle": test["needle"],
        "gold_answers": "",  # For compatibility with existing format
        "system_prompt": test["system_prompt"],
        "use_default_system_prompt": False,
        "task_template": test["task_template"],
        "haystack_path": str(HAYSTACK_DIR / f"rand_shuffle_{context_length}" / f"rand_book_{book_num}.txt"),
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
        "results": [],
        "test_hash": test_hash
    }
    
    # Generate depths
    depths = np.linspace(DOCUMENT_DEPTH_PERCENT_MIN, DOCUMENT_DEPTH_PERCENT_MAX, NUM_DEPTHS) / 100
    
    # Prepare all prompts
    prompts = []
    metadata = []
    
    for depth in depths:
        # Select character (fixed per needle type, not random)
        if "{CHAR}" in test["needle"]:
            selected_character = test["character_set"][0]  # always use the fixed character
            needle = test["needle"].replace("{CHAR}", selected_character)
            question = test["retrieval_question"].replace("{CHAR}", selected_character)
        else:
            selected_character = ""
            needle = test["needle"]
            question = test["retrieval_question"]
        
        # Insert needle
        haystack_with_needle, needle_position = insert_needle(haystack_lines, needle, depth)
        
        # Format with line numbers
        formatted_haystack = format_haystack_with_line_numbers(haystack_with_needle)
        
        # Fill template
        user_prompt = test["task_template"].format(haystack=formatted_haystack, question=question)
        
        # Calculate filled prompt length for compatibility
        filled_prompt_length = len(test["system_prompt"]) + len(user_prompt)
        
        prompts.append((test["system_prompt"], user_prompt))
        metadata.append({
            "selected_character": selected_character,
            "needle": needle,
            "needle_position": needle_position,
            "depth": depth,
            "num_lines": len(haystack_with_needle),
            "context_length_w_filled_template": filled_prompt_length
        })
    
    # Call API in batches
    async def process_batch(batch_prompts, batch_indices):
        tasks = [call_model(config, sp, up) for sp, up in batch_prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    all_responses = [None] * len(prompts)
    
    for i in range(0, len(prompts), config.batch_size):
        batch_end = min(i + config.batch_size, len(prompts))
        batch_prompts = prompts[i:batch_end]
        batch_indices = list(range(i, batch_end))
        
        responses = await process_batch(batch_prompts, batch_indices)
        
        for j, resp in enumerate(responses):
            all_responses[i + j] = resp
        
        # Pause between batches
        if batch_end < len(prompts):
            await asyncio.sleep(config.batch_pause)
    
    # Process responses — skip failed depths entirely (don't pollute results)
    error_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    ctx_label = f"{context_length//1000}K" if context_length < 1_000_000 else f"{context_length//1_000_000}M"
    
    for i, (resp, meta) in enumerate(zip(all_responses, metadata)):
        depth_pct = f"{meta['depth']*100:.0f}%"
        
        if isinstance(resp, Exception):
            error_count += 1
            print(f"    ✗ depth={depth_pct} char={meta['selected_character']}  ERROR: {str(resp)[:120]}")
            # Don't store failed depths — skip entirely
            continue
        
        # Check for empty/null response
        raw_text = resp.get("response", "")
        if not raw_text or raw_text.strip() == "":
            error_count += 1
            print(f"    ✗ depth={depth_pct} char={meta['selected_character']}  EMPTY RESPONSE (skipped)")
            continue
        
        parsed = parse_json_response(raw_text)
        if not parsed:
            error_count += 1
            print(f"    ✗ depth={depth_pct} char={meta['selected_character']}  UNPARSEABLE: {raw_text[:100]}")
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
                "num_haystack_lines_w_needle": meta["num_lines"]
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
        
        # Print response summary
        ans_icon = "✓" if answer_metric else "✗"
        evi_icon = "✓" if evidence_metric else "✗"
        answer_text = str(parsed.get("answer", ""))[:40]
        lines_text = str(parsed.get("lines", []))[:30]
        print(f"    {ans_icon}{evi_icon} depth={depth_pct} char={meta['selected_character']}  "
              f"answer=\"{answer_text}\" lines={lines_text}")
    
    success_count = len(outputs["results"])
    
    # Don't save if no successful depths at all
    if success_count == 0:
        print(f"  ⚠ ALL {len(metadata)} depths failed — result NOT saved")
        return outputs
    
    # Add summary stats to output
    outputs["summary"] = {
        "total_depths": len(metadata),
        "error_count": error_count,
        "success_count": success_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save results
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(outputs, f, indent=2)
    
    if error_count > 0:
        print(f"  Saved {success_count}/{len(metadata)} depths ({error_count} failed depths skipped)")
    
    return outputs

# ============ Main Runner ============

async def run_model_tests(
    model_name: str,
    context_lengths: List[int] = None,
    dry_run: bool = False,
    book_num_override: int = None,
    character_override: str = None
) -> Dict:
    """Run all tests for a specific model."""
    
    if model_name not in MODELS:
        print(f"Error: Unknown model {model_name}")
        return {}
    
    config = MODELS[model_name]
    tests = load_tests()
    
    # Apply character override if specified
    if character_override:
        for t in tests:
            t["character_set"] = [character_override]
    
    all_context_lengths = context_lengths or CONTEXT_LENGTHS
    
    # Filter context lengths based on model's max context limit
    context_lengths = [c for c in all_context_lengths if c <= config.max_context_chars]
    skipped_contexts = [c for c in all_context_lengths if c > config.max_context_chars]
    
    if skipped_contexts:
        print(f"  Note: Skipping contexts > {config.max_context_chars:,} chars: {skipped_contexts}")
    
    # Determine which book(s) to use per context length
    # Default: always book 1 for consistency. Override with --book-num.
    if book_num_override:
        book_nums_for_ctx = {ctx: [book_num_override] for ctx in context_lengths}
    else:
        book_nums_for_ctx = {ctx: [DEFAULT_BOOK] for ctx in context_lengths}
    
    # Count what needs to be done
    total_tests = sum(len(tests) * len(books) for books in book_nums_for_ctx.values())
    completed = 0
    pending = []
    
    for context_length in context_lengths:
        for test in tests:
            for book_num in book_nums_for_ctx[context_length]:
                if result_exists(model_name, context_length, test["test_name"], book_num):
                    completed += 1
                else:
                    pending.append((context_length, test, book_num))
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    if not book_num_override:
        # Show the random book assignment per context length
        book_map = {ctx: books for ctx, books in book_nums_for_ctx.items()}
        sample = list(book_map.items())[:8]
        assignments = ", ".join(f"{ctx//1000}k→book{bks[0]}" for ctx, bks in sample)
        suffix = f", ... ({len(book_map)} total)" if len(book_map) > 8 else ""
        print(f"Book assignments: {assignments}{suffix}")
    print(f"Total tests: {total_tests}")
    print(f"Completed: {completed}")
    print(f"Pending: {len(pending)}")
    
    if dry_run:
        print("\nDry run - would run:")
        for ctx, test, book in pending[:10]:
            print(f"  {ctx} / {test['test_name']} / book_{book}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
        return {"completed": completed, "pending": len(pending)}
    
    if not pending:
        print("All tests completed!")
        return {"completed": completed, "pending": 0}
    
    # Run pending tests with progress bar
    print(f"\nRunning {len(pending)} tests...")
    
    # Group by context length for efficient haystack loading
    by_context = {}
    for ctx, test, book in pending:
        key = (ctx, book)
        if key not in by_context:
            by_context[key] = []
        by_context[key].append(test)
    
    pbar = tqdm(total=len(pending), desc=f"{model_name}")
    errors = []
    
    for (context_length, book_num), tests_for_ctx in by_context.items():
        ctx_label = f"{context_length//1000}K" if context_length < 1_000_000 else f"{context_length//1_000_000}M"
        print(f"\n{'─'*60}")
        print(f"  Context: {ctx_label} ({context_length:,} chars) | Book: {book_num} | Tests: {len(tests_for_ctx)}")
        print(f"{'─'*60}")
        
        # Load haystack once per context/book
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
                    haystack_lines, haystack_hash
                )
            except Exception as e:
                err_msg = str(e)[:80]
                errors.append(f"{ctx_label}/{test['test_name']}/book_{book_num}: {err_msg}")
                print(f"  ✗ FAILED: {err_msg}")
            
            pbar.update(1)
            
            # Rate limiting: each test makes NUM_DEPTHS API calls in batches.
            # Respect RPM (requests/min) and, when set, TPM (tokens/min) so large contexts stay under quota.
            calls_per_test = NUM_DEPTHS
            min_time_per_test_rpm = (calls_per_test / config.rpm_limit) * 60  # seconds
            min_time_per_test = min_time_per_test_rpm
            if config.tpm_limit and config.tpm_limit > 0:
                # Estimate tokens per test: NUM_DEPTHS calls × (input ~context_length/4 + output ~max_tokens)
                est_input_per_call = context_length // CHARS_PER_TOKEN
                est_tokens_per_test = NUM_DEPTHS * (est_input_per_call + config.max_tokens)
                min_time_per_test_tpm = (est_tokens_per_test / config.tpm_limit) * 60
                min_time_per_test = max(min_time_per_test, min_time_per_test_tpm)
            # Subtract batch pauses already accounted for within run_single_test
            batches_per_test = (NUM_DEPTHS + config.batch_size - 1) // config.batch_size
            batch_pause_time = (batches_per_test - 1) * config.batch_pause
            additional_pause = max(0, min_time_per_test - batch_pause_time)
            if additional_pause > 0:
                await asyncio.sleep(additional_pause)
    
    pbar.close()
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:5]:
            print(f"  {e}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    return {"completed": completed + len(pending) - len(errors), "errors": len(errors)}

async def run_provider_models(
    provider: str,
    models: List[str],
    context_lengths: List[int],
    dry_run: bool,
    book_num_override: int = None,
    character_override: str = None
) -> Dict[str, Dict]:
    """Run all models for a single provider. Google (Gemini) runs models in parallel."""
    if provider == "google" and len(models) > 1:
        # Run both Gemini models in parallel to use headroom (RPM 15/1K → ~100/1K with 2)
        tasks = [
            run_model_tests(m, context_lengths, dry_run, book_num_override, character_override)
            for m in models
        ]
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
        results[model] = await run_model_tests(model, context_lengths, dry_run,
                                                book_num_override, character_override)
    return results

async def run_all_models(
    models: List[str] = None,
    context_lengths: List[int] = None,
    dry_run: bool = False,
    parallel: bool = False,
    book_num_override: int = None,
    character_override: str = None
):
    """Run tests for all specified models.
    
    If parallel=True, runs different providers concurrently while
    running models within the same provider sequentially.
    """
    
    models = models or list(MODELS.keys())
    
    print("=" * 60)
    print("NoLiMa Test Runner (Lite Mode)")
    print("=" * 60)
    print(f"Models: {models}")
    print(f"Context lengths: {context_lengths or CONTEXT_LENGTHS}")
    num_tests = len(load_tests())
    book_count = 1 if book_num_override else NUM_BOOKS
    num_contexts = len(context_lengths or CONTEXT_LENGTHS)
    print(f"Tests per model: {num_tests} tests × {book_count} book(s) × up to {num_contexts} contexts × {NUM_DEPTHS} depths")
    print(f"Max API calls per model: ~{num_tests * book_count * num_contexts * NUM_DEPTHS} (before model context-limit filtering)")
    if character_override:
        print(f"Character override: {character_override}")
    if book_num_override:
        print(f"Book override: book {book_num_override}")
    print(f"Parallel providers: {parallel}")
    
    if parallel:
        # Group models by provider
        by_provider: Dict[str, List[str]] = {}
        for model in models:
            if model in MODELS:
                provider = MODELS[model].provider
                if provider not in by_provider:
                    by_provider[provider] = []
                by_provider[provider].append(model)
        
        print(f"\nRunning {len(by_provider)} providers in parallel:")
        for provider, provider_models in by_provider.items():
            print(f"  {provider}: {provider_models}")
        
        # Run each provider's models in parallel
        tasks = [
            run_provider_models(provider, provider_models, context_lengths, dry_run,
                                book_num_override, character_override)
            for provider, provider_models in by_provider.items()
        ]
        
        provider_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results
        results = {}
        for i, (provider, _) in enumerate(by_provider.items()):
            if isinstance(provider_results[i], Exception):
                print(f"\nError running {provider}: {provider_results[i]}")
            else:
                results.update(provider_results[i])
    else:
        # Sequential execution
        results = {}
        for model in models:
            results[model] = await run_model_tests(model, context_lengths, dry_run,
                                                    book_num_override, character_override)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model, stats in results.items():
        print(f"{model}: completed={stats.get('completed', 0)}, pending={stats.get('pending', 0)}, errors={stats.get('errors', 0)}")

def main():
    parser = argparse.ArgumentParser(description="NoLiMa Full-Scale Test Runner")
    parser.add_argument("--model", help="Run specific model only")
    parser.add_argument("--context-length", type=int, help="Run specific context length only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without running")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--parallel", action="store_true", 
                        help="Run different providers in parallel (OpenAI, Anthropic, Google)")
    parser.add_argument("--gemini-only", action="store_true",
                        help="Run only Gemini models (both in parallel); ignores --model")
    parser.add_argument("--character", type=str, default=None,
                        help="Override character name for needle insertion (e.g. 'Yuki')")
    parser.add_argument("--book-num", type=int, default=None,
                        help="Use a specific book number instead of iterating (e.g. 3)")
    parser.add_argument("--num-depths", type=int, default=None,
                        help="Override number of depth positions (default: 4)")
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for name, config in MODELS.items():
            print(f"  {name} ({config.provider})")
        return
    
    # Apply CLI overrides to global config
    global NUM_DEPTHS, NUM_BOOKS, NUM_CHARACTERS
    if args.num_depths:
        NUM_DEPTHS = args.num_depths
    if args.book_num:
        NUM_BOOKS = 1  # Will use only the specified book
    if args.character:
        NUM_CHARACTERS = 1  # Enforced to 1 when character is specified
    
    # Check API keys
    providers_needed = set()
    if args.gemini_only:
        models = [m for m in MODELS if MODELS[m].provider == "google"]
    elif args.model:
        models = [args.model]
    else:
        models = list(MODELS.keys())
    for m in models:
        if m in MODELS:
            providers_needed.add(MODELS[m].provider)
    
    if not args.dry_run:
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
        book_num_override=args.book_num,
        character_override=args.character
    ))

if __name__ == "__main__":
    main()
