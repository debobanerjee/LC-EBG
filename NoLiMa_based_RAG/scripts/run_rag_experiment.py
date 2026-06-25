#!/usr/bin/env python3
"""
RAG Top-K Experiment Runner

Builds context from similarity-ranked chunks (cosine similarity via Qdrant) and
evaluates LLMs on needle-in-haystack questions.  Unlike the LC baseline
(test/run_full_scale.py), needles are NOT inserted at fixed depth positions —
they are retrieved by similarity score.

Context size: determined by the twohop max_rank from topk_character_budget.csv
for the given context_length.  The same K is used for every question (onehop
and twohop) because twohop is the harder retrieval and sets the upper bound.

Context format (per chunk):
    {evidence}: {text}
where `evidence` and `text` come from the similarity-ranked JSON files under
NoLiMa_based_RAG/results/NeedleRanking/{context_length}/.

Prompts are built from datasets/NoLiMa/needlesets/needle_set.json
using the same system_prompt and task_template as the LC baseline.

Output follows the same JSON schema as test/run_full_scale.py.  Depth-related
fields are set to -1 / NA since RAG has no fixed insertion depth.

Usage:
  python3 NoLiMa_based_RAG/scripts/run_rag_experiment.py                # all models, all ctx
  python3 NoLiMa_based_RAG/scripts/run_rag_experiment.py --model claude-sonnet-4-5-20250929
  python3 NoLiMa_based_RAG/scripts/run_rag_experiment.py --context-length 200000
  python3 NoLiMa_based_RAG/scripts/run_rag_experiment.py --dry-run
  python3 NoLiMa_based_RAG/scripts/run_rag_experiment.py --list-models
  python3 NoLiMa_based_RAG/scripts/run_rag_experiment.py --parallel
"""

import os
import json
import asyncio
import argparse
import re
import time
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============ Configuration ============

CHARS_PER_TOKEN = 4
CONTEXT_SAFETY_FACTOR = 0.99


def _ctx(tokens: int, cap: int = 1_000_000) -> int:
    return min(int(tokens * CHARS_PER_TOKEN * CONTEXT_SAFETY_FACTOR), cap)


@dataclass
class ModelConfig:
    name: str
    provider: str          # "openai" | "anthropic" | "google"
    max_tokens: int = 2048
    batch_size: int = 20
    batch_pause: float = 1.0
    rpm_limit: int = 60
    tpm_limit: int = 0     # 0 = not enforced
    max_context_chars: int = 1_000_000


MODELS = {
    # OpenAI models (matching test/run_full_scale.py for direct comparison)
    "gpt-4o": ModelConfig(
        "gpt-4o", "openai", batch_size=20, rpm_limit=500,
        max_context_chars=_ctx(128_000)),
    "gpt-4.1": ModelConfig(
        "gpt-4.1", "openai", batch_size=20, rpm_limit=500,
        max_context_chars=_ctx(1_000_000)),
    "o3-mini-2025-01-31": ModelConfig(
        "o3-mini-2025-01-31", "openai", batch_size=20, rpm_limit=500,
        max_context_chars=_ctx(200_000)),
    "gpt-5-2025-08-07": ModelConfig(
        "gpt-5-2025-08-07", "openai", max_tokens=1024, batch_size=10, rpm_limit=100,
        max_context_chars=_ctx(400_000)),
    # Anthropic models (matching test/run_full_scale.py for direct comparison)
    "claude-sonnet-4-20250514": ModelConfig(
        "claude-sonnet-4-20250514", "anthropic", batch_size=20, rpm_limit=50,
        max_context_chars=_ctx(200_000)),
    "claude-sonnet-4-5-20250929": ModelConfig(
        "claude-sonnet-4-5-20250929", "anthropic", batch_size=20, rpm_limit=50,
        max_context_chars=_ctx(200_000)),
}

# Fixed character per needle type (matches run_full_scale.py NEEDLE_CHARACTER)
NEEDLE_CHARACTER = {
    "0402":    "Yuki",
    "0402Inv": "Yuki",
    "0405":    "Stuart",
    "0405Inv": "Stuart",
}

REASONING_TYPE  = "commonsense_knowledge"
BOOK_NUM        = 1

# Selective haystacks for this experiment (compare against LC baseline at same sizes)
DEFAULT_CONTEXT_LENGTHS = [200_000, 400_000, 500_000, 600_000, 1_000_000]

SCROLL_ALL_DIR  = Path("NoLiMa_based_RAG/results/NeedleRanking")
BUDGET_CSV      = Path("NoLiMa_based_RAG/tables/topk_character_budget_twohop.csv")
NEEDLE_SET_PATH = Path("datasets/NoLiMa/needlesets/needle_set.json")
QUESTIONS_PATH  = Path("NoLiMa_based_RAG/questions/questions.json")
RESULTS_DIR     = Path("NoLiMa_based_RAG/rag_experiment_results")

# ============ Data Loading ============

def validate_required_inputs() -> None:
    """Fail early with setup guidance instead of a later traceback."""
    required_files = [
        (BUDGET_CSV, "top-k budget CSV"),
        (QUESTIONS_PATH, "precomputed questions JSON"),
        (NEEDLE_SET_PATH, "NoLiMa needle set"),
    ]
    required_dirs = [
        (SCROLL_ALL_DIR, "precomputed retrieval results"),
    ]

    missing = []
    for path, label in required_files:
        if not path.is_file():
            missing.append(f"  - {label}: {path}")
    for path, label in required_dirs:
        if not path.is_dir():
            missing.append(f"  - {label}: {path}")

    if missing:
        print("Error: required inputs are missing:")
        print("\n".join(missing))
        print("\nFrom EBG_repo/, make sure datasets/NoLiMa is present.")
        print("If haystacks are missing, run datasets/NoLiMa/scripts/download_NoLiMa_data.sh")
        print("or generate them with datasets/NoLiMa/scripts/generate_missing_haystacks.py.")
        raise SystemExit(1)


def load_budget() -> Dict[str, Dict]:
    """Return {haystack_key: {max_rank, chars}} from the twohop budget CSV.

    `chars` is the actual RAG context size (top-K retrieved chunks), NOT the full
    haystack size.  This is what gets sent to the model and is the right value to
    compare against a model's context window limit.
    """
    budget: Dict[str, Dict] = {}
    with open(BUDGET_CSV) as f:
        for row in csv.DictReader(f):
            budget[row["haystack"]] = {
                "max_rank": int(row["max_rank"]),
                "chars":    int(row["chars"]),
            }
    return budget


def load_questions() -> List[Dict]:
    """Load the 8 pre-filled questions from questions.json (no embeddings needed)."""
    with open(QUESTIONS_PATH) as f:
        data = json.load(f)
    # Strip embeddings — only keep metadata fields
    return [
        {k: v for k, v in q.items() if k != "embedding"}
        for q in data
    ]


def load_needle_configs() -> Dict[str, Dict]:
    """Return needle_set entries keyed by id, filtered to the 4 task IDs."""
    with open(NEEDLE_SET_PATH) as f:
        needle_set = json.load(f)
    allowed = {"0402", "0402Inv", "0405", "0405Inv"}
    return {n["id"]: n for n in needle_set if n.get("id") in allowed}


def load_ranked_chunks(context_length: int, task_id: str, question_key: str) -> List[Dict]:
    """Load the similarity-ranked chunk list for one (context_length, q_id) pair."""
    fname = f"{context_length}_rand_book_{BOOK_NUM}_{task_id}_{question_key}.json"
    fpath = SCROLL_ALL_DIR / str(context_length) / fname
    with open(fpath) as f:
        return json.load(f)


def context_lengths_from_budget() -> List[int]:
    """Return DEFAULT_CONTEXT_LENGTHS, validated against the budget CSV."""
    available = set()
    with open(BUDGET_CSV) as f:
        for row in csv.DictReader(f):
            available.add(int(row["haystack"].split("_")[0]))
    return [c for c in DEFAULT_CONTEXT_LENGTHS if c in available]


# ============ Context Building ============

def format_rag_context(chunks: List[Dict], topk: int) -> str:
    """
    Format the top-K similarity-ranked chunks as line-numbered context.

    Each line: '{evidence}: {text}'
    where `evidence` is the original line number in the haystack (used as the
    reference index the model is asked to cite) and `text` is the chunk content.
    """
    return "\n".join(
        f"{c['evidence']}: {c['text']}"
        for c in chunks[:topk]
    )


def find_needle_chunk(chunks: List[Dict], task_id: str) -> Optional[Dict]:
    """
    Find the relevant needle chunk for task_id in the ranked list.
    Returns dict with 'chunk' and 'rank_1based', or None if not found.
    """
    for i, c in enumerate(chunks):
        if c.get("chunk_type") == "needle" and c.get("needle_item_id") == task_id:
            return {"chunk": c, "rank_1based": i + 1}
    return None


# ============ Prompt Building ============

def build_prompts(
    needle_config: Dict,
    question: str,
    formatted_context: str,
) -> Tuple[str, str]:
    """Build system_prompt and user_prompt from the needle config entry."""
    system_prompt = needle_config["system_prompt"]
    user_prompt = needle_config["task_template"].format(
        haystack=formatted_context,
        question=question,
    )
    return system_prompt, user_prompt


# ============ API Clients ============

_openai_client = None
_anthropic_client = None
# _google_configured = False  # uncomment if adding Gemini support


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


# def get_google_model(model_name: str):
#     global _google_configured
#     import warnings
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     warnings.filterwarnings("ignore", category=DeprecationWarning)
#     import google.generativeai as genai
#     if not _google_configured:
#         genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#         _google_configured = True
#     return genai.GenerativeModel(model_name), genai


# ============ Retry Helpers ============

def is_retryable_error(e: Exception) -> bool:
    s = str(e).lower()
    if any(p in s for p in ["billing", "insufficient funds", "payment required", "access denied"]):
        return False
    if any(p in s for p in ["quota", "resourceexhausted"]):
        return True
    return any(p in s for p in [
        "rate limit", "rate_limit", "429 too many requests",
        "503", "502", "504", "timeout", "timed out",
        "connection", "overloaded", "temporarily unavailable",
        "server error", "internal error",
    ])


# ============ API Call Functions ============

async def call_openai(config: ModelConfig, system_prompt: str, user_prompt: str) -> Dict:
    client = get_openai_client()
    kwargs = {
        "model": config.name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    if any(x in config.name for x in ("gpt-5", "o1", "o3", "o4")):
        kwargs["max_completion_tokens"] = config.max_tokens
    else:
        kwargs["max_tokens"] = config.max_tokens

    last_error = None
    for attempt in range(3):
        try:
            result = await client.chat.completions.create(**kwargs)
            return {
                "response":      result.choices[0].message.content,
                "input_tokens":  result.usage.prompt_tokens,
                "output_tokens": result.usage.completion_tokens,
            }
        except Exception as e:
            last_error = e
            if attempt < 2 and is_retryable_error(e):
                await asyncio.sleep((attempt + 1) * 5)
            elif not is_retryable_error(e):
                raise
    raise last_error


async def call_anthropic(config: ModelConfig, system_prompt: str, user_prompt: str) -> Dict:
    client = get_anthropic_client()
    last_error = None
    for attempt in range(3):
        try:
            result = await client.messages.create(
                model=config.name,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=config.max_tokens,
            )
            return {
                "response":      result.content[0].text,
                "input_tokens":  result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
            }
        except Exception as e:
            last_error = e
            if attempt < 2 and is_retryable_error(e):
                await asyncio.sleep((attempt + 1) * 5)
            elif not is_retryable_error(e):
                raise
    raise last_error


# async def call_google(config: ModelConfig, system_prompt: str, user_prompt: str) -> Dict:
#     model, genai = get_google_model(config.name)
#     full_prompt = f"{system_prompt}\n\n{user_prompt}"
#     loop = asyncio.get_event_loop()
#     gen_config = genai.types.GenerationConfig(
#         max_output_tokens=config.max_tokens, temperature=0.0
#     )
#
#     def sync_generate():
#         return model.generate_content(full_prompt, generation_config=gen_config)
#
#     last_error = None
#     for attempt in range(5):
#         try:
#             result = await loop.run_in_executor(None, sync_generate)
#             response_text = ""
#             try:
#                 if hasattr(result, "text") and result.text:
#                     response_text = result.text
#                 elif result.candidates:
#                     candidate = result.candidates[0]
#                     if (hasattr(candidate, "finish_reason") and
#                             candidate.finish_reason.name == "RECITATION"):
#                         raise ValueError("Response blocked due to RECITATION filter")
#                     if candidate.content and candidate.content.parts:
#                         response_text = candidate.content.parts[0].text
#             except ValueError:
#                 raise
#             except Exception:
#                 pass
#             return {
#                 "response": response_text,
#                 "input_tokens":  (result.usage_metadata.prompt_token_count
#                                   if hasattr(result, "usage_metadata") else 0),
#                 "output_tokens": (result.usage_metadata.candidates_token_count
#                                   if hasattr(result, "usage_metadata") else 0),
#             }
#         except Exception as e:
#             last_error = e
#             if attempt < 4 and is_retryable_error(e):
#                 wait = min(60, 10 * (2 ** attempt))
#                 print(f"      Retry {attempt+1}/4 in {wait}s: {str(e)[:80]}")
#                 await asyncio.sleep(wait)
#             elif not is_retryable_error(e):
#                 raise
#     raise last_error


async def call_model(config: ModelConfig, system_prompt: str, user_prompt: str) -> Dict:
    if config.provider == "openai":
        return await call_openai(config, system_prompt, user_prompt)
    if config.provider == "anthropic":
        return await call_anthropic(config, system_prompt, user_prompt)
    # if config.provider == "google":
    #     return await call_google(config, system_prompt, user_prompt)
    raise ValueError(f"Unknown provider: {config.provider}")


# ============ Response Parsing ============

def _extract_brace_balanced(text: str, start: int) -> str:
    if start >= len(text) or text[start] not in "{[":
        return ""
    open_b, close_b = ("{", "}") if text[start] == "{" else ("[", "]")
    depth, i, in_string, escape = 0, start, None, False
    while i < len(text):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == in_string:
                in_string = None
            i += 1
            continue
        if c in "\"'":
            in_string = c
            i += 1
            continue
        if c == open_b:
            depth += 1
        elif c == close_b:
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
        i += 1
    return ""


def parse_json_response(text: str) -> Optional[Dict]:
    if not text:
        return None
    cleaned = re.sub(r",\s*}", "}", text)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    m = re.search(r"```(?:json)?\s*", cleaned, re.IGNORECASE)
    if m:
        start = m.end()
        if start < len(cleaned) and cleaned[start] == "{":
            obj = _extract_brace_balanced(cleaned, start)
            if obj:
                try:
                    return json.loads(obj)
                except (json.JSONDecodeError, ValueError):
                    pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    first = cleaned.find("{")
    if first != -1:
        obj = _extract_brace_balanced(cleaned, first)
        if obj:
            try:
                return json.loads(obj)
            except (json.JSONDecodeError, ValueError):
                pass

    m = re.search(r'\{[^{}]*"answer"[^{}]*\}', cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    first = cleaned.find("{")
    if first != -1:
        tail = cleaned[first:].rstrip()
        for suffix in ('": null}', '"}', "}"):
            try:
                return json.loads(tail + suffix)
            except (json.JSONDecodeError, ValueError):
                continue
    return None


# ============ Evaluation ============

def evaluate_response(
    parsed: Optional[Dict],
    expected_character: str,
    expected_evidence: Optional[int],
) -> Tuple[int, int]:
    """Return (answer_metric, evidence_metric) as 0/1 each."""
    if not parsed:
        return 0, 0

    answer = str(parsed.get("answer", "")).lower()
    answer_correct = expected_character.lower() in answer

    lines = parsed.get("lines", [])
    if not isinstance(lines, list):
        lines = [lines] if lines is not None else []
    int_lines = []
    for line in lines:
        try:
            int_lines.append(int(line))
        except (ValueError, TypeError):
            pass
    evidence_correct = (expected_evidence in int_lines) if expected_evidence is not None else False

    return int(answer_correct), int(evidence_correct)


# ============ Result Path & Skip Check ============

def get_result_path(model_name: str, context_length: int, q_id: str) -> Path:
    model_dir = model_name.replace(".", "-").replace("/", "-")
    return (
        RESULTS_DIR
        / f"results_{model_dir}"
        / REASONING_TYPE
        / f"rand_shuffle_{context_length}"
        / q_id
        / f"{model_dir}_rand_book_{BOOK_NUM}_{q_id}.json"
    )


def result_exists(model_name: str, context_length: int, q_id: str) -> bool:
    path = get_result_path(model_name, context_length, q_id)
    if not path.exists():
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        results = data.get("results", [])
        if not results:
            return False
        return not results[0].get("error")
    except (json.JSONDecodeError, IOError):
        return False


# ============ Run Single Test ============

async def run_single_test(
    config: ModelConfig,
    q_entry: Dict,
    needle_configs: Dict[str, Dict],
    context_length: int,
    budget: Dict[str, Dict],
) -> Dict:
    task_id      = q_entry["task_id"]
    question_key = q_entry["question_key"]
    q_id         = q_entry["q_id"]
    question     = q_entry["question"]

    haystack_key = f"{context_length}_rand_book_{BOOK_NUM}"
    topk = budget[haystack_key]["max_rank"]

    chunks = load_ranked_chunks(context_length, task_id, question_key)

    needle_info     = find_needle_chunk(chunks, task_id)
    needle_rank     = needle_info["rank_1based"]    if needle_info else None
    needle_evidence = needle_info["chunk"]["evidence"] if needle_info else None
    needle_text     = needle_info["chunk"]["text"]     if needle_info else ""

    formatted_context = format_rag_context(chunks, topk)
    needle_config     = needle_configs[task_id]
    system_prompt, user_prompt = build_prompts(needle_config, question, formatted_context)

    character            = NEEDLE_CHARACTER[task_id]
    filled_prompt_length = len(system_prompt) + len(user_prompt)
    result_path          = get_result_path(config.name, context_length, q_id)
    model_dir            = config.name.replace(".", "-").replace("/", "-")
    chunk_path           = str(
        SCROLL_ALL_DIR / str(context_length) /
        f"{context_length}_rand_book_{BOOK_NUM}_{task_id}_{question_key}.json"
    )

    output: Dict = {
        "eval_name":                f"{model_dir}_rand_book_{BOOK_NUM}_{q_id}",
        "test_name":                q_id,
        "model_name":               config.name,
        "retrieval_question":       question,
        "needle":                   needle_text,
        "gold_answers":             "",
        "system_prompt":            system_prompt,
        "use_default_system_prompt": False,
        "task_template":            needle_config["task_template"],
        "rag_chunk_path":           chunk_path,
        "context_length":           context_length,
        "character_set":            [character],
        "topk_used":                topk,
        "needle_rank_in_full_list": needle_rank,
        # depth fields kept for schema compatibility; not meaningful for RAG
        "document_depth_percent_min":       0,
        "document_depth_percent_max":       100,
        "document_depth_percent_intervals": 1,
        "shift":        0,
        "static_depth": -1,
        "metric":       "contains",
        "result_dir":   str(result_path.parent),
        "results":      [],
    }

    ctx_label = f"{context_length//1000}K" if context_length < 1_000_000 else f"{context_length//1_000_000}M"
    print(f"    {ctx_label} | {q_id} | topk={topk} | needle_rank={needle_rank}")

    try:
        resp = await call_model(config, system_prompt, user_prompt)
    except Exception as e:
        print(f"    ERROR: {str(e)[:120]}")
        output["results"].append({"error": str(e), "error_type": type(e).__name__})
        return output

    raw_text = resp.get("response", "")
    if not raw_text or not raw_text.strip():
        print(f"    EMPTY RESPONSE")
        output["results"].append({"error": "empty_response"})
        return output

    parsed = parse_json_response(raw_text)
    if not parsed:
        print(f"    UNPARSEABLE: {raw_text[:100]}")
        output["results"].append({"error": "unparseable", "raw_response": raw_text[:500]})
        return output

    answer_metric, evidence_metric = evaluate_response(parsed, character, needle_evidence)
    ans_icon = "✓" if answer_metric else "✗"
    evi_icon = "✓" if evidence_metric else "✗"
    print(f"    {ans_icon}{evi_icon}  answer=\"{str(parsed.get('answer',''))[:40]}\"  "
          f"lines={str(parsed.get('lines',[]))[:30]}")

    output["results"].append({
        "selected_character": character,
        "context_length_w_filled_template": filled_prompt_length,
        "placement_metadata": {
            "needle":              needle_text,
            "needle_evidence_line": needle_evidence,
            "needle_rank_1based":   needle_rank,
            "depth":               -1,   # not applicable for RAG
            "topk_used":           topk,
            "num_chunks_in_context": min(topk, len(chunks)),
        },
        "response":        parsed,
        "answer_metric":   answer_metric,
        "evidence_metric": evidence_metric,
        "input_tokens":    resp["input_tokens"],
        "output_tokens":   resp.get("output_tokens", 0),
    })
    output["summary"] = {
        "answer_metric":     answer_metric,
        "evidence_metric":   evidence_metric,
        "total_input_tokens":  resp["input_tokens"],
        "total_output_tokens": resp.get("output_tokens", 0),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


# ============ Model Runner ============

async def run_model_tests(
    model_name: str,
    context_lengths_override: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict:
    if model_name not in MODELS:
        print(f"Error: Unknown model '{model_name}'")
        return {}

    config        = MODELS[model_name]
    budget        = load_budget()
    questions     = load_questions()
    needle_configs = load_needle_configs()

    all_ctx = context_lengths_override or context_lengths_from_budget()

    # Compare actual RAG context chars (top-K chunks) against model limit — NOT the
    # full haystack size, since only retrieved chunks are sent to the model.
    def rag_chars(ctx: int) -> int:
        return budget.get(f"{ctx}_rand_book_{BOOK_NUM}", {}).get("chars", 0)

    ctx_lengths = [c for c in all_ctx if rag_chars(c) <= config.max_context_chars]
    skipped     = [c for c in all_ctx if rag_chars(c) > config.max_context_chars]
    if skipped:
        print(f"  Skipping (RAG context exceeds {config.max_context_chars:,} chars): {skipped}")

    pending: List[Tuple[int, Dict]] = []
    completed = 0
    for ctx in ctx_lengths:
        key = f"{ctx}_rand_book_{BOOK_NUM}"
        if key not in budget:
            print(f"  Warning: {key} not in budget CSV — skipping")
            continue
        chunk_dir = SCROLL_ALL_DIR / str(ctx)
        if not chunk_dir.exists():
            print(f"  Warning: {chunk_dir} does not exist — skipping {ctx}")
            continue
        for q in questions:
            if result_exists(model_name, ctx, q["q_id"]):
                completed += 1
            else:
                pending.append((ctx, q))

    total = completed + len(pending)
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Total: {total}  |  Completed: {completed}  |  Pending: {len(pending)}")

    if dry_run:
        print("\nDry run — would run:")
        for ctx, q in pending[:12]:
            print(f"  {ctx} / {q['q_id']}")
        if len(pending) > 12:
            print(f"  ... and {len(pending)-12} more")
        return {"completed": completed, "pending": len(pending), "errors": 0}

    if not pending:
        print("All tests completed!")
        return {"completed": completed, "pending": 0, "errors": 0}

    errors: List[str] = []
    for ctx, q in pending:
        ctx_label = f"{ctx//1000}K" if ctx < 1_000_000 else f"{ctx//1_000_000}M"
        print(f"\n  [{model_name}] {ctx_label} / {q['q_id']}")
        try:
            await run_single_test(config, q, needle_configs, ctx, budget)
        except Exception as e:
            msg = str(e)[:100]
            errors.append(f"{ctx}/{q['q_id']}: {msg}")
            print(f"  FAILED: {msg}")

        # Rate-limit pause: one API call per test
        min_time_rpm = (1 / config.rpm_limit) * 60
        min_time = min_time_rpm
        if config.tpm_limit:
            est_tokens = ctx // CHARS_PER_TOKEN + config.max_tokens
            min_time = max(min_time, (est_tokens / config.tpm_limit) * 60)
        if min_time > 0:
            await asyncio.sleep(min_time)

    print(f"\n  Done — {len(pending)-len(errors)} ok, {len(errors)} failed")
    if errors:
        for e in errors[:5]:
            print(f"    {e}")
    return {"completed": completed + len(pending) - len(errors), "errors": len(errors)}


# ============ Top-Level Orchestrator ============

async def run_all_models(
    models: Optional[List[str]] = None,
    context_lengths: Optional[List[int]] = None,
    dry_run: bool = False,
    parallel: bool = False,
) -> None:
    models = models or list(MODELS.keys())

    print("=" * 60)
    print("RAG Top-K Experiment Runner")
    print("=" * 60)
    print(f"Models:           {models}")
    print(f"Context lengths:  {context_lengths or context_lengths_from_budget()}")
    print(f"Results dir:      {RESULTS_DIR}")
    print(f"Parallel:         {parallel}")

    results: Dict[str, Dict] = {}

    if parallel:
        by_provider: Dict[str, List[str]] = {}
        for m in models:
            if m in MODELS:
                by_provider.setdefault(MODELS[m].provider, []).append(m)

        provider_tasks = [
            asyncio.gather(*[run_model_tests(m, context_lengths, dry_run) for m in ms])
            for ms in by_provider.values()
        ]
        provider_results = await asyncio.gather(*provider_tasks, return_exceptions=True)

        for provider_models, res_list in zip(by_provider.values(), provider_results):
            if isinstance(res_list, Exception):
                for m in provider_models:
                    results[m] = {"completed": 0, "errors": 1}
            else:
                for m, r in zip(provider_models, res_list):
                    results[m] = r if isinstance(r, dict) else {"completed": 0, "errors": 1}
    else:
        for m in models:
            results[m] = await run_model_tests(m, context_lengths, dry_run)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for m, stats in results.items():
        print(f"  {m}: completed={stats.get('completed',0)}, errors={stats.get('errors',0)}")


# ============ CLI ============

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Top-K Experiment Runner")
    parser.add_argument("--model",          help="Run specific model only")
    parser.add_argument("--context-length", type=int, help="Run specific context length only")
    parser.add_argument("--dry-run",        action="store_true", help="Show what would run, don't call APIs")
    parser.add_argument("--list-models",    action="store_true", help="List available models and exit")
    parser.add_argument("--parallel",       action="store_true",
                        help="Run OpenAI and Anthropic providers concurrently")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, cfg in MODELS.items():
            print(f"  {name}  ({cfg.provider})  max_ctx={cfg.max_context_chars:,} chars")
        return

    models         = [args.model] if args.model else None
    context_lengths = [args.context_length] if args.context_length else None

    validate_required_inputs()

    if not args.dry_run:
        providers_needed = {
            MODELS[m].provider
            for m in (models or list(MODELS.keys()))
            if m in MODELS
        }
        missing = []
        if "openai"    in providers_needed and not os.getenv("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")
        if "anthropic" in providers_needed and not os.getenv("ANTHROPIC_API_KEY"):
            missing.append("ANTHROPIC_API_KEY")
        # if "google" in providers_needed and not os.getenv("GOOGLE_API_KEY"):
        #     missing.append("GOOGLE_API_KEY")
        if missing:
            print(f"Error: Missing API keys: {missing}")
            print("Set them with: export KEY_NAME='your-key'")
            return

    asyncio.run(run_all_models(
        models=models,
        context_lengths=context_lengths,
        dry_run=args.dry_run,
        parallel=args.parallel,
    ))


if __name__ == "__main__":
    main()
