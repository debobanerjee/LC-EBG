"""
Shared experiment code for context length experiments.

This module is used by both:
- context_experiments_ves.ipynb (notebook)
- run_experiments.py (CLI script)
"""

import random
import copy
import json
import os
import re
import sys
import concurrent.futures
import threading
from datetime import datetime
from enum import Enum


class ContextLengthExceeded(Exception):
    """Raised when the API rejects the request because the prompt exceeds the model context window."""

    pass


def is_context_window_exceeded_error(exc: BaseException) -> bool:
    """
    Detect provider errors that mean "this prompt is too large for the model".
    OpenAI, Anthropic, and Google use different exception shapes; we match on message/code text.
    """
    # OpenAI python SDK: BadRequestError with body in str
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") or {}
        if err.get("code") == "context_length_exceeded":
            return True
        msg = (err.get("message") or "").lower()
        if "maximum context length" in msg and "token" in msg:
            return True
    s = str(exc).lower()
    if "context_length_exceeded" in s:
        return True
    if "maximum context length" in s and "token" in s:
        return True
    if "reduce the length of the messages" in s:
        return True
    # Anthropic
    if "prompt is too long" in s and ("token" in s or "maximum" in s):
        return True
    # Google Generative AI
    if "token count exceeds" in s or "exceeds the maximum" in s and "token" in s:
        return True
    return False


def _shutdown_executor_early(executor):
    """Stop thread pool without waiting for queued work (Python 3.9+ cancels pending futures)."""
    if sys.version_info >= (3, 9):
        executor.shutdown(wait=False, cancel_futures=True)
    else:
        executor.shutdown(wait=False)

# ============================================================================
# Output Storage
# ============================================================================

OUTPUT_DIR = "experiment_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_output_lock = threading.Lock()

def get_output_filename(model_name):
    """Generate a timestamped output filename for a model."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "-").replace(":", "-")
    return os.path.join(OUTPUT_DIR, f"{safe_model_name}_{timestamp}.jsonl")

def save_experiment_output(output_file, record):
    """Append a single experiment record to the output file (thread-safe)."""
    with _output_lock:
        with open(output_file, "a") as f:
            f.write(json.dumps(record) + "\n")

def load_existing_results(filepath):
    """Load existing results from a JSONL file and count trials per context length."""
    results_by_length = {}  # {num_characters: [list of result records]}
    
    if not os.path.exists(filepath):
        return results_by_length
    
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                num_chars = record.get("num_characters")
                if num_chars is not None:
                    if num_chars not in results_by_length:
                        results_by_length[num_chars] = []
                    results_by_length[num_chars].append(record)
            except json.JSONDecodeError:
                continue
    
    return results_by_length


def successful_records(records):
    """Records where the API call completed (no error). Used for resume quotas and accuracy."""
    if not records:
        return []
    return [r for r in records if not r.get("error")]


def compute_stats_from_records(records):
    """Compute statistics from a list of result records.

    Only **successful** completions (no ``error`` field) are used for accuracies and
    token/character averages, so connection failures do not count as wrong answers.

    If ``records`` is empty, returns None. If all rows are failed attempts, returns
    zero accuracies and count 0 (so callers can still show a row).
    """
    if not records:
        return None

    good = successful_records(records)
    total = len(good)
    if total == 0:
        return {
            "characters": 0,
            "tokens": 0,
            "evidence_accuracy": 0.0,
            "answer_accuracy": 0.0,
            "total_accuracy": 0.0,
            "count": 0,
            "failed_attempts": len(records),
        }

    total_characters = sum(r.get("num_characters", 0) for r in good)
    total_tokens = sum(r.get("input_tokens", 0) for r in good)
    total_correct = sum(1 for r in good if r.get("correct", False))
    answer_correct = sum(1 for r in good if r.get("answer_correct", False))
    evidence_correct = sum(1 for r in good if r.get("lines_correct", False))

    out = {
        "characters": round(total_characters / total) if total > 0 else 0,
        "tokens": round(total_tokens / total) if total > 0 else 0,
        "evidence_accuracy": evidence_correct / total if total > 0 else 0,
        "answer_accuracy": answer_correct / total if total > 0 else 0,
        "total_accuracy": total_correct / total if total > 0 else 0,
        "count": total,
    }
    failed = len(records) - total
    if failed:
        out["failed_attempts"] = failed
    return out

# ============================================================================
# Model Definitions
# ============================================================================

class Provider(Enum):
    """API providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    FIREWORKS = "fireworks"


class Model(Enum):
    """Available models for experiments.
    
    Each model is defined as (api_name, provider).
    
    Usage:
        model = Model.GPT_4O
        print(model.api_name)   # "gpt-4o"
        print(model.provider)   # Provider.OPENAI
    """
    # OpenAI models
    O4_MINI = ("o4-mini-2025-04-16", Provider.OPENAI)
    O3_MINI = ("o3-mini-2025-01-31", Provider.OPENAI)
    GPT_4O = ("gpt-4o", Provider.OPENAI)
    GPT_4_1 = ("gpt-4.1", Provider.OPENAI)
    GPT_5 = ("gpt-5", Provider.OPENAI)
    GPT_5_1 = ("gpt-5.1", Provider.OPENAI)
    GPT_5_2 = ("gpt-5.2", Provider.OPENAI)
    GPT_5_4 = ("gpt-5.4", Provider.OPENAI)
    GPT_5_5 = ("gpt-5.5", Provider.OPENAI)  # current OpenAI flagship (see platform.openai.com/docs/models)
    
    # Anthropic models
    SONNET_4_6 = ("claude-sonnet-4-6", Provider.ANTHROPIC)
    OPUS_4_6 = ("claude-opus-4-6", Provider.ANTHROPIC)
    SONNET_4_5 = ("claude-sonnet-4-5-20250929", Provider.ANTHROPIC)
    SONNET_4 = ("claude-sonnet-4-20250514", Provider.ANTHROPIC)
    SONNET_3_7 = ("claude-3-7-sonnet-20250219", Provider.ANTHROPIC)
    
    # Google models
    GEMINI_3_1_PRO = ("gemini-3.1-pro-preview", Provider.GOOGLE)
    GEMINI_3_FLASH = ("gemini-3-flash-preview", Provider.GOOGLE)
    GEMINI_2_5_PRO = ("gemini-2.5-pro", Provider.GOOGLE)
    
    # Open-source models via Fireworks AI
    LLAMA_4_SCOUT = ("accounts/fireworks/models/llama4-scout-instruct-basic", Provider.FIREWORKS)
    LLAMA_4_MAVERICK = ("accounts/fireworks/models/llama4-maverick-instruct-basic", Provider.FIREWORKS)
    DEEPSEEK_V3_2 = ("accounts/fireworks/models/deepseek-v3p2", Provider.FIREWORKS)
    QWEN3_235B = ("accounts/fireworks/models/qwen3-235b-a22b", Provider.FIREWORKS)
    KIMI_K2_5 = ("accounts/fireworks/models/kimi-k2p5", Provider.FIREWORKS)
    
    def __init__(self, api_name: str, provider: Provider):
        self.api_name = api_name
        self.provider = provider
    
    @property
    def name(self) -> str:
        """Alias for api_name for backwards compatibility."""
        return self.api_name
    
    def __repr__(self):
        return f"Model.{self._name_}"
    
    def __str__(self):
        return self.api_name
    
    @classmethod
    def from_string(cls, name: str) -> "Model":
        """Look up a model by its CLI name or API name."""
        # Try exact enum name match (e.g., "GPT_4O")
        try:
            return cls[name.upper().replace("-", "_")]
        except KeyError:
            pass
        
        # Try API name match (e.g., "gpt-4o")
        for model in cls:
            if model.api_name == name:
                return model
        
        # Try CLI-style name match (e.g., "gpt-4o" -> GPT_4O)
        cli_name = name.lower().replace(".", "_").replace("-", "_")
        for model in cls:
            if model._name_.lower() == cli_name:
                return model
        
        raise ValueError(f"Unknown model: {name}. Available: {[m._name_ for m in cls]}")


# For CLI compatibility - maps CLI argument names to Model enum values
MODELS = {model._name_.lower().replace("_", "-"): model for model in Model}

# Character length presets
PRESETS = {
    "tiny": [500, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
    "short": [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
    "mid": [200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000],
    "long": [2000000, 3000000, 4000000, 5000000],
    "full": [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 
             200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 
             2000000, 3000000, 4000000, 5000000],
}

# ============================================================================
# Experiment Examples
# ============================================================================

class Example:
    def __init__(self, facts, question, answers):
        self.question = question
        self.answers = answers
        self.facts = facts

TWO_HOP_EXAMPLES = [
    Example(
        ["Pedro Oliver is the CEO of Coloma Inc.", "Pedro Oliver is 67"], 
        "How old is the CEO of Coloma?", 
        ["67", "67 years old", "Pedro Oliver is 67", "Pedro Oliver, the CEO of Coloma Inc., is 67 years old", 
         "Pedro Oliver is 67 years old", "The CEO of Coloma Inc., Pedro Oliver, is 67 years old"]
    ),
    Example(
        ["Carol Smith is the Chief Science Officer of Diagonal Labs", "Carol Smith was born in Berlin."],
        "Where was the CSO of Diagonal Labs born?",
        ["Berlin"]
    ),
    Example(
        ["Sam Jacobs is Helen Kim's only child.", "Sam Jacobs is 25 years old."],
        "How old is Helen Kim's son?",
        ["25", "25 years old", "Helen Kim's son, Sam Jacobs, is 25 years old"]
    ),
    Example(
        ["Chevalier is a French restaurant in Antibes.", "Chevalier is owned by Martin Wong."],
        "Where is the restaurant owned by Martin Wong located?",
        ["Antibes", "Antibes, France"]
    ),
    Example(
        ["SkyCloud is a data storage product by Little Sky Software.", "Little Sky Software is based in the town of Cayuga."],
        "Where is the maker of SkyCloud based?",
        ["Cayuga", "Cayuga, New York", "the town of Cayuga", "The town of Cayuga", 
         "Little Sky Software is based in the town of Cayuga"]
    ),
    Example(
        ["James Lee is the founder of Horizon Tech", "Horizon Tech specializes in cloud computing."],
        "What does James Lee's company specialize in?",
        ["cloud computing", "Cloud computing", "Horizon Tech, founded by James Lee, specializes in cloud computing", 
         "Horizon Tech specializes in cloud computing", "James Lee's company, Horizon Tech, specializes in cloud computing"]
    ),
    Example(
        ["Anna Garcia lives in Madrid", "Anna Garcia is the CFO of Vertex Solutions."],
        "Where does the CFO of Vertex Solutions live?",
        ["Madrid", "Anna Garcia lives in Madrid", "The CFO of Vertex Solutions, Anna Garcia, lives in Madrid"]
    ),
    Example(
        ["NovaTech launched a new AI platform called NovaMind", "NovaTech is based in New York."],
        "Where is the maker of NovaMind based?",
        ["New York", "NovaTech is based in New York", "The maker of NovaMind is NovaTech, which is based in New York"]
    ),
    Example(
        ["Himalaya Inc. launched a new smart wearable called FitBand.", "Himalaya Inc. is based in Mumbai."],
        "Which smart wearable was launched by a company based in Mumbai?",
        ["FitBand", "FitBand by Himalaya Inc"]
    ),
    Example(
        ["Victor Alvarez is the CEO of Zenith Pharmaceuticals.", "Zenith Pharmaceuticals developed a new vaccine named Z-Vax in 2023."],
        "Who is the CEO of the company that developed Z-Vax?",
        ["Victor Alvarez"]
    ),
]

# Some providers' safety filters intermittently refuse specific examples
# (returning empty content with no error). For those providers we skip the
# offending examples so accuracy reflects long-context capability rather
# than refusal rates. Identify by question text to be robust to reordering.
EXCLUDED_QUESTIONS_BY_PROVIDER = {
    Provider.ANTHROPIC: {
        # Anthropic safety filter started intermittently refusing this
        # vaccine/pharmaceutical example in mid-2026.
        "Who is the CEO of the company that developed Z-Vax?",
    },
}


def _examples_for_model(model):
    """Return the subset of TWO_HOP_EXAMPLES allowed for ``model``'s provider."""
    excluded = EXCLUDED_QUESTIONS_BY_PROVIDER.get(model.provider, set())
    if not excluded:
        return TWO_HOP_EXAMPLES
    return [ex for ex in TWO_HOP_EXAMPLES if ex.question not in excluded]

# ============================================================================
# Helper Functions
# ============================================================================

def load_random_facts(filepath="random_facts.txt"):
    """Load random facts from file."""
    facts = []
    with open(filepath, "r") as f:
        for line in f:
            facts.append(line.strip())
    return facts

def parse_llm_json_output(output_text):
    """Parse JSON from LLM output."""
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'(\{.*?\})', output_text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None
    return None

def evaluate_llm_output(llm_output, gold_standard):
    """Evaluate if LLM output matches gold standard."""
    if llm_output is None:
        return False, False
    
    llm_answer = str(llm_output.get("answer"))
    gold_answers = gold_standard.get("answer", [])
    answer_match = llm_answer.strip(".") in gold_answers
    
    llm_lines = llm_output.get("lines", [])
    llm_lines.sort()
    gold_lines = gold_standard.get("lines", [])
    gold_lines.sort()
    lines_match = llm_lines == gold_lines
    
    return answer_match, lines_match

def add_random_facts(random_facts, num_characters, needles):
    """Add random facts to create context of specified length."""
    arr_to_shuffle = copy.deepcopy(random_facts)
    random.shuffle(arr_to_shuffle)
    
    total_characters = 0
    index = 0
    while total_characters < num_characters and index < len(arr_to_shuffle):
        total_characters += len(arr_to_shuffle[index]) + 1
        index += 1
    index -= 1
    arr_to_shuffle = arr_to_shuffle[:index]
    
    positions = []
    for needle in needles:
        pos = random.randint(0, len(arr_to_shuffle))
        positions.append(pos)
        arr_to_shuffle.insert(pos, needle)
        for i in range(len(positions) - 1):
            if positions[i] >= pos:
                positions[i] += 1
    
    return arr_to_shuffle, positions

def print_markdown_row(row, header=False):
    """Print a markdown table row."""
    print("| " + " | ".join([str(x) for x in row]) + " |")
    if header:
        print("| --- " * len(row) + "|")

# ============================================================================
# API Clients
# ============================================================================

def setup_clients(oai_client=None, anthropic_client=None, google_client=None):
    """
    Setup API clients. Can accept pre-configured clients or create new ones.
    
    Args:
        oai_client: Pre-configured OpenAI client (optional)
        anthropic_client: Pre-configured Anthropic client (optional)
        google_client: Pre-configured Google genai module (optional)
    
    Returns:
        dict: Dictionary of available clients
    """
    clients = {}
    
    if oai_client:
        clients['openai'] = oai_client
    else:
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                clients['openai'] = openai.OpenAI(api_key=api_key)
        except ImportError:
            pass
    
    if anthropic_client:
        clients['anthropic'] = anthropic_client
    else:
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                clients['anthropic'] = anthropic.Client(api_key=api_key)
        except ImportError:
            pass
    
    if google_client:
        clients['google'] = google_client
    else:
        try:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                clients['google'] = genai
        except ImportError:
            pass
    
    try:
        import openai as _openai_mod
        fw_key = os.getenv("FIREWORKS_API_KEY")
        if fw_key:
            clients['fireworks'] = _openai_mod.OpenAI(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=fw_key,
            )
    except ImportError:
        pass
    
    return clients

# ============================================================================
# Model Execution
# ============================================================================

SYSTEM_PROMPT = "Your job is to answer the question entirely from the context and provide a reference. Your answer should cite all lines the answer is based on."

# Answer-only system prompt: matches the answer-only baseline used in the
# manuscript (no citation requirement, used to measure Ans* accuracy).
SYSTEM_PROMPT_ANSWER_ONLY = "Your job is to answer the question entirely from the provided context."

EXAMPLE_FORMAT = """
{
  "lines": [25, 412],
  "answer": "New York"
}
"""

# Same JSON structure as EXAMPLE_FORMAT but without the "lines" field, used
# for the answer-only baseline so the only difference from EBG is the
# citation requirement.
EXAMPLE_FORMAT_ANSWER_ONLY = """
{
  "answer": "New York"
}
"""

def run_model(clients, model, user_prompt, system_prompt=None):
    """Run a model with the given prompt.

    Args:
        system_prompt: System prompt to use. Defaults to the EBG SYSTEM_PROMPT
            (answer + evidence). Pass SYSTEM_PROMPT_ANSWER_ONLY for the
            answer-only baseline.
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    if model.provider == Provider.OPENAI:
        if 'openai' not in clients:
            raise Exception("OpenAI client not available")
        result = clients['openai'].chat.completions.create(
            model=model.api_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
    elif model.provider == Provider.ANTHROPIC:
        if 'anthropic' not in clients:
            raise Exception("Anthropic client not available")
        result = clients['anthropic'].messages.create(
            model=model.api_name,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=1024
        )
    elif model.provider == Provider.GOOGLE:
        if 'google' not in clients:
            raise Exception("Google client not available")
        google_model = clients['google'].GenerativeModel(
            model.api_name,
            system_instruction=system_prompt,
        )
        result = google_model.generate_content(user_prompt)
    elif model.provider == Provider.FIREWORKS:
        if 'fireworks' not in clients:
            raise Exception("Fireworks client not available. Set FIREWORKS_API_KEY.")
        result = clients['fireworks'].chat.completions.create(
            model=model.api_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
    else:
        raise Exception(f"Unknown provider: {model.provider}")
    return result

def process_answer(result, model):
    """Extract response and token count from model result."""
    if model.provider == Provider.OPENAI:
        response = result.choices[0].message.content
        input_tokens = result.usage.prompt_tokens
    elif model.provider == Provider.ANTHROPIC:
        # content can be empty (e.g., refusal, unusual stop_reason) or contain
        # blocks without a .text attribute (e.g., tool_use). Defensively pull
        # the first text block, falling back to "".
        response = ""
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                response = text
                break
        input_tokens = result.usage.input_tokens
    elif model.provider == Provider.GOOGLE:
        try:
            response = result.text
        except ValueError:
            response = ""
        input_tokens = result.usage_metadata.prompt_token_count if hasattr(result, 'usage_metadata') else 0
    elif model.provider == Provider.FIREWORKS:
        response = result.choices[0].message.content
        input_tokens = result.usage.prompt_tokens
    return response, input_tokens

# ============================================================================
# Experiment Runner
# ============================================================================

class Result:
    def __init__(self, characters, tokens, correct, answer_correct, lines_correct):
        self.characters = characters
        self.tokens = tokens
        self.correct = correct
        self.answer_correct = answer_correct
        self.lines_correct = lines_correct

def run_experiment(clients, random_facts, num_characters, model, output_file=None, answer_only=False):
    """Run a single experiment.

    Args:
        answer_only: If True, runs the answer-only baseline: no system prompt
            about citations, and the user prompt asks for the final answer
            only (no line numbers). The output is recorded with
            ``answer_only=True`` and ``lines_correct=None`` since evidence is
            not produced. Used to compute Ans* in the manuscript.
    """
    ex = random.choice(_examples_for_model(model))
    fact1 = ex.facts[0]
    fact2 = ex.facts[1]
    question = ex.question
    answers = ex.answers
    
    context, positions = add_random_facts(random_facts, num_characters, [fact1, fact2])
    positions.sort()
    answer_key = {"answer": answers, "lines": positions}
    num_input_characters = sum(len(c) + 1 for c in context)
    numbered_context = "\n".join([f"{i}: {c}" for i, c in enumerate(context)])
    
    if answer_only:
        user_prompt = f"<context>\n{numbered_context}\n<context>\n\nAnswer the question based on information only from the context. If the question is not answerable from the context, answer NA. Your response should only contain a short answer (no details) in json format. For example:{EXAMPLE_FORMAT_ANSWER_ONLY}\n\nQuestion: {question}"
        system_prompt_used = SYSTEM_PROMPT_ANSWER_ONLY
    else:
        user_prompt = f"<context>\n{numbered_context}\n<context>\n\nAnswer the question based on information only from the context. If the question is not answerable from the context, answer NA. Your response should only contain a short answer (no details) and all lines the answer is based on in json format. For example:{EXAMPLE_FORMAT}\n\nQuestion: {question}"
        system_prompt_used = SYSTEM_PROMPT

    response = None
    error_message = None
    
    try:
        result = run_model(clients, model, user_prompt, system_prompt=system_prompt_used)
    except Exception as e:
        print(f"Error: {e}")
        error_message = str(e)
        if output_file:
            output_record = {
                "timestamp": datetime.now().isoformat(),
                "model": model.api_name,
                "num_characters": num_characters,
                "answer_only": answer_only,
                "question": question,
                "facts": [fact1, fact2],
                "needle_positions": positions,
                "gold_answers": answers,
                "gold_lines": positions,
                "response": None,
                "parsed_answer": None,
                "answer_correct": False,
                "lines_correct": False,
                "correct": False,
                "input_tokens": 0,
                "error": error_message,
            }
            if is_context_window_exceeded_error(e):
                output_record["error_kind"] = "context_length_exceeded"
            save_experiment_output(output_file, output_record)
        if is_context_window_exceeded_error(e):
            raise ContextLengthExceeded(error_message) from e
        return Result(characters=num_input_characters, tokens=0, correct=False, answer_correct=False, lines_correct=False)
    
    response, input_tokens = process_answer(result, model)

    # Some providers (notably Anthropic since ~mid-2026) silently return an
    # empty content block when a safety/content filter trips, with no error
    # surfaced. Treat empty responses as a soft error so they're excluded
    # from accuracy rather than counted as wrong answers.
    if not (response or "").strip():
        error_message = "empty_response"
        if output_file:
            output_record = {
                "timestamp": datetime.now().isoformat(),
                "model": model.api_name,
                "num_characters": num_characters,
                "answer_only": answer_only,
                "question": question,
                "facts": [fact1, fact2],
                "needle_positions": positions,
                "gold_answers": answers,
                "gold_lines": positions,
                "response": response,
                "parsed_answer": None,
                "answer_correct": False,
                "lines_correct": False,
                "correct": False,
                "input_tokens": input_tokens,
                "error": error_message,
                "error_kind": "empty_response",
            }
            save_experiment_output(output_file, output_record)
        return Result(characters=num_input_characters, tokens=input_tokens, correct=False, answer_correct=False, lines_correct=False)

    parsed_answer = parse_llm_json_output(response)
    if answer_only:
        # Same JSON parsing as EBG, but only the answer field is required.
        llm_answer = str((parsed_answer or {}).get("answer"))
        answer_correct = llm_answer.strip(".") in answers
        lines_correct = None  # not applicable
        correct = answer_correct  # joint == answer in answer-only mode
    else:
        answer_correct, lines_correct = evaluate_llm_output(parsed_answer, answer_key)
        correct = answer_correct and lines_correct

    if output_file:
        output_record = {
            "timestamp": datetime.now().isoformat(),
            "model": model.api_name,
            "num_characters": num_characters,
            "answer_only": answer_only,
            "question": question,
            "facts": [fact1, fact2],
            "needle_positions": positions,
            "gold_answers": answers,
            "gold_lines": positions,
            "response": response,
            "parsed_answer": parsed_answer,
            "answer_correct": answer_correct,
            "lines_correct": lines_correct,
            "correct": correct,
            "input_tokens": input_tokens,
            "error": None
        }
        save_experiment_output(output_file, output_record)
    
    return Result(characters=num_input_characters, tokens=input_tokens, correct=correct, 
                  answer_correct=answer_correct, lines_correct=lines_correct if lines_correct is not None else False)

def run_experiment_for_num_characters(clients, random_facts, num_characters, model, n, max_parallel=3, output_file=None, answer_only=False):
    """Run multiple experiments for a given context length."""
    def run_single():
        return run_experiment(clients, random_facts, num_characters, model, output_file, answer_only=answer_only)
    
    results = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel)
    try:
        futures = [executor.submit(run_single) for _ in range(n)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    except ContextLengthExceeded:
        _shutdown_executor_early(executor)
        raise
    else:
        executor.shutdown(wait=True)
    
    total = len(results)
    total_characters = sum(r.characters for r in results)
    total_tokens = sum(r.tokens for r in results)
    total_correct = sum(r.correct for r in results)
    answer_correct = sum(r.answer_correct for r in results)
    evidence_correct = sum(r.lines_correct for r in results)
    
    avg_tokens = total_tokens / total if total > 0 else 0
    avg_characters = total_characters / total if total > 0 else 0
    
    return {
        "characters": round(avg_characters),
        "tokens": round(avg_tokens),
        "evidence_accuracy": evidence_correct / total,
        "answer_accuracy": answer_correct / total,
        "total_accuracy": total_correct / total,
        "count": total,
    }

def run_experiments(
    model,
    character_lengths,
    random_facts,
    clients,
    trials=100,
    max_parallel=3,
    output_file=None,
    existing_results=None,
    verbose=True,
    answer_only=False,
):
    """
    Run a full experiment suite.
    
    Args:
        model: Model instance to use
        character_lengths: List of context lengths to test
        random_facts: List of random facts for context
        clients: Dict of API clients
        trials: Number of trials per context length
        max_parallel: Max parallel API calls
        output_file: Path to save outputs (optional)
        existing_results: Dict of existing results for resuming (optional)
        verbose: Print progress
    
    Returns:
        List of result dicts
    """
    if existing_results is None:
        existing_results = {}
    
    if verbose:
        print(f"Running experiments with {model.api_name}")
        print(f"Context lengths: {character_lengths}")
        print(f"Trials per length: {trials}")
        print(f"Max parallel: {max_parallel}")
        print()
        print_markdown_row(["Characters", "Tokens", "Evidence", "Answer", "Total", "Count"], header=True)
    
    all_results = []
    for num_characters in character_lengths:
        records_for_length = existing_results.get(num_characters, [])
        # Resume quota: only successful API completions count toward --trials
        existing_success_count = len(successful_records(records_for_length))
        remaining_trials = max(0, trials - existing_success_count)
        
        result = None
        try:
            if remaining_trials > 0:
                result = run_experiment_for_num_characters(
                    clients, random_facts, num_characters, model,
                    remaining_trials, max_parallel, output_file,
                    answer_only=answer_only,
                )
                
                # Merge with anything already on disk (including prior failed API rows)
                if output_file:
                    updated_results = load_existing_results(output_file)
                    all_records = updated_results.get(num_characters, [])
                    merged = compute_stats_from_records(all_records)
                    if merged is not None:
                        result = merged
                else:
                    result["count"] = remaining_trials
            else:
                result = compute_stats_from_records(records_for_length)
                if verbose and result:
                    failed_n = len(records_for_length) - existing_success_count
                    extra = f" ({failed_n} failed API calls in file)" if failed_n else ""
                    print(f"  (skipping {num_characters:,} chars - already have {existing_success_count} successful trials{extra})")
        except ContextLengthExceeded as e:
            # Flush stats for this length (partial batch may be on disk)
            if output_file:
                updated_results = load_existing_results(output_file)
                all_records = updated_results.get(num_characters, [])
                merged = compute_stats_from_records(all_records)
                if merged is not None:
                    result = merged
            if result:
                all_results.append(result)
                if verbose:
                    print_markdown_row([
                        result["characters"],
                        result["tokens"],
                        f"{result['evidence_accuracy']:.2f}",
                        f"{result['answer_accuracy']:.2f}",
                        f"{result['total_accuracy']:.2f}",
                        result.get("count", trials),
                    ])
            if verbose:
                print(
                    "\nStopping: prompt exceeded the model context window. "
                    "No further context lengths will be run."
                )
                print(f"  ({e})")
            raise
        
        if result:
            all_results.append(result)
            if verbose:
                print_markdown_row([
                    result["characters"],
                    result["tokens"],
                    f"{result['evidence_accuracy']:.2f}",
                    f"{result['answer_accuracy']:.2f}",
                    f"{result['total_accuracy']:.2f}",
                    result.get("count", trials),
                ])
    
    return all_results


