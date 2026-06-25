"""Break down v2-fixcite + v2 results by answerability.

Reports judge & Ans-F1 separately for answerable questions (extractive,
abstractive, boolean) vs unanswerable (question_type == 'none'), broken down
by condition.

Also reports refusal rates: how often does each setting answer 'Unanswerable'?
"""

import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2FC = HERE / "experiment_outputs_disambig_v2_fixcite"
V2 = HERE / "experiment_outputs_disambig_v2"

PATTERN = re.compile(
    r"qasper_mp_(?P<model>.+?)(?P<mode>_answer_only)?_(?P<ts>\d{8}_\d{6})\.judged\.jsonl$"
)
MODEL_LABEL = {
    "gpt-5": "GPT-5",
    "gpt-5.5": "GPT-5.5",
    "claude-opus-4-6": "Opus-4.6",
    "claude-sonnet-4-5-20250929": "Sonnet-4.5",
    "gemini-3-flash-preview": "Gemini-3-Fl",
}
MODEL_ORDER = list(MODEL_LABEL)
CONDITIONS = ["lc", "flat_pool11_k3", "struct_pool11_k3", "flat_full_k10"]
COND_LABEL = {
    "lc": "LC",
    "flat_pool11_k3": "flat_pool11@3",
    "struct_pool11_k3": "struct_pool11@3",
    "flat_full_k10": "flat_full@10",
}


def latest(folder: Path):
    out = {}
    for p in folder.glob("*.judged.jsonl"):
        m = PATTERN.search(p.name)
        if not m or m["mode"]:
            continue
        key = m["model"]
        if key not in out or m["ts"] > out[key][1]:
            out[key] = (p, m["ts"])
    return {k: v[0] for k, v in out.items()}


def is_refusal(pred: str) -> bool:
    if not pred:
        return True
    return pred.strip().lower().startswith("unanswerable")


def stats(path: Path):
    ans = {"n": 0, "judge_sum": 0.0, "f1_sum": 0.0, "refuse": 0}
    unans = {"n": 0, "judge_sum": 0.0, "f1_sum": 0.0, "refuse": 0}
    for line in open(path):
        r = json.loads(line)
        if r.get("error"):
            continue
        qt = r.get("question_type", "")
        bucket = unans if qt == "none" else ans
        bucket["n"] += 1
        bucket["judge_sum"] += r.get("judge_score") or 0
        bucket["f1_sum"] += r.get("answer_f1") or 0
        pred = (r.get("parsed") or {}).get("answer", "") or r.get("pred_answer", "")
        if is_refusal(pred):
            bucket["refuse"] += 1
    return ans, unans


def fmt_pct(num, denom):
    return f"{num/denom*100:>5.1f}%" if denom else "    -"


def fmt_score(s, n):
    return f"{s/n:>5.3f}" if n else "    -"


def report(root: Path, label: str):
    print(f"\n=== {label} (root: {root.name}) ===")
    print(f"{'Model':<12}{'Cond':<20}"
          f"{'Ans-n':>6}{'A-Judge':>9}{'A-F1':>7}{'A-refuse%':>11}"
          f"{'Un-n':>6}{'Un-Judge':>10}{'Un-F1':>8}{'Un-refuse%':>12}")
    print("-" * 102)
    for cond in CONDITIONS:
        folder = root / cond
        if not folder.is_dir():
            continue
        files = latest(folder)
        for model in MODEL_ORDER:
            if model not in files:
                continue
            a, u = stats(files[model])
            print(f"{MODEL_LABEL[model]:<12}{COND_LABEL[cond]:<20}"
                  f"{a['n']:>6}{fmt_score(a['judge_sum'],a['n']):>9}"
                  f"{fmt_score(a['f1_sum'],a['n']):>7}"
                  f"{fmt_pct(a['refuse'],a['n']):>11}"
                  f"{u['n']:>6}{fmt_score(u['judge_sum'],u['n']):>10}"
                  f"{fmt_score(u['f1_sum'],u['n']):>8}"
                  f"{fmt_pct(u['refuse'],u['n']):>12}")


def main():
    report(V2FC, "v2-fixcite")
    report(V2, "v2 (original prompt)")


if __name__ == "__main__":
    main()
