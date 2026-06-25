"""Aggregate the v2 disambiguation sweep (short author/title cues).

Reads experiment_outputs_disambig_v2/{lc, flat_pool11_k3, struct_pool11_k3,
flat_full_k10}/qasper_mp_*.jsonl and prints a table of Ans-F1 / Evid-F1
for every (model, mode, condition).

Also prints v1 vs v2 deltas if both sweeps are present.
"""

import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2_ROOT = HERE / "experiment_outputs_disambig_v2"
V2FC_ROOT = HERE / "experiment_outputs_disambig_v2_fixcite"
V1_ROOT = HERE / "experiment_outputs_disambig"
RAG_V2_PROC = HERE / "processed" / "rag_disambig_v2"
RAG_V1_PROC = HERE / "processed" / "rag_disambig"

PATTERN = re.compile(
    r"qasper_mp_(?P<model>.+?)(?P<mode>_answer_only)?_(?P<ts>\d{8}_\d{6})\.jsonl$")

CONDITIONS = ["lc", "flat_pool11_k3", "struct_pool11_k3", "flat_full_k10"]
COND_LABEL = {
    "lc":              "LC (full)",
    "flat_pool11_k3":  "flat_pool11@3",
    "struct_pool11_k3":"struct_pool11@3",
    "flat_full_k10":   "flat_full@10",
}
MODEL_ORDER = ["gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-flash-preview"]
MODEL_LABEL = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-5-20250929": "Sonnet-4.5",
    "gemini-3-flash-preview": "Gemini-3-Fl",
}


def latest_in(folder: Path) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], tuple[Path, str]] = {}
    for p in folder.glob("qasper_mp_*.jsonl"):
        m = PATTERN.search(p.name)
        if not m:
            continue
        key = (m["model"], "answer_only" if m["mode"] else "ebg")
        if key not in out or m["ts"] > out[key][1]:
            out[key] = (p, m["ts"])
    return {k: v[0] for k, v in out.items()}


def summarize(path: Path) -> dict:
    recs = [json.loads(l) for l in open(path)]
    ok = [r for r in recs if not r.get("error")]
    ao = bool(recs and recs[0].get("answer_only"))
    n = len(ok)
    if n == 0:
        return {"n": 0, "Ans": 0.0, "Evid": float("nan"), "ao": ao}
    af = sum(r["answer_f1"] for r in ok) / n
    ef = (sum(r["evidence_f1"] for r in ok) / n) if not ao else float("nan")
    return {"n": n, "Ans": af, "Evid": ef, "ao": ao}


def gather(root: Path) -> dict:
    """{ (model, mode, condition): summary } """
    out = {}
    for cond in CONDITIONS:
        folder = root / cond
        if not folder.is_dir():
            continue
        for (model, mode), path in latest_in(folder).items():
            out[(model, mode, cond)] = summarize(path)
    return out


def coverage(rag_proc_dir: Path) -> dict:
    cov = defaultdict(lambda: {"covered": 0, "tot": 0, "fi": 0, "ft": 0})
    for v in ("flat_pool11", "struct_pool11", "flat_full", "struct_full"):
        for K in (3, 10):
            p = rag_proc_dir / f"queries_{v}_k{K}.jsonl"
            if not p.exists():
                continue
            pool = "pool11" if v.endswith("pool11") else "full"
            for line in open(p):
                r = json.loads(line)
                key = (pool, K)
                cov[key]["tot"] += 1
                cov[key]["fi"] += r["needles_in_topk"]
                cov[key]["ft"] += r["needles_total"]
                if r["needles_in_topk"] == r["needles_total"]:
                    cov[key]["covered"] += 1
    return cov


def print_table(title: str, summaries: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"{'Model':<12}{'Mode':<13}{'Condition':<22}{'n':>4}{'Ans-F1':>10}{'Evid-F1':>10}")
    print("-" * 71)
    for mode in ("ebg", "answer_only"):
        for model in MODEL_ORDER:
            for cond in CONDITIONS:
                k = (model, mode, cond)
                if k not in summaries:
                    continue
                s = summaries[k]
                ev = "-" if s["ao"] else f"{s['Evid']:.3f}"
                print(f"{MODEL_LABEL[model]:<12}{mode:<13}"
                      f"{COND_LABEL[cond]:<22}{s['n']:>4}"
                      f"{s['Ans']:>10.3f}{ev:>10}")


def print_delta(v1: dict, v2: dict) -> None:
    print("\n=== v1 -> v2 deltas (Ans-F1, EBG mode only) ===")
    print(f"{'Model':<12}{'Condition':<22}{'v1':>8}{'v2':>8}{'delta':>8}")
    print("-" * 60)
    for model in MODEL_ORDER:
        for cond in CONDITIONS:
            a = v1.get((model, "ebg", cond))
            b = v2.get((model, "ebg", cond))
            if not a or not b:
                continue
            d = b["Ans"] - a["Ans"]
            arrow = "↓" if d < -1e-4 else ("↑" if d > 1e-4 else " ")
            print(f"{MODEL_LABEL[model]:<12}{COND_LABEL[cond]:<22}"
                  f"{a['Ans']:>8.3f}{b['Ans']:>8.3f}{d:>+7.3f}{arrow}")


def print_coverage(label: str, cov: dict) -> None:
    print(f"\nNeedle coverage ({label}):")
    print(f"  {'pool':<7}{'K':>5}{'all-needles q%':>20}{'fact recall':>14}")
    seen = set()
    for v in ("flat_pool11", "flat_full"):
        pool = "pool11" if v.endswith("pool11") else "full"
        for K in (3, 10):
            key = (pool, K)
            if key in seen:
                continue
            seen.add(key)
            s = cov.get(key)
            if not s or s["tot"] == 0:
                continue
            agq = s["covered"] / s["tot"]
            fr = s["fi"] / s["ft"] if s["ft"] else 0.0
            print(f"  {pool:<7}{K:>5}{agq*100:>19.1f}%{fr*100:>13.1f}%")


def main() -> None:
    v2 = gather(V2_ROOT)
    print_table("v2 (short cues, original prompt)", v2)

    if V2FC_ROOT.is_dir():
        v2fc = gather(V2FC_ROOT)
        if v2fc:
            print_table("v2-fixcite (no <title>/<abstract>/... citations, "
                        "discontinuity ... markers)", v2fc)
            print("\n=== v2 -> v2-fixcite deltas (Ans-F1, EBG) ===")
            print(f"{'Model':<12}{'Condition':<22}{'v2':>8}{'fixcite':>10}{'delta':>8}")
            print("-" * 60)
            for model in MODEL_ORDER:
                for cond in CONDITIONS:
                    a = v2.get((model, "ebg", cond))
                    b = v2fc.get((model, "ebg", cond))
                    if not a or not b:
                        continue
                    d = b["Ans"] - a["Ans"]
                    arrow = "↓" if d < -1e-4 else ("↑" if d > 1e-4 else " ")
                    print(f"{MODEL_LABEL[model]:<12}{COND_LABEL[cond]:<22}"
                          f"{a['Ans']:>8.3f}{b['Ans']:>10.3f}{d:>+7.3f}{arrow}")
            print("\n=== v2 -> v2-fixcite deltas (Evid-F1, EBG) ===")
            print(f"{'Model':<12}{'Condition':<22}{'v2':>8}{'fixcite':>10}{'delta':>8}")
            print("-" * 60)
            for model in MODEL_ORDER:
                for cond in CONDITIONS:
                    a = v2.get((model, "ebg", cond))
                    b = v2fc.get((model, "ebg", cond))
                    if not a or not b or a["ao"] or b["ao"]:
                        continue
                    d = b["Evid"] - a["Evid"]
                    arrow = "↓" if d < -1e-4 else ("↑" if d > 1e-4 else " ")
                    print(f"{MODEL_LABEL[model]:<12}{COND_LABEL[cond]:<22}"
                          f"{a['Evid']:>8.3f}{b['Evid']:>10.3f}{d:>+7.3f}{arrow}")

    if V1_ROOT.is_dir():
        v1 = gather(V1_ROOT)
        if v1:
            print_table("v1 disambiguation sweep (verbose cues, baseline)", v1)
            print_delta(v1, v2)

    print_coverage("v2", coverage(RAG_V2_PROC))
    if RAG_V1_PROC.is_dir():
        print_coverage("v1", coverage(RAG_V1_PROC))


if __name__ == "__main__":
    main()
