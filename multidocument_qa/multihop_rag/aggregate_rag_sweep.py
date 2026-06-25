"""
Aggregate the RAG sweep + LC pilot into one LC-vs-RAG comparison table.

For each (model, mode in {EBG, answer-only}, condition in {LC, RAG@K})
we report:
  n_ok, Ans, Evid (strict), Joint  -- where applicable.

We also report the per-K needle-coverage diagnostics so the RAG numbers can
be read alongside an upper bound on what RAG could possibly achieve.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


PATTERN_LC = re.compile(
    r"multihop_rag_(?P<model>.+?)(?P<mode>_answer_only)?_(?P<ts>\d{8}_\d{6})\.jsonl$"
)


def load_jsonl(p: str) -> list[dict]:
    return [json.loads(l) for l in open(p)]


def latest(paths: list[str]) -> str | None:
    if not paths:
        return None
    return sorted(paths)[-1]


def lc_files(out_dir: str) -> dict[tuple[str, str], str]:
    """Return {(model, mode): path} of latest LC pilot files."""
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in glob.glob(os.path.join(out_dir, "multihop_rag_*.jsonl")):
        m = PATTERN_LC.search(os.path.basename(p))
        if not m:
            continue
        key = (m["model"], "answer_only" if m["mode"] else "ebg")
        out[key].append(p)
    return {k: latest(v) for k, v in out.items()}


def rag_files(out_root: str) -> dict[tuple[str, str, int], str]:
    """Return {(model, mode, K): path} of latest RAG files."""
    out: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(out_root, "rag_k*"))):
        K = int(re.search(r"rag_k(\d+)$", d).group(1))
        for p in glob.glob(os.path.join(d, "multihop_rag_*.jsonl")):
            m = PATTERN_LC.search(os.path.basename(p))
            if not m:
                continue
            key = (m["model"], "answer_only" if m["mode"] else "ebg", K)
            out[key].append(p)
    return {k: latest(v) for k, v in out.items()}


def score(path: str) -> dict:
    recs = load_jsonl(path)
    ok = [r for r in recs if not r.get("error")]
    n = len(ok)
    if n == 0:
        return {"n": 0, "Ans": 0.0, "Evid": 0.0, "Joint": 0.0}
    return {
        "n": n,
        "Ans": sum(1 for r in ok if r["answer_correct"]) / n,
        "Evid": sum(1 for r in ok if r["evidence_strict"]) / n,
        "Joint": sum(1 for r in ok if r["joint"]) / n,
    }


MODEL_ORDER = ["gpt-5", "claude-sonnet-4-5-20250929", "gemini-3-flash-preview"]
MODEL_LABEL = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-5-20250929": "Sonnet-4.5",
    "gemini-3-flash-preview": "Gemini-3-Fl",
}


def print_table(rows: list[tuple]) -> None:
    print(f"{'Model':<13}{'Mode':<13}{'Condition':<14}{'n':>4}"
          f"{'Ans':>8}{'Evid':>8}{'Joint':>8}")
    print("-" * 72)
    for r in rows:
        model, mode, cond, s = r
        ev = "-" if mode == "answer_only" else f"{s['Evid']*100:>6.1f}%"
        print(f"{model:<13}{mode:<13}{cond:<14}{s['n']:>4}"
              f"{s['Ans']*100:>7.1f}%{ev:>8}{s['Joint']*100:>7.1f}%")


def coverage_table(rag_proc_dir: str, ks: list[int]) -> None:
    print("\nNeedle coverage (50 pilot queries, retriever = text-embedding-3-large + tokens256):")
    print(f"  {'K':>4}{'all-needles q%':>18}{'fact recall':>14}")
    for K in ks:
        p = os.path.join(rag_proc_dir, f"queries_rag_k{K}.jsonl")
        if not os.path.exists(p):
            continue
        recs = load_jsonl(p)
        tot = sum(r["needles_total"] for r in recs)
        intk = sum(r["needles_in_topk"] for r in recs)
        all_in = sum(1 for r in recs if r["needles_in_topk"] == r["needles_total"])
        n = len(recs)
        print(f"  {K:>4}{all_in/n*100:>17.1f}%{intk/tot*100:>13.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "experiment_outputs"))
    ap.add_argument("--rag-proc-dir", default=os.path.join(HERE, "processed/rag"))
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[4, 10, 25, 50, 100, 200])
    args = ap.parse_args()

    lc = lc_files(args.out_dir)
    rag = rag_files(args.out_dir)

    rows: list[tuple] = []
    for mode in ("ebg", "answer_only"):
        for model in MODEL_ORDER:
            p_lc = lc.get((model, mode))
            if p_lc:
                rows.append((MODEL_LABEL[model], mode, "LC (full)", score(p_lc)))
            for K in args.ks:
                p = rag.get((model, mode, K))
                if p:
                    rows.append((MODEL_LABEL[model], mode, f"RAG@{K}", score(p)))
        rows.append(("---", "", "", {"n": 0, "Ans": 0, "Evid": 0, "Joint": 0}))
    rows = [r for r in rows if r[0] != "---" or True]

    print_table(rows)
    coverage_table(args.rag_proc_dir, args.ks)


if __name__ == "__main__":
    main()
