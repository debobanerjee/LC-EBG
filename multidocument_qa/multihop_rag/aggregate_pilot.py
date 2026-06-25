"""Aggregate the latest Experiment A pilot run per (model, mode)."""

import glob
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "experiment_outputs")

PATTERN = re.compile(r"multihop_rag_(?P<model>.+?)(?P<mode>_answer_only)?_(?P<ts>\d{8}_\d{6})\.jsonl$")


def latest_per_config() -> dict[tuple[str, str], str]:
    latest: dict[tuple[str, str], tuple[str, str]] = {}
    for p in glob.glob(os.path.join(OUT_DIR, "multihop_rag_*.jsonl")):
        m = PATTERN.search(os.path.basename(p))
        if not m:
            continue
        key = (m["model"], "answer_only" if m["mode"] else "ebg")
        if key not in latest or m["ts"] > latest[key][1]:
            latest[key] = (p, m["ts"])
    return {k: v[0] for k, v in latest.items()}


def summarize(path: str) -> dict:
    recs = [json.loads(l) for l in open(path)]
    n_ok = sum(1 for r in recs if not r.get("error"))
    ok = [r for r in recs if not r.get("error")]
    ans = sum(1 for r in ok if r["answer_correct"]) / max(1, len(ok))
    ev = sum(1 for r in ok if r["evidence_strict"]) / max(1, len(ok))
    jt = sum(1 for r in ok if r["joint"]) / max(1, len(ok))
    by_type: dict[str, list] = defaultdict(list)
    for r in ok:
        by_type[r["question_type"]].append(r)
    return {
        "n_total": len(recs),
        "n_ok": n_ok,
        "Ans": ans,
        "Evid": ev,
        "Joint": jt,
        "by_type": {
            t: {
                "n": len(rs),
                "Ans": sum(1 for r in rs if r["answer_correct"]) / len(rs),
                "Evid": sum(1 for r in rs if r["evidence_strict"]) / len(rs),
                "Joint": sum(1 for r in rs if r["joint"]) / len(rs),
            }
            for t, rs in sorted(by_type.items())
        },
    }


def main() -> None:
    paths = latest_per_config()
    rows = []
    for (model, mode), p in sorted(paths.items()):
        s = summarize(p)
        rows.append((model, mode, s, p))

    print(f"{'Model':<35}{'Mode':<14}{'n':>4}{'Ans':>7}{'Evid':>7}{'Joint':>7}")
    print("-" * 80)
    for model, mode, s, _ in rows:
        print(f"{model:<35}{mode:<14}{s['n_ok']:>4}"
              f"{s['Ans']*100:>6.1f}%{s['Evid']*100:>6.1f}%{s['Joint']*100:>6.1f}%")

    print("\nBy question type")
    print("-" * 80)
    for model, mode, s, _ in rows:
        print(f"\n  {model}  /  {mode}")
        print(f"    {'type':<22}{'n':>4}{'Ans':>7}{'Evid':>7}{'Joint':>7}")
        for t, b in s["by_type"].items():
            print(f"    {t:<22}{b['n']:>4}"
                  f"{b['Ans']*100:>6.1f}%{b['Evid']*100:>6.1f}%{b['Joint']*100:>6.1f}%")

    print("\nSource files:")
    for _, _, _, p in rows:
        print(f"  {p}")


if __name__ == "__main__":
    main()
