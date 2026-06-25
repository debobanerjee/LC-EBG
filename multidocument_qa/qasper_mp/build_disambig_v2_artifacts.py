"""Build artifacts for v2 disambiguated questions (short author / title cues).

Writes:
  - processed/queries_dev_d10_disambig_v2.jsonl   (LC queries with rewritten text)
  - retrieval_results/dev_large_disambig_v2/      (re-embedded queries)
  - processed/rag_disambig_v2/                    (RAG haystacks for K=3,10)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
load_dotenv(ROOT / ".env")

DISAMBIG = HERE / "processed" / "disambiguated_questions_v2.json"
LC_QUERIES = HERE / "processed" / "queries_dev_d10.jsonl"
LC_OUT = HERE / "processed" / "queries_dev_d10_disambig_v2.jsonl"
RETR_SRC = HERE / "retrieval_results" / "dev_large"
RETR_DST = HERE / "retrieval_results" / "dev_large_disambig_v2"
RAG_OUT = HERE / "processed" / "rag_disambig_v2"
EMB_MODEL = "text-embedding-3-large"


def embed(client: OpenAI, texts: list[str]) -> np.ndarray:
    out = []
    B = 64
    for i in range(0, len(texts), B):
        chunk = texts[i:i + B]
        r = client.embeddings.create(model=EMB_MODEL, input=chunk)
        out.extend([np.asarray(d.embedding, dtype=np.float32) for d in r.data])
    arr = np.vstack(out)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
    return arr


def main() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    disambig = json.load(open(DISAMBIG))
    pilot_qids = list(disambig.keys())
    print(f"v2 disambiguated questions: {len(pilot_qids)}")

    queries = [disambig[q]["rewritten"] for q in pilot_qids]
    print("Embedding v2 queries ...")
    q_emb = embed(client, queries)
    print(f"  embeddings shape={q_emb.shape}")

    RETR_DST.mkdir(parents=True, exist_ok=True)
    np.save(RETR_DST / "query_emb.npy", q_emb)
    with open(RETR_DST / "queries.jsonl", "w") as f:
        for qid, qtext in zip(pilot_qids, queries):
            f.write(json.dumps({"qid": qid, "query": qtext}) + "\n")
    for fname in ("paragraphs.jsonl", "paragraph_emb.npy"):
        dst = RETR_DST / fname
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(RETR_SRC / fname)
    print(f"Wrote retrieval dir {RETR_DST}")

    n_lc = 0
    with open(LC_OUT, "w") as fout:
        for line in open(LC_QUERIES):
            r = json.loads(line)
            if r["qid"] in disambig:
                r["query_original"] = r["query"]
                r["query"] = disambig[r["qid"]]["rewritten"]
                r["cue_kind"] = disambig[r["qid"]]["cue_kind"]
                r["reference"] = disambig[r["qid"]]["reference"]
                fout.write(json.dumps(r) + "\n")
                n_lc += 1
    print(f"Wrote LC queries v2: {n_lc} -> {LC_OUT}")

    cmd = [
        sys.executable,
        str(HERE / "build_rag_haystack.py"),
        "--qids-from", str(LC_OUT),
        "--lc-queries", str(LC_OUT),
        "--retrieval", str(RETR_DST),
        "--ks", "3", "10",
        "--out-dir", str(RAG_OUT),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=HERE)


if __name__ == "__main__":
    main()
