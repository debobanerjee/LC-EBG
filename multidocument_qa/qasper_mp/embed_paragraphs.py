"""
Embed every text paragraph of every QASPER paper in a split plus every
query, with OpenAI's ``text-embedding-3-large``. Cached on disk so the
per-query RAG retrieval is a cheap dot product.

Outputs (under ``--out-dir``):
  paragraphs.jsonl       one record per paragraph
                         {paper_id, par_idx_in_paper, text}
  paragraph_emb.npy      (N_paragraphs, D) float32 embeddings
  queries.jsonl          one record per QA
                         {qid, gold_paper_id, query}
  query_emb.npy          (N_queries, D) float32 embeddings
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(REPO_ROOT, ".env"))

import openai  # noqa: E402


def embed_batches(client, texts: list[str], model: str,
                  batch_size: int = 256) -> np.ndarray:
    out: list[list[float]] = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(6):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                out.extend(d.embedding for d in resp.data)
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  retry {attempt+1} (batch {i//batch_size+1}): {e}; "
                      f"sleeping {wait}s", flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError(f"Embedding failed for batch starting at {i}")
        if ((i + batch_size) % (batch_size * 4) == 0) or (i + batch_size) >= total:
            print(f"  embedded {min(i+batch_size, total)}/{total}", flush=True)
    return np.asarray(out, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-path", default="dataset/qasper-dev-v0.3.json")
    ap.add_argument("--model", default="text-embedding-3-large")
    ap.add_argument("--out-dir", default="retrieval_results/dev_large")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    src = json.load(open(args.split_path))

    paragraphs: list[dict] = []
    queries: list[dict] = []
    for paper_id, paper in src.items():
        par_idx = 0
        for section in paper.get("full_text", []):
            for par in section.get("paragraphs", []):
                par = par.strip()
                if not par:
                    continue
                paragraphs.append({
                    "paper_id": paper_id,
                    "par_idx_in_paper": par_idx,
                    "text": par,
                })
                par_idx += 1
        for qa in paper["qas"]:
            queries.append({
                "qid": qa["question_id"],
                "gold_paper_id": paper_id,
                "query": qa["question"],
            })

    print(f"Loaded {len(paragraphs)} paragraphs, {len(queries)} queries "
          f"from {len(src)} papers")

    client = openai.OpenAI()

    par_emb_path = os.path.join(args.out_dir, "paragraph_emb.npy")
    par_meta_path = os.path.join(args.out_dir, "paragraphs.jsonl")
    if os.path.exists(par_emb_path) and os.path.exists(par_meta_path):
        print(f"Paragraph embeddings exist at {par_emb_path}; skipping.")
    else:
        print(f"Embedding {len(paragraphs)} paragraphs ({args.model})...")
        emb = embed_batches(client, [p["text"] for p in paragraphs], args.model)
        np.save(par_emb_path, emb)
        with open(par_meta_path, "w") as f:
            for p in paragraphs:
                f.write(json.dumps(p) + "\n")
        print(f"Wrote {par_emb_path}  shape={emb.shape}")

    q_emb_path = os.path.join(args.out_dir, "query_emb.npy")
    q_meta_path = os.path.join(args.out_dir, "queries.jsonl")
    if os.path.exists(q_emb_path) and os.path.exists(q_meta_path):
        print(f"Query embeddings exist at {q_emb_path}; skipping.")
    else:
        print(f"Embedding {len(queries)} queries ({args.model})...")
        emb = embed_batches(client, [q["query"] for q in queries], args.model)
        np.save(q_emb_path, emb)
        with open(q_meta_path, "w") as f:
            for q in queries:
                f.write(json.dumps(q) + "\n")
        print(f"Wrote {q_emb_path}  shape={emb.shape}")


if __name__ == "__main__":
    main()
