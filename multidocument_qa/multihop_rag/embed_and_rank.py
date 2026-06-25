"""
Experiment B: retrieval-position of MultiHop-RAG gold evidence.

For each non-null query we embed the query and every chunk of the MultiHop-RAG
corpus with an OpenAI embedding model, rank chunks by cosine similarity, and
record at what rank each gold-evidence ``fact`` first appears.

Pipeline:
    1. Chunk the corpus.
         - ``--chunker tokens256`` : 256-token windows, 10% overlap (matches the
           MultiHop-RAG paper, Table 3).
         - ``--chunker sentences`` : sentence-level segments (matches the line
           indexing used by ``build_haystack.py``).
    2. Embed all chunks + all queries with ``--embed-model`` (default
       ``text-embedding-3-small``). Embeddings are cached to disk keyed by
       (model, chunker) so re-runs are free.
    3. For each query, compute cosine similarity to every chunk, argsort, and
       locate the rank of every chunk whose text contains a gold ``fact``
       substring. Persist one record per (query, fact).
    4. Print aggregate Hits@k, MAP@10 and MRR@10, matching the metrics in
       ``retrieval_evaluate.py``.

Outputs land under ``retrieval_results/{embed_model}_{chunker}/`` as:

    chunks.jsonl       # chunk_id, doc_idx, title, text
    fact_ranks.jsonl   # qid, question_type, fact_idx, gold_chunk_ids, best_rank, best_score
    summary.json       # Hits@k, MAP@10, MRR@10 (overall and per question type)

Cost: ~6M tokens at $0.02/M ≈ $0.12 for the default config.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import openai
import tiktoken
from dotenv import load_dotenv

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))


# --- chunking -------------------------------------------------------------

SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z0-9\"\'])")


def chunk_tokens(body: str, encoder, target_tokens: int = 256, overlap: float = 0.1) -> list[str]:
    """256-token windows with 10% overlap on raw token ids (matches the
    MultiHop-RAG paper's ``SentenceSplitter(chunk_size=256)`` setting closely
    enough for retrieval-rank comparison).
    """
    ids = encoder.encode(body)
    if not ids:
        return []
    step = max(1, int(target_tokens * (1.0 - overlap)))
    out = []
    i = 0
    while i < len(ids):
        window = ids[i : i + target_tokens]
        out.append(encoder.decode(window))
        if i + target_tokens >= len(ids):
            break
        i += step
    return out


def chunk_sentences(body: str) -> list[str]:
    body = body.replace("\r", " ")
    body = re.sub(r"\n{2,}", "\n", body)
    out = []
    for paragraph in body.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for s in SENT_SPLIT_RE.split(paragraph):
            s = s.strip()
            if s:
                out.append(s)
    return out


def build_chunks(corpus: list[dict], chunker: str, encoder) -> list[dict]:
    """Return list of {chunk_id, doc_idx, title, text}."""
    chunks: list[dict] = []
    for doc_idx, doc in enumerate(corpus):
        body = doc.get("body", "") or ""
        if chunker == "tokens256":
            parts = chunk_tokens(body, encoder, 256, 0.1)
        elif chunker == "sentences":
            parts = chunk_sentences(body)
        else:
            raise ValueError(chunker)
        for p in parts:
            chunks.append({
                "chunk_id": len(chunks),
                "doc_idx": doc_idx,
                "title": doc["title"],
                "text": p,
            })
    return chunks


# --- embedding ------------------------------------------------------------


def embed_texts(client: openai.OpenAI, texts: list[str], model: str, batch_size: int = 256) -> np.ndarray:
    """Embed ``texts`` in batches and L2-normalize so cosine sim becomes
    a plain dot product later."""
    out = np.zeros((len(texts), 1536 if "small" in model else 3072), dtype=np.float32)
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        # OpenAI replaces empty strings with a single space silently in some
        # SDKs; do it ourselves to avoid surprises.
        batch = [t if t.strip() else " " for t in batch]
        # Truncate to 8192 tokens (the model's input cap) by characters as a
        # cheap safeguard; sentence/256-token chunks are well below this.
        batch = [t[:30000] for t in batch]
        attempts = 0
        while True:
            try:
                resp = client.embeddings.create(model=model, input=batch)
                break
            except Exception as e:
                attempts += 1
                if attempts > 5:
                    raise
                wait = min(30, 2 ** attempts)
                print(f"  embed retry {attempts} after error: {e!s:.100s} (sleep {wait}s)")
                time.sleep(wait)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # Normalize.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        out[start : start + len(batch)] = vecs
        if start % (batch_size * 8) == 0:
            print(f"  embedded {start + len(batch)}/{n}")
    return out


# --- ranking --------------------------------------------------------------


def normalize_for_match(s: str) -> str:
    """Match the MultiHop-RAG ``retrieval_evaluate.py`` normalization:
    strip whitespace + newlines so 'fact in chunk' is robust to wrapping."""
    return s.replace(" ", "").replace("\n", "")


def find_gold_chunk_ids(fact: str, chunks_norm: list[str]) -> list[int]:
    """Return all chunk ids whose normalized text contains the fact."""
    f = normalize_for_match(fact)
    if not f:
        return []
    return [i for i, c in enumerate(chunks_norm) if f in c]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(HERE, "dataset", "corpus.json"))
    ap.add_argument("--queries", default=os.path.join(HERE, "dataset", "MultiHopRAG.json"))
    ap.add_argument("--chunker", choices=("tokens256", "sentences"), default="tokens256")
    ap.add_argument("--embed-model", default="text-embedding-3-small",
                    choices=("text-embedding-3-small", "text-embedding-3-large"))
    ap.add_argument("--out-dir", default=None,
                    help="Default: retrieval_results/{embed_model}_{chunker}/")
    ap.add_argument("--limit-queries", type=int, default=None,
                    help="Run only the first N non-null queries (pilot mode).")
    args = ap.parse_args()

    if args.out_dir is None:
        safe_model = args.embed_model.replace("/", "_")
        args.out_dir = os.path.join(HERE, "retrieval_results", f"{safe_model}_{args.chunker}")
    os.makedirs(args.out_dir, exist_ok=True)

    # --- 1. chunk corpus
    corpus = json.load(open(args.corpus))
    queries = json.load(open(args.queries))
    enc = tiktoken.get_encoding("cl100k_base")
    print(f"Corpus: {len(corpus)} docs. Chunking with '{args.chunker}'...")
    chunks = build_chunks(corpus, args.chunker, enc)
    print(f"  {len(chunks):,} chunks total")

    chunks_path = os.path.join(args.out_dir, "chunks.jsonl")
    with open(chunks_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    # --- 2. embeddings (cached)
    chunk_emb_path = os.path.join(args.out_dir, "chunk_embeddings.npy")
    query_emb_path = os.path.join(args.out_dir, "query_embeddings.npy")
    query_ids_path = os.path.join(args.out_dir, "query_ids.json")

    client = openai.OpenAI()

    if os.path.exists(chunk_emb_path):
        print(f"Loading cached chunk embeddings from {chunk_emb_path}")
        chunk_emb = np.load(chunk_emb_path)
        assert chunk_emb.shape[0] == len(chunks), \
            f"cached chunk embedding count {chunk_emb.shape[0]} != current chunks {len(chunks)}; delete cache to rebuild"
    else:
        print(f"Embedding {len(chunks):,} chunks with {args.embed_model}...")
        chunk_emb = embed_texts(client, [c["text"] for c in chunks], args.embed_model)
        np.save(chunk_emb_path, chunk_emb)
        print(f"  saved {chunk_emb_path}")

    # Queries: filter null + obey --limit-queries
    qrecords = []
    for qid, q in enumerate(queries):
        if q.get("question_type") == "null_query":
            continue
        qrecords.append((qid, q))
        if args.limit_queries and len(qrecords) >= args.limit_queries:
            break
    print(f"Embedding {len(qrecords)} queries...")

    if os.path.exists(query_emb_path) and os.path.exists(query_ids_path):
        cached_ids = json.load(open(query_ids_path))
        if cached_ids == [qid for qid, _ in qrecords]:
            print(f"Loading cached query embeddings from {query_emb_path}")
            query_emb = np.load(query_emb_path)
        else:
            query_emb = None
    else:
        query_emb = None
    if query_emb is None:
        query_emb = embed_texts(client, [q["query"] for _, q in qrecords], args.embed_model)
        np.save(query_emb_path, query_emb)
        with open(query_ids_path, "w") as f:
            json.dump([qid for qid, _ in qrecords], f)
        print(f"  saved {query_emb_path}")

    # --- 3. rank gold facts per query
    print("Pre-normalizing chunk texts for substring match...")
    chunks_norm = [normalize_for_match(c["text"]) for c in chunks]

    # Pre-build: for each gold fact across all queries, which chunk ids contain
    # it? Many facts repeat across queries (paper had ~3 facts/query average,
    # 2255 queries) -> cache to save time.
    fact_to_chunks: dict[str, list[int]] = {}

    print("Computing similarities and ranks...")
    fact_records: list[dict] = []
    n_facts_unmapped = 0
    t0 = time.time()
    # chunk_emb is (N, d), query_emb is (Q, d). We compute sims one query at a time
    # to keep memory bounded but vectorized.
    for i, (qid, q) in enumerate(qrecords):
        qv = query_emb[i]  # (d,)
        sims = chunk_emb @ qv  # (N,)
        # argsort descending; rank = position (1-indexed).
        order = np.argsort(-sims, kind="stable")
        rank_of_chunk = np.empty(len(chunks), dtype=np.int32)
        rank_of_chunk[order] = np.arange(1, len(chunks) + 1, dtype=np.int32)

        for fact_idx, ev in enumerate(q.get("evidence_list", [])):
            fact = ev["fact"]
            cand = fact_to_chunks.get(fact)
            if cand is None:
                cand = find_gold_chunk_ids(fact, chunks_norm)
                fact_to_chunks[fact] = cand
            if not cand:
                n_facts_unmapped += 1
                continue
            ranks = rank_of_chunk[cand]
            best = int(ranks.min())
            best_id = int(cand[int(ranks.argmin())])
            fact_records.append({
                "qid": qid,
                "question_type": q.get("question_type"),
                "fact_idx": fact_idx,
                "fact": fact,
                "gold_chunk_ids": cand,
                "best_rank": best,
                "best_score": float(sims[best_id]),
            })

        if (i + 1) % 200 == 0:
            print(f"  ranked {i+1}/{len(qrecords)} queries  ({time.time()-t0:.1f}s)")

    facts_path = os.path.join(args.out_dir, "fact_ranks.jsonl")
    with open(facts_path, "w") as f:
        for r in fact_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {facts_path}  ({len(fact_records):,} facts; "
          f"{n_facts_unmapped} unmapped facts skipped)")

    # --- 4. aggregate metrics
    # Hits@k requires the *query-level* notion: all gold facts retrieved in top-k.
    by_query: dict[int, list[dict]] = defaultdict(list)
    for r in fact_records:
        by_query[r["qid"]].append(r)

    def metrics_for(rs_per_q: dict[int, list[dict]]) -> dict:
        Q = len(rs_per_q)
        if Q == 0:
            return {"n_queries": 0}
        hits_k = {1: 0, 4: 0, 10: 0, 50: 0, 100: 0, 500: 0}
        all_hits_k = {1: 0, 4: 0, 10: 0, 50: 0, 100: 0, 500: 0}
        ranks_first = []
        ranks_worst = []
        map10 = 0.0
        mrr10 = 0.0
        for qid, recs in rs_per_q.items():
            best_ranks = [r["best_rank"] for r in recs]
            for k in hits_k:
                if any(b <= k for b in best_ranks):
                    hits_k[k] += 1
                if all(b <= k for b in best_ranks):
                    all_hits_k[k] += 1
            ranks_first.append(min(best_ranks))
            ranks_worst.append(max(best_ranks))
            # MAP@10
            sorted_ranks = sorted(best_ranks)
            ap = 0.0
            for j, r in enumerate(sorted_ranks, start=1):
                if r <= 10:
                    ap += j / r
            map10 += ap / min(len(sorted_ranks), 10)
            # MRR@10: first relevant rank
            first = min(best_ranks)
            mrr10 += (1.0 / first) if first <= 10 else 0.0
        return {
            "n_queries": Q,
            "Hits@k (any gold)": {k: round(v / Q, 4) for k, v in hits_k.items()},
            "Hits@k (all gold)": {k: round(v / Q, 4) for k, v in all_hits_k.items()},
            "MAP@10": round(map10 / Q, 4),
            "MRR@10": round(mrr10 / Q, 4),
            "best_rank (median, mean)": (int(np.median(ranks_first)), float(np.mean(ranks_first))),
            "worst_rank (median, mean)": (int(np.median(ranks_worst)), float(np.mean(ranks_worst))),
        }

    overall = metrics_for(by_query)
    by_type: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in fact_records:
        by_type[r["question_type"]][r["qid"]].append(r)
    per_type = {t: metrics_for(d) for t, d in by_type.items()}
    summary = {
        "config": {
            "embed_model": args.embed_model,
            "chunker": args.chunker,
            "n_chunks": len(chunks),
            "n_queries": len(qrecords),
            "n_facts_total": len(fact_records),
            "n_facts_unmapped": n_facts_unmapped,
        },
        "overall": overall,
        "per_question_type": per_type,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
