# ragwithtopk/rag/retriever.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ragwithtopk.embeddings.openai_embedder import OpenAIEmbedder
from ragwithtopk.vectorstore.qdrant_store import QdrantStore
from ragwithtopk.utils.io import write_json


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class Retriever:
    embedder: OpenAIEmbedder
    store: QdrantStore

    def retrieve(
        self,
        query: Optional[str] = None,
        qvector: Optional[List[float]] = None,
        *,
        top_k: int = 5,
        needle_item_id: Optional[str] = None,
        source_file: Optional[str] = None,
        q_id: str = None,
        # score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        

        if needle_item_id is None:
            needle_item_id = q_id.split('_')[0]
        if (query is None) == (qvector is None):
            raise ValueError("Provide exactly one of: query OR qvector")
        
        
        qvec = qvector if qvector is not None else self.embedder.embed_texts([query])[0]
        hits = self.store.search(
            qvec,
            top_k=top_k,
            source_file=source_file,
            with_payload=True,
            needle_item_id = needle_item_id,
            # score_threshold=score_threshold,

        )

        # flatten the response into “RAG friendly” results
        out = []
        for h in hits:
            p = h.get("payload") or {}
            out.append({
                "score": h["score"],
                "run_id": p.get("run_id"),
                "source_file": p.get("source_file"),
                "evidence": p.get("evidence"),
                "chunk_type": p.get("chunk_type"),
                "needle_item_id": p.get("needle_item_id"),
                "text": p.get("text"),
            })
        # if ranking:
        #     if q_id:
        #         path = f"NoLiMa_based_RAG/results/NeedleRanking/{source_file}_{q_id}.json"
        #         write_json(path, out)
        
        return out

    def retrieve_all_with_scores(
        self,
        query: Optional[str] = None,
        qvector: Optional[List[float]] = None,
        *,
        needle_item_id: Optional[str] = None,
        source_file: Optional[str] = None,
        q_id: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve ALL chunks for a source_file using scroll (no top_k limit),
        compute cosine similarity locally, and return sorted results.
        """
        if needle_item_id is None and q_id is not None:
            needle_item_id = q_id.split('_')[0]
        if (query is None) == (qvector is None):
            raise ValueError("Provide exactly one of: query OR qvector")

        qvec = qvector if qvector is not None else self.embedder.embed_texts([query])[0]

        # Use scroll to get ALL points (no Qdrant search limit)
        points = self.store.scroll_all(
            source_file=source_file,
            needle_item_id=needle_item_id,
            with_payload=True,
            with_vectors=True,
            q_id=q_id,
            limit=256,
        )
        print(f"Scrolled {len(points)} points for source_file={source_file}")

        # Compute cosine similarity locally for each point
        out = []
        for pt in points:
            p = pt.get("payload") or {}
            v = pt.get("vector")
            if v is None:
                continue

            score = _cosine_similarity(qvec, v)
            out.append({
                "score": float(score),
                "run_id": p.get("run_id"),
                "source_file": p.get("source_file"),
                "evidence": p.get("evidence"),
                "chunk_type": p.get("chunk_type"),
                "needle_item_id": p.get("needle_item_id"),
                "text": p.get("text"),
            })

        # Sort by score descending (highest similarity first)
        out.sort(key=lambda x: x["score"], reverse=True)

        return out
