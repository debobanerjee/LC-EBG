# ragwithtopk/vectorstore/qdrant_store.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import uuid
from typing import Any, Dict, List, Optional
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    IsNullCondition,
)
from qdrant_client.http.models import SearchParams

from qdrant_client.http.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
)
import json

@dataclass
class QdrantStore:
    url: str
    api_key: Optional[str]
    collection: str

    def __post_init__(self) -> None:
        self.client = QdrantClient(url=self.url, api_key=self.api_key)

    def ensure_collection(self, vector_size: int, distance: Distance = Distance.COSINE) -> None:
        existing = self.client.get_collections().collections
        if any(c.name == self.collection for c in existing):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )

    def upsert(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        assert len(vectors) == len(payloads)
        points: List[PointStruct] = []
        batch_size = 200
        # print("num_vectors:", len(vectors))
        # payload_bytes = sum(len(json.dumps(p, ensure_ascii=False).encode("utf-8")) for p in payloads)
        # print("total_payload_bytes:", payload_bytes)

        # # Worst 5 payloads
        # sizes = [(len(json.dumps(p, ensure_ascii=False).encode("utf-8")), p.get("chunk_id"), p.get("source_file")) for p in payloads]
        # sizes.sort(reverse=True)
        # print("top5_payload_sizes:", sizes[:5])
        for i in range(0, len(vectors), batch_size):
            points: List[PointStruct] = []
            for v, p in zip(vectors[i:i+batch_size], payloads[i:i+batch_size]):
                # Stop using evidence in the ID. It’s wasteful + unstable.
                key = f"{p['run_id']}|{p['source_file']}|{p['evidence']}"
                pid = uuid.uuid5(uuid.NAMESPACE_URL, key)

                points.append(PointStruct(id=str(pid), vector=v, payload=p))

            self.client.upsert(collection_name=self.collection, points=points)
        # for v, p in zip(vectors, payloads):
        #     # Unique point id, stable for reruns if run_id+chunk_id stable
            # key = f"{p['run_id']}|{p['source_file']}|{p['evidence']}"
        #     pid = uuid.uuid5(uuid.NAMESPACE_URL, key)
        #     points.append(PointStruct(id=str(pid), vector=v, payload=p))
        # self.client.upsert(collection_name=self.collection, points=points)






    def search(
    self,
    query_vector: List[float],
    *,
    top_k: int = 10,
    needle_item_id: Optional[str] = None,
    source_file: Optional[str] = None,
    with_payload: bool = True,
    search_params: Optional[Dict[str, Any]] = None,  # accept but ignore
) -> List[Dict[str, Any]]:
            

        must_conditions = []
        if source_file:
            must_conditions.append(
                FieldCondition(key="source_file", match=MatchValue(value=str(source_file)))
            )

        # qfilter = None
        # qfilter = Filter(must=must_conditions)
        should_conditions = []
        if needle_item_id is not None:
            # Equivalent to: needle_item_id is None OR needle_item_id == <id>
            # We can't match null with MatchValue(None), so use chunk_type to include documents.
            should_conditions.append(
                FieldCondition(key="chunk_type", match=MatchValue(value="document"))
            )
            should_conditions.append(
                FieldCondition(key="needle_item_id", match=MatchValue(value=str(needle_item_id)))
            )
        qfilter = None
        # if should_conditions:
        #     qfilter = Filter(
        #         should=should_conditions or None,
        #     )
        if must_conditions or should_conditions:
            qfilter = Filter(
                must=must_conditions or None,
                should=should_conditions or None,
                # minimum_should_match=1 if should_conditions else None,
            )
        # _ = search_params
        res = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=qfilter,
            with_payload=with_payload,
            score_threshold = -1.0,
            # search_params={"hnsw_ef": max(4611 * 2, 512)},
        #     search_params=SearchParams(
        # hnsw_ef=max(20000, top_k * 3),  # big because you want thousands back
        # exact=False,
    # ),
        )

        hits = res.points
        return [{"id": str(h.id), "score": float(h.score), "payload": h.payload} for h in hits]
    
    def count_by_filter(self, needle_item_id: Optional[str] = None, source_file: Optional[str] = None) -> int:
        must_conditions = []
        if source_file:
            must_conditions.append(
                FieldCondition(key="source_file", match=MatchValue(value=str(source_file)))
            )

        should_conditions = []
        if needle_item_id is not None:
            should_conditions.append(
                FieldCondition(key="chunk_type", match=MatchValue(value="document"))
            )
            should_conditions.append(
                FieldCondition(key="needle_item_id", match=MatchValue(value=str(needle_item_id)))
            )
        qfilter = None
        if must_conditions or should_conditions:
            qfilter = Filter(
                must=must_conditions or None,
                should=should_conditions or None,
            )

        count_res = self.client.count(
            collection_name=self.collection,
            count_filter=qfilter,
        )
        return count_res.count

    def scroll_all(
        self,
        *,
        source_file: Optional[str] = None,
        needle_item_id: Optional[str] = None,
        with_payload: bool = True,
        with_vectors: bool = True,
        limit: int = 256,
        q_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scroll all points matching a filter (no similarity ranking).
        Returns payload + vectors so caller can score locally.
        """
        must_conditions = []
        if source_file:
            must_conditions.append(
                FieldCondition(key="source_file", match=MatchValue(value=str(source_file)))
            )

        should_conditions = []
        if q_id.split('_')[1] == "twohop" and needle_item_id is not None:
            should_conditions.append(
                FieldCondition(key="chunk_type", match=MatchValue(value="document"))
            )
            should_conditions.append(
                FieldCondition(key="needle_item_id", match=MatchValue(value=str(needle_item_id)))
            )
            should_conditions.append(
                FieldCondition(key="needle_item_id", match=MatchValue(value=f"{needle_item_id}_T"))
            )
        else:
            should_conditions.append(
                FieldCondition(key="chunk_type", match=MatchValue(value="document"))
            )
            should_conditions.append(
                FieldCondition(key="needle_item_id", match=MatchValue(value=str(needle_item_id)))
            )

        qfilter = None
        if must_conditions or should_conditions:
            qfilter = Filter(
                must=must_conditions or None,
                should=should_conditions or None,
            )

        all_points: List[Dict[str, Any]] = []
        next_offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=qfilter,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors,
                offset=next_offset,
            )
            for p in points:
                all_points.append({
                    "id": str(p.id),
                    "payload": p.payload,
                    "vector": p.vector,
                })
            if next_offset is None:
                break

        return all_points