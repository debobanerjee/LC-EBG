# ragwithtopk/experiments/runner.py
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List
import time

from ragwithtopk.utils.io import write_json, write_jsonl, sha256_text
from ragwithtopk.utils.rng import seed_for_file
from ragwithtopk.nolima.needles import NeedleExtractor
from ragwithtopk.chunking.line_chunker import LineChunker
from ragwithtopk.injection.inserter import NeedleInserter
from ragwithtopk.embeddings.openai_embedder import OpenAIEmbedder
from ragwithtopk.vectorstore.qdrant_store import QdrantStore

class ExperimentRunner:
    def __init__(self, cfg):
        self.cfg = cfg

        self.needle_extractor = NeedleExtractor(cfg.needle_set_path)
        self.chunker = LineChunker(drop_empty=True)
        self.inserter = NeedleInserter(allow_before_first=True)

        self.embedder = OpenAIEmbedder(
            api_key=cfg.openai_api_key,
            model=cfg.embedding_model,
            batch_size=cfg.embedding_batch_size,
        )

        self.store = QdrantStore(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key,
            collection=cfg.qdrant_collection,
        )

    def run_all(self) -> None:
        needles = self.needle_extractor.extract(self.cfg.reasoning_type)

        run_id_base = self.cfg.run_name
        out_base = Path(self.cfg.output_dir)

        for text_path in self.cfg.text_files:
            p = Path(text_path)
            source_file = p.name

            per_file_seed = seed_for_file(self.cfg.base_seed, p)
            run_id = run_id_base

            base_chunks = self.chunker.chunk_file(text_path)
            records = self.inserter.inject(
                base_chunks=base_chunks,
                needles=needles,
                seed=per_file_seed,
                source_file=p,
                run_id=run_id,
            )

            # Write injected chunks for reproducibility/debug
            injected_out = out_base / "inserted" / f"{run_id}.jsonl"
            write_jsonl(injected_out, records)

            # Embed + upsert
            texts = [r["text"] for r in records]
            vectors = self.embedder.embed_texts(texts)
            self.store.ensure_collection(vector_size=len(vectors[0]))
            self.store.upsert(vectors=vectors, payloads=records)

            # Manifest (the “receipt” that makes it reproducible)
            manifest = {
                "run_id": run_id,
                "source_file": source_file,
                "text_path": str(p),
                "reasoning_type": self.cfg.reasoning_type,
                "base_seed": self.cfg.base_seed,
                "per_file_seed": per_file_seed,
                "needle_set_path": self.cfg.needle_set_path,
                "needles_count": len(needles),
                "doc_chunks_count": len(base_chunks),
                "final_chunks_count": len(records),
                "embedding_model": self.cfg.embedding_model,
                "qdrant_collection": self.cfg.qdrant_collection,
                "created_at_unix": int(time.time()),
                "config": asdict(self.cfg),
            }
            manifest_out = out_base / "manifests" / f"{run_id}.json"
            write_json(manifest_out, manifest)
