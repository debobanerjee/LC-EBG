# ragwithtopk/embeddings/openai_embedder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from openai import OpenAI

@dataclass
class OpenAIEmbedder:
    api_key: str
    model: str = "text-embedding-3-small"
    batch_size: int = 128

    def __post_init__(self) -> None:
        self.client = OpenAI(api_key=self.api_key)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ") for t in texts[i:i+self.batch_size]]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend([d.embedding for d in resp.data])
        return vectors
