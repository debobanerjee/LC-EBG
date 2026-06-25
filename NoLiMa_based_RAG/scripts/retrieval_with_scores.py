import argparse
import os
import yaml
import json

from ragwithtopk.embeddings.openai_embedder import OpenAIEmbedder
from ragwithtopk.vectorstore.qdrant_store import QdrantStore
from ragwithtopk.rag.retriever import Retriever
from ragwithtopk.rag.formatters import RetrievedContextFormatter

def load_qvector(questions_path: str, q_id: str):
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # questions is a list of dicts
    for q in questions:
        if q.get("q_id", -1) == q_id:
            if "embedding" not in q:
                raise ValueError(f"Found q_id={q_id} but it has no embedding field.")
            return q["embedding"]

    raise KeyError(f"q_id={q_id} not found in {questions_path}")

def main():
    formatter = RetrievedContextFormatter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    ap.add_argument("--questions_json", default=None, help="Path to questions.json with embeddings")
    ap.add_argument("--q_id", type=str, default=None, help="string q_id to load embedding from questions_json")

    ap.add_argument("--query", default=None, help="Raw text query (used if q_id not provided)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--source_file", default=None)
    ap.add_argument("--needle_item_id", default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    api_key = os.environ["OPENAI_API_KEY"]

    embedder = OpenAIEmbedder(
        api_key=api_key,
        model=d.get("embedding_model", "text-embedding-3-small"),
        batch_size=d.get("embedding_batch_size", 16),
    )
    store = QdrantStore(
        url=d["qdrant_url"],
        api_key=d.get("qdrant_api_key"),
        collection=d["qdrant_collection"],
    )
    retriever = Retriever(embedder=embedder, store=store)

    # Decide vector source
    qvector = None
    query = None

    if args.q_id is not None:
        if not args.questions_json:
            raise ValueError("If you pass --q_id, you must also pass --questions_json")
        qvector = load_qvector(args.questions_json, args.q_id)
    else:
        if not args.query:
            raise ValueError("Provide either --query OR (--questions_json and --q_id)")
        query = args.query

    # hits = retriever.retrieve(
    #     query=query,
    #     qvector=qvector,
    #     top_k=args.topk,
    #     source_file=args.source_file,
    #     needle_item_id=args.needle_item_id,
    #     ranking=True,
    #     q_id = args.q_id  # optional
    # )
    # texts = formatter.format_with_line_numbers(hits)
    # print(texts)
    hits = retriever.retrieve_all_with_scores(
        query=query,
        qvector=qvector,
        needle_item_id=args.needle_item_id,
        source_file=args.source_file,
        q_id=args.q_id
        )
    print(hits)
    # for i, h in enumerate(hits, 1):
    #     print(f"\n#{i} score={h['score']:.4f} evidence={h['evidence']} type={h['chunk_type']} needle={h['needle_item_id']}")
    #     print(h["text"])

if __name__ == "__main__":
    main()
