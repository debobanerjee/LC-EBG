import argparse
import os
from pathlib import Path

from ragwithtopk.nolima.questions import QuestionExtractor
from ragwithtopk.utils.io import load_json, write_json
from ragwithtopk.embeddings.openai_embedder import OpenAIEmbedder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--needle_set", required=True)
    ap.add_argument("--reasoning_type", default="commonsense_knowledge")
    ap.add_argument("--qtype", default="all")           # all|onehop|twohop
    ap.add_argument("--test_pick", default="first")     # first|sorted_first
    ap.add_argument("--out", required=True)             # e.g. NoLiMa_based_RAG/questions/questions.json
    ap.add_argument("--model", default="text-embedding-3-small")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--overwrite_embeddings", action="store_true")
    args = ap.parse_args()

    existing_by_qid = {}
    if Path(args.out).exists() and not args.overwrite_embeddings:
        for q in load_json(args.out):
            if q.get("q_id") and q.get("embedding"):
                existing_by_qid[q["q_id"]] = q["embedding"]

    # Step 1: extract questions from NoLiMa
    extractor = QuestionExtractor(args.needle_set)

    extractor.extract(
        reasoning_type=args.reasoning_type,
        qtype=args.qtype,
        test_pick=args.test_pick,
        out_path=args.out,
        include_q_id=True,
    )

    # Step 2: load that JSON and add embeddings
    questions = load_json(args.out)
    for q in questions:
        if q.get("q_id") in existing_by_qid and "embedding" not in q:
            q["embedding"] = existing_by_qid[q["q_id"]]

    to_embed_idx = []
    to_embed_texts = []
    for i, q in enumerate(questions):
        if args.overwrite_embeddings or ("embedding" not in q):
            to_embed_idx.append(i)
            to_embed_texts.append(q["question"])

    if to_embed_texts:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY env var or rerun with an output file that already has embeddings.")
        embedder = OpenAIEmbedder(api_key=api_key, model=args.model, batch_size=args.batch_size)
        vecs = embedder.embed_texts(to_embed_texts)
        for idx, vec in zip(to_embed_idx, vecs):
            # questions[idx]["embedding_model"] = args.model
            questions[idx]["embedding"] = vec

    write_json(args.out, questions)

    print(f"Done. Wrote questions + embeddings to: {args.out}")

if __name__ == "__main__":
    main()
