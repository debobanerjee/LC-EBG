import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from qdrant_client.http.exceptions import UnexpectedResponse

import yaml

from ragwithtopk.embeddings.openai_embedder import OpenAIEmbedder
from ragwithtopk.rag.retriever import Retriever
from ragwithtopk.utils.io import load_json, write_json
from ragwithtopk.vectorstore.qdrant_store import QdrantStore


def needle_rank(hits: List[Dict[str, Any]], needle_item_id: str) -> Optional[int]:
    for i, h in enumerate(hits, start=1):
        if str(h.get("needle_item_id")) == str(needle_item_id) and h.get("chunk_type") == "needle":
            return i
    return None


def derive_source_file(text_path: str) -> str:
    """
    datasets/NoLiMa/.../rand_shuffle_10000/rand_book_1.txt -> 10000_rand_book_1
    """
    p = Path(text_path)
    parent = p.parent.name
    if not parent.startswith("rand_shuffle_"):
        raise ValueError(f"Expected rand_shuffle_<N> folder, got: {parent}")
    char_len = parent.replace("rand_shuffle_", "", 1)
    return f"{char_len}_{p.stem}"


def build_qid_index(questions_path: str) -> Dict[str, Tuple[List[float], Dict[str, Any]]]:
    questions = load_json(questions_path)
    idx: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}
    for q in questions:
        qid = str(q.get("q_id"))
        if not qid or qid == "None":
            continue
        if "embedding" not in q:
            raise ValueError(f"Found q_id={qid} but it has no embedding field.")
        idx[qid] = (q["embedding"], q)
    return idx


def safe_name(s: str) -> str:
    return s.replace("/", "_")


def load_existing_summary(summary_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load existing summary and return as dict keyed by (source_file, q_id).
    """
    if not summary_path.exists():
        return {}
    existing = load_json(str(summary_path))
    # Build index by (source_file, q_id) for easy lookup/update
    return {(item["source_file"], item["q_id"]): item for item in existing}


def merge_summary(existing: Dict[str, Dict[str, Any]], new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge new_items into existing. If (source_file, q_id) already exists, update it.
    Returns a sorted list.
    """
    merged = existing.copy()
    for item in new_items:
        key = (item["source_file"], item["q_id"])
        merged[key] = item  # overwrite if exists, add if new
    
    # Sort by source_file (numeric prefix) then q_id
    def sort_key(item: Dict[str, Any]) -> tuple:
        sf = item["source_file"]
        # Extract numeric prefix for proper numeric sorting
        try:
            num = int(sf.split("_")[0])
        except ValueError:
            num = 0
        return (num, item["q_id"])
    
    return sorted(merged.values(), key=sort_key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--questions_json", required=True)

    # Optional: if not provided, derive from cfg['text_files']
    ap.add_argument("--source_files", nargs="+", default=None)

    ap.add_argument("--out_dir", default="NoLiMa_based_RAG/results/NeedleRanking")
    ap.add_argument("--limit_qids", type=int, default=None)
    ap.add_argument("--save_hits", action="store_true", help="Write per-(source_file,q_id) scored hits JSON files")
    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument("--extra_needles", type=bool, default=False, help="Whether to consider extra needles beyond the main one (e.g. for multi-needle tasks)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Resolve source_files
    if args.source_files is not None:
        source_files = args.source_files
    else:
        text_files = cfg.get("text_files") or []
        if not text_files:
            raise ValueError("No --source_files provided and config has no text_files.")
        source_files = [derive_source_file(tp) for tp in text_files]

    api_key = os.environ["OPENAI_API_KEY"]
    embedder = OpenAIEmbedder(
        api_key=api_key,
        model=cfg.get("embedding_model", "text-embedding-3-small"),
        batch_size=cfg.get("embedding_batch_size", 16),
    )
    store = QdrantStore(
        url=cfg["qdrant_url"],
        api_key=cfg.get("qdrant_api_key"),
        collection=cfg["qdrant_collection"],
    )
    retriever = Retriever(embedder=embedder, store=store)

    qid_index = build_qid_index(args.questions_json)
    qids = list(qid_index.keys())
    if args.limit_qids is not None:
        qids = qids[: args.limit_qids]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "needle_rank_summary.json"

    # Load existing summary to merge with
    existing_summary = load_existing_summary(summary_path)
    print(f"Loaded {len(existing_summary)} existing entries from {summary_path}")

    new_summary: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    total = len(source_files) * len(qids) if qids else 0
    n = 0

    for source_file in source_files:
        for q_id in qids:
            n += 1
            try:
                qvec, qmeta = qid_index[q_id]
                needle_item_id = str(q_id).split("_")[0]
                hits = retriever.retrieve_all_with_scores(
                    qvector=qvec,
                    source_file=source_file,
                    q_id=q_id,
                    needle_item_id=needle_item_id,
                )

                if args.save_hits:
                    # Group outputs by the character-length prefix (e.g., 10000/10000_rand_book_1_0405_onehop.json)
                    group_dir = out_dir / source_file.split("_")[0]
                    group_dir.mkdir(parents=True, exist_ok=True)
                    out_path = group_dir / f"{safe_name(source_file)}_{safe_name(q_id)}.json"
                    write_json(str(out_path), hits)

                rank = needle_rank(hits, needle_item_id)

                new_summary.append(
                    {
                        "source_file": source_file,
                        "q_id": q_id,
                        "task_id": qmeta.get("task_id"),
                        "question_key": qmeta.get("question_key"),
                        "test_key": qmeta.get("test_key"),
                        "needle_item_id": needle_item_id,
                        "needle_rank_1based": rank,
                        "num_scored": len(hits),
                    }
                )

                if n % 50 == 0 or n == total:
                    print(f"[{n}/{total}] done")

            except Exception as e:
                err: Dict[str, Any] = {"source_file": source_file, "q_id": q_id}
                if isinstance(e, UnexpectedResponse):
                    err["error_type"] = "UnexpectedResponse"
                    err["status_code"] = getattr(e, "status_code", None)
                    content = getattr(e, "content", None)
                    err["content"] = (
                        content.decode("utf-8", errors="replace")
                        if isinstance(content, (bytes, bytearray))
                        else str(content) if content is not None else None
                    )
                    err["raw"] = getattr(e, "raw", None)
                else:
                    err["error"] = repr(e)

                errors.append(err)
                if not args.continue_on_error:
                    raise

    # Merge new results with existing and write
    merged_summary = merge_summary(existing_summary, new_summary)
    write_json(str(summary_path), merged_summary)

    if errors:
        write_json(str(out_dir / "errors.json"), errors)

    print(f"Wrote merged summary ({len(merged_summary)} total entries): {summary_path}")
    if errors:
        print(f"Wrote errors: {out_dir / 'errors.json'} ({len(errors)} failures)")


if __name__ == "__main__":
    main()