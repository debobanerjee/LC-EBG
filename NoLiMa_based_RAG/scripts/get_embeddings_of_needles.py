import os
from typing import List, Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

from ragwithtopk.embeddings.openai_embedder import OpenAIEmbedder


def cosine_similarity_matrix(query_vec: np.ndarray, doc_mat: np.ndarray) -> np.ndarray:
    """
    Returns cosine similarity between 1 query vector and N doc vectors.
    """
    # Normalize
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    d = doc_mat / (np.linalg.norm(doc_mat, axis=1, keepdims=True) + 1e-12)
    return d @ q  # shape: (N,)

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def plot_similarities(question: str, needles: List[str], sims: np.ndarray, top_k: int = 30) -> None:
    """
    Bar plot of top_k most similar needles to the question.
    """
    order = np.argsort(-sims)  # descending
    order = order[: min(top_k, len(needles))]

    labels = [needles[i] for i in order]
    values = sims[order]

    plt.figure(figsize=(12, max(4, 0.35 * len(order))))
    plt.barh(range(len(order)), values)
    plt.gca().invert_yaxis()
    plt.yticks(range(len(order)), [f"{i}: {labels[idx][:80]}{'...' if len(labels[idx])>80 else ''}" for i, idx in enumerate(range(len(order)))])
    plt.xlabel("Cosine similarity")
    plt.title(f"Top-{len(order)} needle similarities for question:\n{question}")
    plt.tight_layout()
    plt.savefig("RAG_Topk/data/plots/<plot_name>.png", dpi=300, bbox_inches="tight")
    plt.close()


def _extract_name(sentence: str) -> str:
    """
    Extract name from sentences shaped like:
    'Then <Name> mentioned ...'
    """
    parts = sentence.split()
    if len(parts) > 1 and parts[0] == "Then":
        return parts[1]
    return sentence[:20]


def plot_rephrased_similarity_comparison(
    question: str,
    needles: List[str],
    new_needles: List[str],
    sims_original: np.ndarray,
    sims_rephrased: np.ndarray,
) -> None:
    """
    Compare question->needle cosine similarities for original vs rephrased pairs
    matched by index.
    """
    pair_count = min(len(needles), len(new_needles), len(sims_original), len(sims_rephrased))
    if pair_count == 0:
        return

    labels = [_extract_name(needles[i]) for i in range(pair_count)]
    orig_vals = np.array(sims_original[:pair_count], dtype=np.float32)
    reph_vals = np.array(sims_rephrased[:pair_count], dtype=np.float32)
    deltas = reph_vals - orig_vals

    x = np.arange(pair_count, dtype=np.float32)
    bar_w = 0.38

    plt.figure(figsize=(14, max(5, 0.5 * pair_count)))
    plt.bar(x - bar_w / 2, orig_vals, width=bar_w, label="Original needle")
    plt.bar(x + bar_w / 2, reph_vals, width=bar_w, label="Rephrased needle")

    for i in range(pair_count):
        top_v = max(orig_vals[i], reph_vals[i])
        plt.text(i, top_v + 0.002, f"Δ={deltas[i]:+.3f}", ha="center", va="bottom", fontsize=8)

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Cosine similarity to question")
    plt.xlabel("Needle pair (matched by index)")
    plt.title(f"Original vs Rephrased needle similarity for question:\n{question}")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig("RAG_Topk/data/plots/CharacterWOGender.png", dpi=600, bbox_inches="tight")
    plt.close()


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY env var")

    embedder = OpenAIEmbedder(api_key=api_key, model="text-embedding-3-small", batch_size=128)

    # ---- Your needles ----
    needles = [
    "Then Yuki mentioned that he has been vegan for years.",
    "Then Stuart mentioned that he has been vegan for years.",
    "Then Katie mentioned that he has been vegan for years.",
    "Then Veronica mentioned that he has been vegan for years.",
    "Then Gary mentioned that he has been vegan for years.",
    "Then Megan mentioned that he has been vegan for years.",
    "Then Calvin mentioned that he has been vegan for years.",
    "Then Mandy mentioned that he has been vegan for years.",
    "Then Diana mentioned that he has been vegan for years.",
    "Then Caxleb mentioned that he has been vegan for years."
  ]
    new_needles = [
    "Then Yuki mentioned being vegan for years.",
    "Then Stuart mentioned being vegan for years.",
    "Then Katie mentioned being vegan for years.",
    "Then Veronica mentioned being vegan for years.",
    "Then Gary mentioned being vegan for years.",
    "Then Megan mentioned being vegan for years.",
    "Then Calvin mentioned being vegan for years.",
    "Then Mandy mentioned being vegan for years.",
    "Then Diana mentioned being vegan for years.",
    "Then Caxleb mentioned being vegan for years."
]


    questions = [
        "Which character cannot eat Brandade?",
    ]

    # 1) Embed needles once
    needle_vecs = embedder.embed_texts(needles)
    needle_mat = np.array(needle_vecs, dtype=np.float32)  # shape (N, D)
    new_needle_vecs = embedder.embed_texts(new_needles)
    new_needle_mat = np.array(new_needle_vecs, dtype=np.float32)  # shape (N, D)

    # 2) For each question, embed and compute similarities against all needles
    for q in questions:
        q_vec = np.array(embedder.embed_texts([q])[0], dtype=np.float32)  # shape (D,)
        sims = cosine_similarity_matrix(q_vec, needle_mat)               # shape (N,)
        new_sims = cosine_similarity_matrix(q_vec, new_needle_mat)       # shape (N,)

        # Print all similarities
        print("\nQUESTION:", q)
        for i, (text, s) in enumerate(zip(needles, sims)):
            print(f"{i:02d}  sim={s:.4f}  needle={text}")

        for i, (text, s) in enumerate(zip(new_needles, new_sims)):
            print(f"{i:02d}  sim={s:.4f}  needle={text}")
        # 3) Plot top-k
        plot_similarities(q, needles, sims, top_k=len(needles))
        plot_rephrased_similarity_comparison(q, needles, new_needles, sims, new_sims)


if __name__ == "__main__":
    main()
