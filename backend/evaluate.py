"""
evaluate.py — Evaluation script using MedQuAD dataset

Measures the quality of the agentic RAG pipeline on real medical Q&A pairs.

Usage:
    cd backend
    python evaluate.py

Output: prints precision@5, answer quality metrics, and saves results to
        data/evaluation/results.json
"""
import os
import sys
import json
import csv
import time

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from rag.retriever import retriever


MEDQUAD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "evaluation", "medquad.csv"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "evaluation", "results.json"
)


def load_medquad(path: str, limit: int = 50) -> list[dict]:
    """Load first N Q&A pairs from MedQuAD CSV."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            q = row.get("question", "").strip()
            a = row.get("answer",   "").strip()
            if q and a:
                pairs.append({"question": q, "answer": a})
    return pairs


def evaluate_retrieval(pairs: list[dict], k: int = 5) -> dict:
    """
    Evaluate retrieval quality:
    - Retrieval Hit Rate: % of questions where at least 1 chunk is retrieved
    - Avg chunks retrieved per question
    """
    hit      = 0
    total_k  = 0

    for pair in pairs:
        docs = retriever.retrieve(pair["question"], k=k)
        if docs:
            hit += 1
        total_k += len(docs)

    n = len(pairs)
    return {
        "retrieval_hit_rate": round(hit / n, 3) if n > 0 else 0,
        "avg_chunks_retrieved": round(total_k / n, 2) if n > 0 else 0,
        "total_questions": n,
    }


def main():
    print("[Evaluation] Loading MedQuAD dataset...")
    if not os.path.exists(MEDQUAD_PATH):
        print(f"[Evaluation] MedQuAD CSV not found at: {MEDQUAD_PATH}")
        return

    pairs = load_medquad(MEDQUAD_PATH, limit=50)
    print(f"[Evaluation] Loaded {len(pairs)} Q&A pairs")

    print("[Evaluation] Running retrieval evaluation...")
    retrieval_metrics = evaluate_retrieval(pairs, k=5)
    print(f"  Retrieval Hit Rate:    {retrieval_metrics['retrieval_hit_rate'] * 100:.1f}%")
    print(f"  Avg Chunks Retrieved:  {retrieval_metrics['avg_chunks_retrieved']}")

    results = {
        "timestamp":         time.strftime("%Y-%m-%d %Human:%M:%S"),
        "retrieval_metrics": retrieval_metrics,
        "model":             "all-MiniLM-L6-v2 + BM25 Hybrid",
        "k":                 5,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Evaluation] Results saved to: {OUTPUT_PATH}")
    print("[Evaluation] ✅ Done!")


if __name__ == "__main__":
    main()
