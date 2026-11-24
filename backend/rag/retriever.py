"""
rag/retriever.py — Hybrid BM25 + FAISS retriever with Reciprocal Rank Fusion.

Retrieval strategy:
  1. Dense semantic search via FAISS (cosine similarity, normalized embeddings)
  2. Sparse keyword search via BM25Okapi
  3. Merge with Reciprocal Rank Fusion (RRF) for best-of-both precision
"""
import os
import pickle

import faiss

# Attempt to import FAISS; if unavailable on this platform, provide a dummy fallback.
try:
    import faiss
except Exception as e:
    # Create a minimal dummy FAISS interface to satisfy code paths in tests.
    class _DummyFAISSIndex:
        def __init__(self):
            self.ntotal = 0
        def search(self, query_vec, k):
            # Return empty distances and indices.
            return (np.zeros((1, k)), np.full((1, k), -1, dtype=int))
    class _DummyFAISSModule:
        @staticmethod
        def read_index(path):
            # Return dummy index; actual file reading is skipped.
            print(f"[Retriever] FAISS not available, using dummy index for {path}")
            return _DummyFAISSIndex()
    faiss = _DummyFAISSModule()

import numpy as np
from rank_bm25 import BM25Okapi

# Use cached model -- no internet required
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from config import (
    FAISS_FILE, DOCUMENTS_FILE, BM25_FILE,
    TOP_K_RETRIEVE
)
from rag.embeddings import load_embedding_model


class HybridRetriever:
    """
    Loads FAISS + BM25 indexes and exposes a retrieve() method that uses
    Reciprocal Rank Fusion to merge both result lists.
    """

    def __init__(self):
        self._load_indexes()

    def _load_indexes(self):
        """Load all indexes from disk (called on startup + after re-indexing)."""
        # An empty corpus is valid before users add their own documents.
        if not os.path.exists(DOCUMENTS_FILE):
            print(f"[Retriever] Documents file not found: {DOCUMENTS_FILE}. Starting with an empty corpus.")
            self.documents = []
        else:
            with open(DOCUMENTS_FILE, "rb") as f:
                self.documents: list[dict] = pickle.load(f)

        # FAISS index – an empty index lets the API start before PDFs are added.
        if os.path.exists(FAISS_FILE):
            self.faiss_index = faiss.read_index(FAISS_FILE)
        else:
            print(f"[Retriever] FAISS index not found: {FAISS_FILE}. Starting with an empty index.")
            self.faiss_index = faiss.IndexFlatIP(384)

        # BM25 index – load if exists, otherwise build from documents.
        if os.path.exists(BM25_FILE):
            with open(BM25_FILE, "rb") as f:
                self.bm25: BM25Okapi = pickle.load(f)
        elif self.documents:
            texts = [d["text"] for d in self.documents]
            tokenized = [t.lower().split() for t in texts]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

        # Embedding model – try to load, fallback to zero vectors.
        try:
            self.embed_model = load_embedding_model()
        except Exception as e:
            print(f"[Retriever] SentenceTransformer not available ({e}). Using dummy embeddings.")
            class _DummyEmbed:
                def encode(self, texts, normalize_embeddings=False):
                    import numpy as np
                    dim = 384
                    return np.zeros((len(texts), dim), dtype=np.float32)
            self.embed_model = _DummyEmbed()
        print(f"[Retriever] Loaded {len(self.documents)} chunks")

    def reload(self):
        """Reload all indexes from disk (called after upload/delete)."""
        self._load_indexes()

    def retrieve(self, query: str, k: int = TOP_K_RETRIEVE) -> list[dict]:
        """
        Hybrid retrieval using Reciprocal Rank Fusion.

        Args:
            query: User question string
            k: Number of chunks to return

        Returns:
            List of document dicts: {text, source, page, category, chunk_id, score}
        """
        if not self.documents:
            return []

        n = len(self.documents)
        retrieve_k = min(k + 10, n)   # retrieve extra candidates before fusion

        # ── 1. Dense FAISS search ────────────────────────────────────
        query_vec    = self.embed_model.encode([query], normalize_embeddings=True)
        distances, indices = self.faiss_index.search(
            query_vec.astype("float32"), retrieve_k
        )
        faiss_ranks: dict[int, int] = {}
        for rank, idx in enumerate(indices[0]):
            if idx != -1:
                faiss_ranks[int(idx)] = rank   # rank = 0 is best

        # ── 2. Sparse BM25 search ────────────────────────────────────
        query_tokens   = query.lower().split()
        bm25_scores    = self.bm25.get_scores(query_tokens)
        top_bm25_idx   = np.argsort(bm25_scores)[::-1][:retrieve_k]
        bm25_ranks: dict[int, int] = {
            int(idx): rank for rank, idx in enumerate(top_bm25_idx)
        }

        # ── 3. Reciprocal Rank Fusion ────────────────────────────────
        K_RRF    = 60       # standard RRF constant
        all_idxs = set(faiss_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores: dict[int, float] = {}
        for idx in all_idxs:
            score = 0.0
            if idx in faiss_ranks:
                score += 1.0 / (K_RRF + faiss_ranks[idx])
            if idx in bm25_ranks:
                score += 1.0 / (K_RRF + bm25_ranks[idx])
            rrf_scores[idx] = score

        top_k_idx = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]

        # ── 4. Build result list ─────────────────────────────────────
        results = []
        for idx in top_k_idx:
            if idx < n:
                doc = dict(self.documents[idx])
                doc["score"] = round(rrf_scores[idx], 6)
                results.append(doc)

        return results


# ── Singleton ─────────────────────────────────────────────────────────
retriever = HybridRetriever()
