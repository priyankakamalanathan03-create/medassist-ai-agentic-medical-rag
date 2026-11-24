"""
rag/indexer.py -- Builds and saves FAISS + BM25 indexes from PDF documents.

Run this script once to index all raw PDFs:
    cd backend
    python -m rag.indexer
"""
import os
import pickle
import sys
import re

# Force UTF-8 output so Unicode in print() does not crash on Windows cp1252
if (sys.stdout.encoding or "").lower().replace("-", "") != "utf8" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# Use cached models -- no network access needed
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    RAW_DIR, DATA_DIR, FAISS_FILE, DOCUMENTS_FILE,
    BM25_FILE, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
)

CATEGORY_MAP = {
    "Asthma.pdf":                            "Respiratory",
    "Cancer.pdf":                            "Oncology",
    "Cardiovascular diseases (CVDs).pdf":    "Cardiovascular",
    "Diabetes.pdf":                          "Endocrine",
    "What Is Diabetes_ - NIDDK.pdf":         "Endocrine",
    "main.pdf":                              "Medical Research",
    "clinmed-20-1-43.pdf":                   "Clinical Medicine",
    "nihms286538.pdf":                       "Medical Research",
    "41401_2018_Article_173.pdf":            "Medical Research",
    "foods-09-01214.pdf":                    "Nutrition",
}


def extract_pages(pdf_path: str) -> list:
    reader = PdfReader(pdf_path)
    pages = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": idx + 1, "text": text})
    return pages


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def build_index(pdf_folder: str = RAW_DIR) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    documents = []

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print("[Indexer] No PDFs found in", pdf_folder)
        return

    print("[Indexer] Processing", len(pdf_files), "PDF(s)...")

    for file_name in pdf_files:
        pdf_path = os.path.join(pdf_folder, file_name)
        category = CATEGORY_MAP.get(file_name, "General Medicine")
        print("  ->", file_name, "[" + category + "]")

        pages = extract_pages(pdf_path)
        for page_data in pages:
            cleaned = clean_text(page_data["text"])
            chunks  = chunk_text(cleaned)
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text":     chunk,
                    "source":   file_name,
                    "category": category,
                    "page":     page_data["page"],
                    "chunk_id": f"{file_name}_{page_data['page']}_{i}",
                })

    print("[Indexer] Total chunks:", len(documents))

    # Save documents
    with open(DOCUMENTS_FILE, "wb") as f:
        pickle.dump(documents, f)
    print("[Indexer] Documents saved ->", DOCUMENTS_FILE)

    # Generate embeddings
    print("[Indexer] Generating embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [d["text"] for d in documents]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Build FAISS index
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, FAISS_FILE)
    print("[Indexer] FAISS index saved:", index.ntotal, "vectors")

    # Build BM25 index
    tokenized = [t.lower().split() for t in texts]
    bm25      = BM25Okapi(tokenized)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    print("[Indexer] BM25 index saved ->", BM25_FILE)
    print("[Indexer] Indexing complete!")


def add_document_to_index(pdf_path: str) -> int:
    file_name = os.path.basename(pdf_path)
    category  = CATEGORY_MAP.get(file_name, "Uploaded Document")

    pages = extract_pages(pdf_path)
    new_documents = []
    for page_data in pages:
        cleaned = clean_text(page_data["text"])
        chunks  = chunk_text(cleaned)
        for i, chunk in enumerate(chunks):
            new_documents.append({
                "text":     chunk,
                "source":   file_name,
                "category": category,
                "page":     page_data["page"],
                "chunk_id": f"{file_name}_{page_data['page']}_{i}",
            })

    if not new_documents:
        return 0

    documents = []
    if os.path.exists(DOCUMENTS_FILE):
        with open(DOCUMENTS_FILE, "rb") as f:
            documents = pickle.load(f)

    documents.extend(new_documents)

    with open(DOCUMENTS_FILE, "wb") as f:
        pickle.dump(documents, f)

    model      = SentenceTransformer(EMBEDDING_MODEL)
    texts      = [d["text"] for d in documents]
    embeddings = model.encode(texts, normalize_embeddings=True)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, FAISS_FILE)

    tokenized = [t.lower().split() for t in texts]
    bm25      = BM25Okapi(tokenized)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)

    print("[Indexer] Added", len(new_documents), "chunks from", file_name)
    return len(new_documents)


def remove_document_from_index(filename: str) -> bool:
    if not os.path.exists(DOCUMENTS_FILE):
        return False

    with open(DOCUMENTS_FILE, "rb") as f:
        documents = pickle.load(f)

    original_count = len(documents)
    documents = [d for d in documents if d.get("source") != filename]

    if len(documents) == original_count:
        return False

    with open(DOCUMENTS_FILE, "wb") as f:
        pickle.dump(documents, f)

    if documents:
        model      = SentenceTransformer(EMBEDDING_MODEL)
        texts      = [d["text"] for d in documents]
        embeddings = model.encode(texts, normalize_embeddings=True)

        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype("float32"))
        faiss.write_index(index, FAISS_FILE)

        tokenized = [t.lower().split() for t in texts]
        bm25      = BM25Okapi(tokenized)
        with open(BM25_FILE, "wb") as f:
            pickle.dump(bm25, f)
    else:
        for path in [FAISS_FILE, BM25_FILE]:
            if os.path.exists(path):
                os.remove(path)

    print("[Indexer] Removed", filename, "- Remaining chunks:", len(documents))
    return True


if __name__ == "__main__":
    build_index()
