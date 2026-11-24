"""
api/upload.py — Document Management Endpoints

Handles PDF upload, listing, and deletion.
After upload/delete, the indexer rebuilds FAISS + BM25 and the retriever is hot-reloaded.
"""
import os
import pickle
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException

from rag.indexer import add_document_to_index, remove_document_from_index
from rag.retriever import retriever
from config import DOCUMENTS_FILE, UPLOAD_DIR

router = APIRouter()
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a new medical PDF."""
    filename = os.path.basename((file.filename or "").replace("\\", "/"))
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = os.path.join(UPLOAD_DIR, filename)

    if os.path.exists(file_path):
        return {
            "message":  "Document already exists in the library.",
            "filename": filename,
        }

    # Save PDF
    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # Index into FAISS + BM25
    try:
        n_chunks = add_document_to_index(file_path)
        retriever.reload()   # hot-reload the retriever
    except Exception as e:
        # Clean up on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index document: {str(e)}"
        )

    return {
        "message":  f"Document uploaded and indexed successfully ({n_chunks} chunks).",
        "filename": file.filename,
        "chunks":   n_chunks,
    }


@router.get("/documents")
def list_documents():
    """Return list of all indexed documents with metadata."""
    if not os.path.exists(DOCUMENTS_FILE):
        return {"files": []}

    try:
        with open(DOCUMENTS_FILE, "rb") as f:
            documents = pickle.load(f)

        unique_docs: dict[str, dict] = {}
        for doc in documents:
            source   = doc.get("source",   "Unknown")
            category = doc.get("category", "General")
            if source not in unique_docs:
                unique_docs[source] = {
                    "name":     source,
                    "category": category,
                    "chunks":   0,
                }
            unique_docs[source]["chunks"] += 1

        return {"files": list(unique_docs.values())}

    except Exception as e:
        print(f"[Upload API] Error listing documents: {e}")
        return {"files": []}


@router.delete("/documents/{filename:path}")
def delete_document(filename: str):
    """Remove a document from the library and rebuild indexes."""
    filename = os.path.basename(filename.replace("\\", "/"))

    # Remove physical file (uploads dir)
    upload_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(upload_path):
        os.remove(upload_path)

    # Remove from indexes
    removed = remove_document_from_index(filename)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found in the search index."
        )

    retriever.reload()   # hot-reload the retriever

    return {"message": f"Document '{filename}' deleted successfully."}
