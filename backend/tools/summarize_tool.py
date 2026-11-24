"""
tools/summarize_tool.py — Document summarization tool.

The agent calls this to get a concise summary of all chunks
from a specific source document.
"""
from tools.tool_registry import ToolRegistry, Tool


def _summarize_document(document_name: str) -> dict:
    """
    Summarize all content from a specific indexed document.

    Args:
        document_name: The PDF filename (e.g. 'Diabetes.pdf')

    Returns:
        {
            "document": str,
            "summary_chunks": [str, ...],   # representative passages with page citations
            "total_chunks": int,
            "category": str
        }
    """
    from rag.retriever import retriever

    # Filter all chunks belonging to this document
    doc_chunks = [
        d for d in retriever.documents
        if d.get("source", "").lower() == document_name.lower()
    ]

    if not doc_chunks:
        # Try partial match
        doc_chunks = [
            d for d in retriever.documents
            if document_name.lower() in d.get("source", "").lower()
        ]

    if not doc_chunks:
        return {
            "error":    f"Document '{document_name}' not found in the knowledge base.",
            "document": document_name,
        }

    # Sort by page then chunk position
    doc_chunks.sort(key=lambda x: (x.get("page", 0), x.get("chunk_id", "")))

    # Return up to 6 evenly-spaced representative chunks
    total   = len(doc_chunks)
    step    = max(1, total // 6)
    sampled = doc_chunks[::step][:6]

    category = doc_chunks[0].get("category", "General")

    return {
        "document":      document_name,
        "category":      category,
        "total_chunks":  total,
        "summary_chunks": [
            f"[{chunk['source']}, p. {chunk.get('page', 1)}] {chunk['text']}"
            for chunk in sampled
        ],
    }


def register_summarize_tool(registry: ToolRegistry) -> None:
    registry.register(Tool(
        name="summarize_document",
        description=(
            "Retrieve a structured summary of all content from a specific indexed "
            "medical document by its filename. Use when the user asks to 'summarize' "
            "a document or needs an overview of a specific file."
        ),
        func=_summarize_document,
        parameters={
            "document_name": "The exact or partial PDF filename to summarize (e.g. 'Diabetes.pdf')",
        },
    ))
