"""
tools/search_tool.py — Document search tool using the hybrid retriever.

The agent calls this to retrieve relevant medical document chunks for a query.
"""
from tools.tool_registry import ToolRegistry, Tool


def _search_documents(query: str, k: int = 5) -> dict:
    """
    Search the indexed medical documents for relevant passages.

    Args:
        query: The search query (medical question or keyword)
        k: Number of document chunks to retrieve (default 5)

    Returns:
        {
            "results": [
                {"text": str, "source": str, "page": int, "category": str, "score": float}
            ],
            "num_results": int
        }
    """
    # Import here to avoid circular import at module level
    from rag.retriever import retriever

    k   = max(1, min(int(k), 10))    # clamp between 1 and 10
    docs = retriever.retrieve(query, k=k)

    results = [
        {
            "text":     d["text"],
            "source":   d["source"],
            "page":     d.get("page", 1),
            "category": d.get("category", "General"),
            "score":    d.get("score", 0.0),
        }
        for d in docs
    ]

    return {
        "results":     results,
        "num_results": len(results),
    }


def register_search_tool(registry: ToolRegistry) -> None:
    registry.register(Tool(
        name="search_documents",
        description=(
            "Search the indexed medical document library for passages relevant to "
            "a query. Returns ranked text chunks with source file, page, and relevance score. "
            "Use this whenever you need factual medical information from the knowledge base."
        ),
        func=_search_documents,
        parameters={
            "query": "The search query string (e.g. 'symptoms of diabetes')",
            "k":     "Number of document chunks to retrieve (integer, 1-10, default 5)",
        },
    ))
