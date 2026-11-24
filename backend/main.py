"""
main.py -- MedAssist-AI Agentic System -- FastAPI Application Entry Point

Architecture:
  User -> Orchestrator -> [Intent Classifier] -> [RAG Agent | Symptom Agent]
                       -> [Quality Metrics Agent] -> Answer + Scores -> User

Run with:
    cd backend
    uvicorn main:app --reload --port 8000
"""
import sys

# Ensure UTF-8 output on Windows
if (sys.stdout.encoding or "").lower().replace("-", "") != "utf8" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import CORS_ORIGINS
from api.chat   import router as chat_router
from api.upload import router as upload_router

# Import tools -- side-effect registers all tools into the registry
import tools  # noqa: F401

app = FastAPI(
    title="MedAssist-AI Agentic System",
    description=(
        "Final Year Project -- Multi-Agent Medical RAG System. "
        "Uses ReAct agents, hybrid BM25+FAISS retrieval, "
        "conversation memory, and transparent prompt and answer-evidence metrics."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router,   prefix="/api")
app.include_router(upload_router, prefix="/api")
# analytics endpoint – usage statistics
from api.analytics import router as analytics_router
app.include_router(analytics_router, prefix="/api")
# citations endpoint – BibTeX export for a session
from api.citations import router as citations_router
app.include_router(citations_router, prefix="/api")


@app.get("/")
def home():
    return {
        "system":      "MedAssist-AI",
        "version":     "2.0.0",
        "status":      "running",
        "description": "Multi-agent medical document assistant with RAG + LLM + ReAct reasoning",
    }


@app.get("/api/status")
def get_status():
    from agents.rag_agent import rag_agent
    from agents.symptom_agent import symptom_agent
    from llm.groq_client import llm_client
    from rag.retriever   import retriever
    from tools           import tool_registry

    rate_limit_status = llm_client.get_rate_limit_status()
    agents_list = [rag_agent.agent_name, symptom_agent.agent_name]

    return {
        "llm_model":        llm_client.get_model_name(),
        "llm_configured":   llm_client.is_configured(),
        "llm_rate_limited": rate_limit_status["limited"],
        "llm_retry_after_seconds": rate_limit_status["retry_after_seconds"],
        "indexed_chunks":   len(retriever.documents),
        "total_chunks":     len(retriever.documents),
        "tools_registered": len(tool_registry.names),
        "tools":            tool_registry.names,
        "agents_loaded":    len(agents_list),
        "agents":           agents_list,
        "agent_details": [
            {
                "name": agent.agent_name,
                "type": "Groq ReAct specialist",
                "max_iterations": agent.MAX_ITERATIONS,
            }
            for agent in (rag_agent, symptom_agent)
        ],
        "orchestrator": {
            "name": "Medical Orchestrator",
            "routing": "Deterministic intent rules",
        },
        "quality_scorer": {
            "name": "Quality Metrics Scorer",
            "type": "Deterministic heuristic (not an LLM agent)",
        },
        "tool_details": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tool_registry.tools
        ],
        "retriever_ready":  len(retriever.documents) > 0,
        "memory":           "Sliding Window (per session)",
        "retrieval":        "Hybrid BM25 + FAISS with Reciprocal Rank Fusion",
    }
