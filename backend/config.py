"""
config.py -- Central configuration for MedAssist-AI Agentic System
"""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Paths
BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DATA_DIR        = os.path.join(BASE_DIR, "data", "processed_documents")
RAW_DIR         = os.path.join(BASE_DIR, "data", "raw_documents")
UPLOAD_DIR      = os.path.join(BASE_DIR, "backend", "uploads")

FAISS_FILE      = os.path.join(DATA_DIR, "faiss_index.bin")
DOCUMENTS_FILE  = os.path.join(DATA_DIR, "documents.pkl")
BM25_FILE       = os.path.join(DATA_DIR, "bm25_index.pkl")

# RAG
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 600
CHUNK_OVERLAP   = 120
TOP_K_RETRIEVE  = 8

# Agent
MAX_ITERATIONS  = int(os.getenv("MAX_AGENT_ITERATIONS", "5"))
MAX_HISTORY     = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))

# CORS
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
