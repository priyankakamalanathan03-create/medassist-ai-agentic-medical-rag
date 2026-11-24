# MedAssist-AI

**A multi-agent medical document assistant using hybrid retrieval and retrieval-augmented generation.**

MedAssist-AI lets users ask questions about a collection of medical PDFs. It combines lexical and dense retrieval, a bounded tool-using language-model workflow, page-level citations, and a React interface.

> **Scope and safety:** This is an educational research prototype, not a medical device or diagnostic system. Its citation and relevance indicators are heuristics, not proof of factual or clinical correctness. Do not use it to make health decisions or submit identifiable patient information.

## Features

- Hybrid retrieval with BM25, FAISS, and Reciprocal Rank Fusion (RRF).
- PDF ingestion that preserves source filename and page metadata.
- Deterministic routing to document Q&A, document summarization, or symptom-analysis workflows.
- Two bounded ReAct-style specialists with three registered tools:
  - `search_documents`
  - `summarize_document`
  - `check_symptoms`
- Streamed chat responses, sources, and agent/tool trace.
- Session-scoped conversation memory, document upload/deletion, analytics, and citation export.
- Separate prompt-clarity and citation/relevance heuristic indicators.

## Architecture

```text
React + Vite
     │ HTTP / Server-Sent Events
     ▼
FastAPI ──► deterministic intent router
                  │
          ┌───────┴────────┐
          ▼                ▼
    RAG specialist    Symptom specialist
          │                │
          └───────┬────────┘
                  ▼
              Tool registry
          ┌───────┼──────────┐
          ▼       ▼          ▼
       Search   Summary   Symptom match
          │
          ▼
 BM25 + FAISS + RRF over indexed PDF chunks
          │
          ▼
 Groq-hosted language model ──► cited answer + trace
```

The retriever and indexes run locally. **Answer generation calls the Groq API**, so the complete application is not offline and requires network access, a valid API key, model availability, and provider quota.

## Repository layout

```text
.
├── backend/
│   ├── agents/          # Orchestrator, specialists, and heuristic scorer
│   ├── api/             # Chat, upload, analytics, and citation endpoints
│   ├── llm/             # Groq client and provider error handling
│   ├── memory/          # In-memory, per-session conversation window
│   ├── rag/             # PDF indexing and hybrid retrieval
│   ├── tests/           # Backend regression tests
│   └── tools/           # Search, summary, and symptom tools
├── data/
│   ├── raw_documents/   # User-supplied PDFs; not included in this repository
│   ├── processed_documents/ # Generated indexes; not included
│   └── evaluation/      # Optional evaluation data; not included
├── frontend/            # React 19 and Vite application
├── requirements.txt
├── start_backend.bat
└── start_frontend.bat
```

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or later (developed with Python 3.10)
- Node.js and npm
- A Groq API key and a model available to your Groq account
- Medical PDF files that you are permitted to use

## Run locally

### 1. Configure the backend

From the repository root, create a Python environment and install dependencies:

```powershell
python -m venv backend\venv
backend\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `backend/.env.example` to `backend/.env` and set your own key:

```text
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant
```

The model name is configurable and must be available to your account. Never commit or share `backend/.env`.
If chat reports a TLS certificate validation error, check the Windows trust store or ask your network administrator to verify HTTPS inspection for `api.groq.com`. Do not disable certificate verification.

### 2. Add documents and build the indexes

Place PDFs you have permission to process in `data/raw_documents/`. The project repository intentionally does not distribute its medical PDFs or generated indexes.

PDF indexing uses the `sentence-transformers/all-MiniLM-L6-v2` embedding model. It is loaded from the local Hugging Face cache without making network requests; the model must be available in that cache before indexing or uploading documents.

```powershell
Set-Location backend
python -m rag.indexer
```

The indexer creates the processed documents and retrieval indexes under `data/processed_documents/`.
The backend can also start before documents are indexed; in that case, retrieval returns no results until you upload a PDF through the UI or build an index.

### 3. Start the backend

From the `backend` directory:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API status endpoint is available at `http://127.0.0.1:8000/api/status`.

### 4. Start the frontend

Open a second terminal:

```powershell
Set-Location frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. The frontend uses `http://127.0.0.1:8000` as its default API URL. Set `VITE_API_BASE_URL` when the backend runs elsewhere. The root-level `start_backend.bat` and `start_frontend.bat` scripts are optional Windows launch helpers.

## Validation

Run backend regression tests from `backend/`:

```powershell
python -m unittest discover -s tests -v
```

Run frontend checks from `frontend/`:

```powershell
npm run lint
npm run build
```

The repository includes an optional MedQuAD retrieval script. Its current hit-rate measure only checks whether any chunks were returned; it is **not** answer accuracy, relevance, or clinical validation. No benchmark performance is claimed here.

## Evaluation and limitations

- Citation coverage checks whether answer units contain citation text; source matching checks whether cited file/page pairs appeared in retrieved sources. Neither checks whether the passage entails the claim.
- Topic-word overlap and answer-structure metrics are deterministic heuristics, not semantic or factuality evaluations.
- The symptom tool uses a fixed keyword map. Its match label is not a diagnosis or probability.
- The document-summary tool selects representative passages; the language model synthesizes the final summary.
- Retrieval quality depends on the supplied documents, PDF text extraction, chunking, and query wording.
- The current project has not been clinically validated and must not be used for diagnosis or treatment.
- Groq quota and API failures are surfaced to the user; a new API key does not itself increase an organization or model quota.

## Data and privacy

This repository excludes source PDFs, generated indexes, evaluation data, screenshots, build outputs, and credentials. Add only documents that you have permission to process and redistribute where applicable. Chat analytics stores submitted query text locally; do not enter identifiable patient data.
