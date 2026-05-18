# ResearchMind

A document-grounded QA prototype that combines sparse and dense retrieval with reranking and small-scale evaluation tooling. The code is implemented as a FastAPI backend with a Streamlit frontend for experimentation.

This repository is a research-oriented engineering project built for local use and experimentation. It demonstrates a hybrid retrieval pipeline (BM25 + vector search), Reciprocal Rank Fusion (RRF), cross-encoder reranking, session-scoped storage, and basic evaluation support using RAGAS.

---

## Quick summary

- Hybrid retrieval: BM25 (keyword) + Chroma vector search (semantic), fused by RRF.
- Reranking: cross-encoder re-scores retrieved chunks before generation.
- Query routing: simple heuristic classifier adjusts retrieval depth for different query types.
- Session-scoped datasets: ingested documents are stored per `session_id` (Chroma collections + an in-memory BM25 index).
- Backend + UI: FastAPI provides the API, Streamlit provides a minimal chat UI.
- Evaluation: scripts to run RAGAS-based evaluation and a local ablation utility (requires an LLM key).

---

## Features (concise)

- Hybrid retrieval (BM25 + dense vectors) fused with Reciprocal Rank Fusion (RRF).
- Cross-encoder reranking for a second-pass relevance ordering.
- Heuristic query router (factual / summary / comparison / general) that adjusts retrieval depth.
- Session-scoped storage: each session uses a dedicated Chroma collection and an in-memory BM25 index.
- Conversation memory: the last 4 exchanges are included in prompts for follow-up context.
- Confidence labels: simple high/medium/low confidence derived from top chunk scores.
- Source citations: responses return the list of document sources used to build the answer.
- FastAPI backend with simple REST endpoints; Streamlit frontend for interactive use.
- Evaluation helpers: RAGAS-based evaluation and an ablation script to compare dense vs hybrid retrieval (requires configuration).

---

## Architecture 

1. Ingest: PDFs or URLs → text extraction → chunking → store chunks
2. Retrieval: vector search (Chroma) + BM25 search (rank-bm25)
3. Fusion: Reciprocal Rank Fusion merges dense + sparse results
4. Rerank: Cross-encoder scores query-document pairs and reorders results
5. Generation: prompt builder includes recent conversation; backend calls configured LLM
6. Post-processing: response includes sources and a confidence label; optional evaluation runs in the evaluation scripts

The code maps to these components in `ingestion/`, `retrieval/`, `generation/`, `api/`, and `frontend/`.

---

## Tech stack (what's actually used)

- Embeddings: `sentence-transformers` (configurable via `configs/settings.py`)
- Vector store: `chromadb` (persistent client; path controlled by `CHROMA_PERSIST_DIR`)
- Sparse retrieval: `rank_bm25` (in-memory per-session index)
- Reranker: `sentence_transformers.CrossEncoder` (model configured by `RERANKER_MODEL`)
- LLM client: thin wrapper around `groq` (the default provider in settings). The client requires `GROQ_API_KEY` for the default configuration.
- API: `FastAPI` (API entry point in `api/main.py`)
- Frontend: `Streamlit` app at `frontend/app.py`
- Evaluation: optional `ragas`-based components under `evaluation/`

---

## Quickstart (local, realistic)

Requirements: Python 3.10+ recommended. This project uses a local Chroma directory by default and an in-memory BM25 index.

1) Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\Activate.ps1   # Windows PowerShell
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) Configure credentials (only if you want to use the default Groq LLM provider)

Create a `.env` file in the project root (or set environment variables). If you use the default LLM configuration, set at least:

```
GROQ_API_KEY=your_groq_api_key_here
```

If you do not set an LLM key, you can still run ingestion and local retrieval, but generation/evaluation calls will fail until a working LLM client is configured.

4) Run the API

```powershell
python -m api.main
```

5) Run the Streamlit frontend (separate terminal)

```powershell
streamlit run frontend/app.py
```

The Streamlit UI uses a browser session-local `session_id` to scope ingestions. The API also accepts `session_id` parameters on relevant endpoints.

---

## Important implementation notes

- Session scope: Vector embeddings are stored in Chroma collections named with a session prefix; BM25 indices are kept in memory per `session_id` (see `retrieval/vector_store.py` and `retrieval/bm25_retriever.py`). BM25 indexes are lost on process restart; Chroma collections persist to disk if `CHROMA_PERSIST_DIR` is set.
- Conversation memory lives in `generation/rag_chain.py` and keeps the last 4 exchanges per session.
- Confidence is derived from the top chunk's reranker or similarity score (thresholds implemented in `generation/rag_chain.py`).
- Evaluation and ablation scripts exist under `evaluation/`; they call out to an LLM and embeddings during evaluation and therefore require proper credentials to run end-to-end.

---

## Usage (API endpoints)

The FastAPI app exposes simple endpoints (see `api/routes/`):

- `POST /ingest/url` — ingest a URL into a session
- `POST /ingest/pdf` — upload and ingest a PDF into a session
- `GET /ingest/sources` — list sources for a session
- `DELETE /ingest/source` — remove a source from a session
- `POST /query/` — run a query against a session's documents
- `POST /feedback/` — submit simple feedback

Start the API and open `/docs` for interactive API documentation.

---

## Limitations

- BM25 index is in-memory: it does not survive process restarts; Chroma storage is persistent to disk.
- This project is designed for local experiments, not for multi-tenant production use. Session scoping is intended for convenience rather than strict isolation guarantees.
- Evaluation components (RAGAS) and the ablation utility require an accessible LLM and embedding provider to run; they are provided as tools for experiments rather than production monitoring.
- OCR/scanned PDFs are not supported; inputs must be text-extractable PDFs or web pages.

---

## Future improvements (short, realistic)

- Add optional persistent BM25 storage or re-indexing on startup.
- Add configurable LLM providers and a pluggable interface for local models.
- Add streaming responses and richer UI controls for long answers.
- Harden session isolation and access controls for multi-tenant deployments.

---

## What I learned

- Practical trade-offs when combining sparse and dense retrieval (engineerability vs. recall).
- How RRF and a cross-encoder reranker can be combined in a small pipeline to improve ordering.
- The value of keeping a small local evaluation loop (RAGAS) to compare retrieval strategies on targeted examples.
- Simple session scoping patterns for experimental multi-document QA prototypes.

---

## Development pointers

- Configuration lives in `configs/settings.py`. Adjust `CHROMA_PERSIST_DIR`, `RERANKER_MODEL`, and `EMBEDDING_MODEL` there.
- Useful entry points:
  - `api/main.py` — FastAPI app
  - `frontend/app.py` — Streamlit UI
  - `evaluation/ablation.py` — compare dense vs hybrid pipelines on sample queries

---

## Author

Aditeya Mitra

