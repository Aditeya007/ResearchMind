# 🧠 ResearchMind

> A multi-source Retrieval Augmented Generation (RAG) system with hybrid retrieval, cross-encoder reranking, adaptive query routing, per session knowledge isolation, and automated faithfulness evaluation.

---

## What is this?

ResearchMind is a document-grounded question answering system. You feed it PDFs or web pages, and it answers questions strictly based on what you gave it — not from the LLM's general knowledge.

It is **not a wrapper around ChatGPT**. The core of the project is the retrieval pipeline — combining two retrieval strategies, re-ranking results with a cross-encoder, and automatically measuring answer quality using RAGAS.

---

## Features

| Feature | Description |
|---|---|
| Hybrid Retrieval | Dense vector search (semantic) + BM25 (keyword) fused via Reciprocal Rank Fusion |
| Cross-Encoder Reranking | Second-pass scoring of retrieved chunks as query-document pairs |
| Adaptive Query Routing | Classifies query type (factual/summary/comparison/general) and adjusts retrieval depth |
| Per-Session Isolation | Each user gets their own ChromaDB collection and BM25 index — documents never leak between users |
| Conversation Memory | Last 4 exchanges remembered per session for follow-up questions |
| Confidence Scoring | Low/Medium/High confidence badge on every answer based on reranker scores |
| Source Citations | Every answer shows which document it came from |
| Document Manager | View and delete ingested sources per session |
| RAGAS Evaluation | Background faithfulness, answer relevance, and context precision scoring on every query |
| Ablation Study | Live comparison of dense-only vs hybrid retrieval with metric tables |
| FastAPI Backend | Full REST API with Swagger UI at `/docs` |
| Streamlit Frontend | Clean chat interface with sidebar document management |

---

## Architecture

```
User Query
    │
    ▼
Query Router (classify: factual / summary / comparison / general)
    │
    ├──► Dense Retrieval (ChromaDB + sentence-transformers)
    │
    ├──► Sparse Retrieval (BM25Okapi)
    │
    ▼
Reciprocal Rank Fusion (RRF)
    │
    ▼
Cross-Encoder Reranker (ms-marco-MiniLM)
    │
    ▼
Prompt Builder (context + conversation history)
    │
    ▼
LLM Generation (Groq / llama-3.1-8b-instant)
    │
    ▼
Answer + Confidence Score + Sources
    │
    ▼ (background)
RAGAS Evaluation → SQLite Logging
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Sparse Retrieval | rank-bm25 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq (llama-3.1-8b-instant) — free tier |
| Evaluation | RAGAS |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Database | SQLite + SQLAlchemy |
| Config | Pydantic Settings |

---

## Ablation Study Results

Hybrid retrieval vs Dense-only on the same query set:

| Metric | Dense Only | Hybrid |
|---|---|---|
| Faithfulness | 0.78 | 0.91 |
| Answer Relevance | 0.81 | 0.89 |
| Context Precision | 0.75 | 0.86 |

Hybrid retrieval outperforms dense-only across all three RAGAS metrics.

---

## Project Structure

```
researchmind/
├── ingestion/
│   ├── pdf_loader.py       # PDF parsing via PyMuPDF
│   ├── web_loader.py       # Web scraping via BeautifulSoup
│   ├── chunker.py          # Recursive text chunking
│   └── pipeline.py         # Unified ingestion entry point
├── retrieval/
│   ├── embedder.py         # Sentence-transformer embeddings
│   ├── vector_store.py     # ChromaDB interface (session-scoped)
│   ├── bm25_retriever.py   # BM25 sparse retrieval (session-scoped)
│   ├── hybrid_retriever.py # RRF fusion of dense + sparse
│   ├── reranker.py         # Cross-encoder reranking
│   └── query_router.py     # Query classification + retrieval orchestration
├── generation/
│   ├── llm_client.py       # Groq API wrapper
│   ├── prompt_builder.py   # Context-aware prompt templates
│   └── rag_chain.py        # Full RAG pipeline + memory + confidence
├── evaluation/
│   ├── ragas_eval.py       # RAGAS faithfulness/relevance/precision
│   ├── metrics_logger.py   # SQLite logging of scores and feedback
│   └── ablation.py         # Dense vs hybrid comparison
├── api/
│   ├── main.py             # FastAPI app entry point
│   ├── schemas.py          # Pydantic request/response models
│   └── routes/
│       ├── ingest.py       # POST /ingest/url, POST /ingest/pdf
│       ├── query.py        # POST /query/
│       └── feedback.py     # POST /feedback/, GET /feedback/metrics
├── db/
│   ├── models.py           # SQLAlchemy models
│   └── session.py          # DB session factory
├── frontend/
│   └── app.py              # Streamlit chat UI
├── configs/
│   └── settings.py         # Pydantic BaseSettings
├── .env.example
├── requirements.txt
└── Dockerfile
```

---

## Setup & Running

### 1. Clone the repo

```bash
git clone https://github.com/your-username/researchmind.git
cd researchmind
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Copy `.env.example` to `.env` and fill in your Groq API key (free at [console.groq.com](https://console.groq.com)):

```
GROQ_API_KEY=your_key_here
LLM_MODEL_NAME=llama-3.1-8b-instant
```

### 5. Run the API

```bash
python -m api.main
```

### 6. Run the frontend (new terminal)

```bash
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## API Reference

Full interactive docs available at `http://localhost:8000/docs`

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ingest/url` | Ingest a web page |
| POST | `/ingest/pdf` | Upload and ingest a PDF |
| GET | `/ingest/sources` | List ingested sources for a session |
| DELETE | `/ingest/source` | Delete a source from a session |
| POST | `/query/` | Ask a question |
| POST | `/feedback/` | Submit thumbs up/down |
| GET | `/feedback/metrics` | Get average RAGAS scores |
| GET | `/health` | Health check |

---

## Limitations

- Scanned/image-based PDFs are not supported (text-based only)
- Sessions are in-memory — documents must be re-ingested after API restart
- Only PDF and web URL sources supported
- Answers are not streamed token by token

---

## Author

**Aditeya Mitra**
