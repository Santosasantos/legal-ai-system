# ⚖️ Legal AI System — Pearson Specter Litt

> Document Understanding, Grounded Drafting, and Improvement from Edits

A production-quality AI pipeline that ingests messy legal documents, extracts structured information, retrieves grounded evidence, generates legal drafts, and continuously improves from operator edits.

---

## ⚡ Quick Start — Up and Running in 2 Minutes

### Prerequisites
- Docker + Docker Compose
- A free Gemini API key (takes 30 seconds to get)

### Step 1 — Get a free Gemini API key

Go to **https://aistudio.google.com/app/apikey** → click "Create API key" → copy it.

No credit card required. Free tier: 15 requests/min, 1M tokens/day.

### Step 2 — Configure

```bash
git clone <your-repo>
cd legal-ai-system
cp .env.example .env
```

Open `.env` and replace `your-gemini-api-key-here` with your key:

```bash
GEMINI_API_KEY=AIza...your-key-here
```

### Step 3 — Start

```bash
docker compose up -d
```

This pulls two small images (ChromaDB ~200MB, backend ~500MB) and starts:
- **Backend** — FastAPI on port 8000
- **ChromaDB** — vector store on port 8001
- **UI** — Streamlit on port 8501

> No model downloads. No GPU required. Ready in ~30 seconds.

### Step 4 — Generate sample documents

```bash
docker compose exec backend python scripts/generate_sample_docs.py
```

### Step 5 — Open the UI

```
http://localhost:8501
```

Or explore the API directly:
```
http://localhost:8000/docs
```

---

## Switching LLM Provider

Change **one line** in `.env` and restart the backend:

```bash
# Gemini (default — free, no credit card)
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Local Ollama + Qwen2.5 (requires ~5GB download, no API key)
LLM_PROVIDER=ollama
# Then run: docker compose --profile ollama up -d
```

```bash
docker compose restart backend
```

---

## API Reference

Base URL: `http://localhost:8000/api/v1`
Interactive docs: `http://localhost:8000/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Upload and process a PDF document |
| POST | `/draft` | Generate a grounded legal draft |
| POST | `/edit` | Submit operator edit → extract preference rules |
| GET | `/retrieve` | Semantic search with metadata filtering |
| GET | `/documents` | List all processed documents |
| GET | `/documents/{id}` | Get document details |
| GET | `/chunks/{id}` | Inspect a specific chunk (evidence tracing) |
| GET | `/rules` | List learned preference rules |
| DELETE | `/rules/{id}` | Deactivate a preference rule |
| GET | `/health` | System health check |
| GET | `/stats` | System statistics |

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.

---

## Key Documents for Reviewers

| Document | What it covers |
|----------|---------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Full system design, data flow diagrams, design decisions |
| [ASSUMPTIONS_AND_TRADEOFFS.md](ASSUMPTIONS_AND_TRADEOFFS.md) | Every design choice explained with alternatives considered |
| [data/outputs/SAMPLE_OUTPUTS.md](data/outputs/SAMPLE_OUTPUTS.md) | Guide to pre-generated sample inputs and outputs |
| [data/outputs/sample_ingest_response.json](data/outputs/sample_ingest_response.json) | Example document ingestion result |
| [data/outputs/sample_draft_output.json](data/outputs/sample_draft_output.json) | Example generated draft with evidence links |
| [data/outputs/sample_edit_response.json](data/outputs/sample_edit_response.json) | Example operator edit → extracted preference rules |
| [data/outputs/evaluation_results.json](data/outputs/evaluation_results.json) | Full evaluation run with metrics |

---

## Project Structure

```
legal-ai-system/
├── app/
│   ├── core/
│   │   ├── config.py              # All configuration, env-driven
│   │   └── logging.py             # Structured logging
│   ├── models/
│   │   └── schemas.py             # Pydantic schemas (single source of truth)
│   ├── services/
│   │   ├── llm_provider.py        # Provider abstraction (Gemini/OpenAI/Ollama/Anthropic)
│   │   ├── document_processor.py  # OCR + vision + structured extraction
│   │   ├── chunker.py             # Semantic chunking with rich metadata
│   │   ├── vector_store.py        # ChromaDB + context-aware retrieval
│   │   ├── draft_generator.py     # Grounded draft generation
│   │   └── preference_store.py    # Operator edit learning loop
│   ├── api/
│   │   └── routes.py              # FastAPI endpoints
│   ├── utils/
│   │   └── document_store.py      # Document/draft persistence
│   └── main.py                    # FastAPI app entry point
├── ui/
│   └── app.py                     # Streamlit UI
├── scripts/
│   ├── generate_sample_docs.py    # Synthetic legal document generator
│   └── evaluate.py                # End-to-end evaluation
├── data/
│   ├── sample_documents/          # Input PDFs
│   ├── outputs/                   # Processed docs + drafts (JSON)
│   └── feedback/                  # Preference rules store
├── Dockerfile.backend
├── Dockerfile.ui
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
