# Verdact

**Evidence-grade compliance analysis for security policy documents.**

Verdact ingests PDF policy documents, indexes them with hybrid semantic + keyword search, and generates structured evidence reports — complete with citations, source excerpts, and gap analysis — using a locally-hosted LLM. No data leaves your machine.

---

## What it does

You upload a SOC 2, ISO 27001, or internal security policy PDF. You ask a natural-language question like:

> *"Show evidence that MFA is required for privileged accounts."*

Verdact retrieves the most relevant policy sections, passes them to a local LLM, and returns a structured report:

- **Summary** — a plain-language answer grounded in the document
- **Evidence** — each claim tied to an exact filename, section title, page number, and source excerpt
- **Gaps** — policy areas the document doesn't address

Reports export as JSON or PDF.

---

## Architecture

```
┌──────────────┐     PDF upload     ┌─────────────────────────────────────────┐
│  Streamlit   │ ────────────────── │              FastAPI (main.py)          │
│   UI         │ ◄─── report JSON ─ │                                         │
└──────────────┘                    │  /ingest  →  ingestor.py                │
                                    │               ├── chunker.py (PyMuPDF)  │
                                    │               ├── embedder.py (SBERT)   │
                                    │               └── Qdrant upsert         │
                                    │                                         │
                                    │  /investigate →  searcher.py            │
                                    │                  ├── dense (SBERT)      │
                                    │                  ├── sparse (BM25)      │
                                    │                  └── RRF fusion         │
                                    │               →  generator.py           │
                                    │                  └── Ollama (llama3.1)  │
                                    └─────────────────────────────────────────┘
```

**Ingestion pipeline**

1. PyMuPDF extracts text block-by-block, detecting headings by font size/weight
2. Sections are chunked at 512 tokens with 80-token sentence-aware overlap
3. Each chunk is embedded with `all-mpnet-base-v2` (dense, 768d) and BM25 (sparse)
4. Both vectors are upserted to Qdrant under a per-session `ingestion_version` key

**Retrieval**

Hybrid search runs a dense prefetch and a BM25 prefetch in parallel, then merges via Reciprocal Rank Fusion (RRF). Results below a score threshold are discarded before the LLM sees them — preventing hallucination on low-confidence retrievals.

**Generation**

A strict system prompt instructs the model to cite only what appears in the retrieved context. Raw output is parsed with `json-repair` as a fallback, and any non-JSON preamble is stripped with `rfind`-based extraction.

---

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| PDF parsing | PyMuPDF (fitz) |
| Chunking | tiktoken (cl100k_base), custom sentence-aware splitter |
| Dense embeddings | `sentence-transformers/all-mpnet-base-v2` |
| Sparse embeddings | fastembed BM25 (`Qdrant/bm25`) |
| Vector store | Qdrant (hybrid collections) |
| LLM | Ollama — `llama3.1:8b` (swappable via env var) |
| PDF export | ReportLab |
| Registry | JSON flat file (metadata cache; Qdrant is source of truth) |

---

## Prerequisites

- Python 3.11+
- [Qdrant](https://qdrant.tech/documentation/quick-start/) running locally on port 6333
- [Ollama](https://ollama.com/) running locally with `llama3.1:8b` pulled

```bash
# Pull the model once
ollama pull llama3.1:8b

# Start Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant
```

---

## Setup

```bash
git clone https://github.com/your-username/verdact.git
cd verdact

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env             # Edit if your Qdrant/Ollama URLs differ
```

---

## Running

Open two terminals:

```bash
# Terminal 1 — API server
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run ui/app.py
```

Then open [http://localhost:8501](http://localhost:8501).

---

## Project structure

```
verdact/
├── api/
│   └── main.py              # FastAPI routes + PDF export
├── ingestion/
│   ├── chunker.py           # PDF → structured Chunk objects
│   ├── embedder.py          # SentenceTransformer dense embeddings
│   └── ingestor.py          # Qdrant collection setup + upsert pipeline
├── retrieval/
│   └── searcher.py          # Hybrid RRF search + score filtering
├── generation/
│   └── generator.py         # Ollama prompt + JSON extraction + repair
├── persistence/
│   └── registry.py          # Flat-file ingestion version registry
├── ui/
│   └── app.py               # Streamlit frontend
├── data/
│   └── documents/           # Upload target — gitignored
├── .env.example
├── requirements.txt
└── README.md
```

---

## Key design decisions

**Why Qdrant as source of truth?** Early versions maintained a parallel JSON registry for existence checks. Dual-registry patterns create sync bugs (e.g. a crash mid-ingest leaves the registry stale but Qdrant populated). The registry is now metadata-only; all existence checks query Qdrant directly.

**Why RRF over a weighted sum?** RRF is rank-based, not score-based, which makes it insensitive to the scale differences between cosine similarity (dense) and BM25 scores (sparse). No manual weight tuning required.

**Why score-threshold filtering before the LLM?** Passing low-relevance chunks to the model reliably produces hallucinated citations. An empty retrieval result triggers a clean "no relevant policy found" response rather than fabricated evidence.

**Why `json_repair` as fallback?** `llama3.1:8b` occasionally emits a conversational prefix before the JSON object. `rfind` on `}` handles truncated output; `json_repair` handles partial key omissions. Both are preferable to returning a 500 to the user.

---

## Configuration

All tuneable values are environment variables (see `.env.example`):

| Variable | Default | Effect |
|---|---|---|
| `VERDACT_MODEL` | `llama3.1:8b` | Ollama model name |
| `OLLAMA_READ_TIMEOUT` | `600` | Seconds before LLM call times out |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance URL |
| `COLLECTION_NAME` | `verdact_policies` | Qdrant collection name |
| `RRF_SCORE_THRESHOLD` | `0.02` | Minimum RRF score to pass a chunk to the LLM |
| `UPSERT_BATCH_SIZE` | `100` | Chunks per Qdrant upsert batch |

---

## Limitations / roadmap

- Single-document investigation per query (multi-doc cross-referencing planned)
- No authentication on the API — intended for local/internal use only
- Sentence splitter is regex-based; spaCy integration would improve chunking quality on dense regulatory text
- Table extraction is heuristic (span-count based); a dedicated table parser would improve recall on matrix-style controls
