# Agentic RAG over Technical PDFs

A production-shaped AI service that ingests technical PDFs and answers questions
about them using retrieval-augmented generation, with an agentic tool-calling
layer, an HTTP API, an automated evaluation harness, and structured observability.

## Architecture

```
Client
  │
  │ POST /ask
  ▼
FastAPI (api.py)
  │  per-request trace_id → structured JSON logs
  ▼
Agent loop (agent.py)          ← decides when and what to retrieve
  │  max 5 iterations
  │  tools: search_docs(), list_documents()
  ▼
Retriever (retriever.py)       ← two-stage retrieve-then-rerank
  ├─ 1. Vector search    (Chroma, cosine similarity, top-20)
  ├─ 2. BM25 keyword     (rank_bm25, top-20)
  ├─ 3. RRF merge        (Reciprocal Rank Fusion, dedup)
  └─ 4. Cross-encoder    (ms-marco-MiniLM-L-6-v2, top-5)
  ▼
Generator (generate.py)        ← prompt + context → grounded answer
  │  cite source page, say "I don't know" if not in context
  ▼
llm.py                         ← single provider wrapper (OpenAI)
                                  swap models in one file

──── offline ────────────────────────────────────────────────
Ingest (ingest.py)
  PDF files → pypdf → chunk (~500 tok, 50 overlap) → embed → Chroma
                                                        ↑
                                               data/store/ (gitignored)
```

## Status

- [x] Phase 0 — Project setup (config, provider wrapper, Makefile)
- [x] Phase 1 — Ingestion pipeline (load → chunk → embed → store)
- [x] Phase 2 — Retrieval (vector + BM25 hybrid + cross-encoder rerank)
- [x] Phase 3 — Generation with citations
- [x] Phase 4 — Agentic tool-calling loop
- [x] Phase 5 — FastAPI service
- [x] Phase 6 — Evaluation harness (LLM-as-judge + retrieval hit rate)
- [x] Phase 7 — Structured logging + per-request tracing
- [x] Phase 8 — Tests, Dockerfile, docs
- [x] Phase 9 — Frontend 

## Setup

Requires **Python 3.11+** and an [OpenAI API key](https://platform.openai.com/api-keys).

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# edit .env — set OPENAI_API_KEY=sk-...

# 3. Drop PDFs into data/pdfs/ then build the vector store
make ingest           # idempotent — safe to re-run
# make ingest-rebuild   # wipe and rebuild from scratch

# 4. Start the API
make api              # http://localhost:8000
                      # Swagger UI: http://localhost:8000/docs

# 5. Run evals
make eval

# 6. Run tests
make test
```

## API

| Method | Endpoint  | Description |
|--------|-----------|-------------|
| GET    | /health   | Readiness check — returns chunk count |
| POST   | /ask      | Answer a question (agent loop) |
| POST   | /ask?stream=true | Same, token-by-token SSE streaming |
| POST   | /ingest   | Re-ingest PDFs; accepts optional file upload |

**POST /ask** request / response:
```json
// request
{ "question": "How does SEVER remove corrupted gradients?" }

// response
{
  "answer": "SEVER uses a spectral approach ...",
  "sources": [{ "filename": "Sever A Robust Meta-Algorithm.pdf", "page": 1 }],
  "latency_ms": 4200.1,
  "iterations": 2,
  "tool_calls_made": ["search_docs({'query': 'SEVER corrupted gradients'})"]
}
```

## Docker

```bash
docker build -t rag-agent .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/data:/app/data \
  rag-agent
```

`data/` is mounted as a volume so PDFs and the vector store persist outside the
image and survive container restarts.

## File layout

```
src/
  config.py          settings loaded from .env
  llm.py             the only file that calls OpenAI (complete + embed)
  ingest.py          PDF → chunks → embeddings → Chroma
  retriever.py       vector + BM25 hybrid + cross-encoder rerank
  generate.py        build context prompt → grounded answer with citations
  agent.py           tool-calling loop (decides when to retrieve)
  api.py             FastAPI service (3 endpoints, lifespan, tracing)
  logging_conf.py    structlog: JSON or pretty, per-request trace_id
evals/
  golden_set.json    15 Q/A pairs (10 answerable, 5 "not in docs")
  run_eval.py        LLM-as-judge scoring + retrieval hit rate
tests/
  test_ingest.py     chunk logic, metadata, id determinism (unit)
  test_retriever.py  RRF unit tests + live store integration tests
data/
  pdfs/              drop your PDFs here (gitignored)
  store/             Chroma vector DB (gitignored, rebuildable)
```

## Design decisions

**Provider behind a thin wrapper (`llm.py`).**
`complete()` and `embed()` are the only two functions that touch the OpenAI SDK.
Swapping to Claude or adding a fallback is a one-file change, not a refactor.

**Chunking strategy matters more than model choice.**
We split on paragraph boundaries, target ~500 tokens, and keep 50-token overlap
so an answer straddling a boundary isn't lost. Page numbers are stored as
metadata so every answer can cite the source page.

**Two-stage retrieve-then-rerank.**
Vector search misses exact terms (error codes, acronyms, paper titles). BM25
catches them. Reciprocal Rank Fusion merges the two lists without needing score
normalisation. The cross-encoder only runs on the shortlist (~40 candidates):
too slow to run on all chunks, accurate enough to matter on the shortlist.

**Agent, not just RAG.**
The agent decides whether to retrieve, what to search for, and whether to search
again. Built by hand (raw tool-calling) so the failure modes are visible: runaway
loops (capped at 5 iterations), tool errors (returned as strings so the agent can
react), and token cost (logged on every LLM call).

**Eval first, then iterate.**
A 15-question golden set with LLM-as-judge scoring separates retrieval quality
(hit rate) from generation quality (correctness/faithfulness). Changing chunk
size or prompts gives a number, not a vibe.

**Structured logging with trace IDs.**
Every LLM call logs latency and token counts. Every retrieval call logs which
chunks were returned. Every request gets a UUID trace_id that appears on every
log line for that request — grep for one ID to follow a question end-to-end.

Note: Python's `run_in_executor` does not copy `contextvars` to thread-pool
workers in this environment. The trace_id is captured before dispatch and
explicitly re-set inside each thread via a closure.
