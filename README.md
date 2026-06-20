# Agentic RAG over Technical PDFs

A service that ingests technical PDFs and answers questions about them using
retrieval-augmented generation, with an agentic tool-calling layer, an HTTP API,
and an automated evaluation harness.

Built phase by phase from [INSTRUCTIONS.md](INSTRUCTIONS.md).

## Status

- [x] **Phase 0** — Project setup (config, provider wrapper, Makefile)
- [x] **Phase 1** — Ingestion pipeline (load → chunk → embed → store)
- [ ] Phase 2 — Retrieval (vector + BM25 hybrid + rerank)
- [ ] Phase 3 — Generation with citations
- [ ] Phase 4 — Agentic layer
- [ ] Phase 5 — FastAPI service
- [ ] Phase 6 — Evaluation harness
- [ ] Phase 7 — Observability
- [ ] Phase 8 — Tests / Docker / docs
- [ ] Phase 9 — Frontend (optional)

## Setup

Requires **Python 3.11+**.

```bash
# 1. Install dependencies
make install            # or: python3.11 -m pip install -r requirements.txt

# 2. Configure your API key
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...

# 3. Add PDFs
# drop 2–3 technical PDFs into data/pdfs/

# 4. Build the vector store
make ingest             # or: python -m src.ingest
# re-run anytime; it's idempotent. Use `make ingest-rebuild` to wipe & rebuild.
```

## Layout

```
src/config.py   settings loaded from .env (models, chunk sizes, paths)
src/llm.py      the only place that talks to OpenAI: complete() + embed()
src/ingest.py   the ingestion pipeline (Phase 1)
data/pdfs/      drop your PDFs here (gitignored)
data/store/     persisted Chroma vector DB (gitignored, rebuildable)
```

## Design decisions

- **Provider behind a thin wrapper** (`llm.py`): swapping models or adding a
  fallback is a one-file change, not a refactor.
- **Chunking** splits on paragraph boundaries, targets ~500 tokens with ~50-token
  overlap, and keeps `{source, page, chunk_index}` metadata so answers can cite
  the source page. Overlap prevents losing an answer that straddles a boundary.
- **Idempotent ingest**: each chunk's id is a content hash, so re-running upserts
  instead of duplicating. `--rebuild` clears stale chunks from removed PDFs.
