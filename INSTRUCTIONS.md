# Build Spec: Agentic RAG Service over Technical PDFs

This is a build specification for an end-to-end, production-shaped AI system. Hand this file to Claude Code and build it phase by phase. Do **not** build everything in one shot. Finish a phase, run it, understand it, then move on.

The goal is two things at once:
1. A real, working repo you can point at.
2. Enough hands-on understanding that you can talk about every piece in a technical interview without bluffing.

After each phase there is a **"Talk about it"** note. That is the line you should be able to say out loud and defend. If you can't, you didn't really finish the phase.

---

## What we're building

A service that ingests technical PDFs, answers questions about them using retrieval-augmented generation, decides *when* to retrieve using an agent loop, serves answers over an HTTP API, and measures its own answer quality with an automated eval set.

It is not just a RAG script. It covers: retrieval + reranking, an agentic tool-calling layer, a FastAPI service, an evaluation harness, and basic observability. Each of these maps to a line in the job description.

### Target architecture

```
rag-agent/
  README.md
  pyproject.toml            # or requirements.txt
  .env.example              # API keys, never commit real .env
  Makefile                  # convenience commands
  data/
    pdfs/                   # drop technical PDFs here
    store/                  # persisted vector DB (gitignored)
  src/
    config.py               # settings, loaded from env
    llm.py                  # thin provider wrapper (one place to swap models)
    ingest.py               # load -> chunk -> embed -> store
    retriever.py            # vector search + BM25 hybrid + rerank
    generate.py             # build context prompt -> answer with citations
    agent.py                # tool-calling loop (decides when to retrieve)
    api.py                  # FastAPI app
    logging_conf.py         # structured logging / tracing
  evals/
    golden_set.json         # questions + reference answers
    run_eval.py             # LLM-as-judge scoring + regression
  tests/
    test_retriever.py
    test_ingest.py
  Dockerfile                # optional, Phase 8
  frontend/                 # optional stretch, Phase 9 (React + TS)
```

### Tech choices (defaults, all cheap)

- **Language:** Python 3.11+
- **Embeddings:** OpenAI `text-embedding-3-small` (about $0.02 / 1M tokens). Local alternative: `sentence-transformers` (`BAAI/bge-small-en-v1.5`) if you want zero API cost.
- **Vector store:** Chroma (local, persistent, dead simple). Mention pgvector / Qdrant / Pinecone as the production-scale options.
- **LLM:** OpenAI `gpt-4o-mini` for generation and judging (cheap, fast). Wrap it so you can swap to Claude or another provider in one file.
- **Reranker:** local cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`. Mention Cohere Rerank as the API option.
- **Keyword search (for hybrid):** `rank_bm25`.
- **API:** FastAPI + Uvicorn.
- **Logging:** `structlog` (or stdlib logging configured for JSON). Optional: Langfuse for LLM tracing.
- **Tests:** pytest.

Keep total spend under a couple of euros. This whole thing is cents to run.

---

## Phase 0 — Project setup

**Build:**
- Initialise the repo structure above.
- `pyproject.toml` (or `requirements.txt`) with: `openai`, `chromadb`, `pypdf`, `rank_bm25`, `sentence-transformers`, `fastapi`, `uvicorn`, `structlog`, `python-dotenv`, `pytest`, `tiktoken`.
- `.env.example` with `OPENAI_API_KEY=`, `LLM_MODEL=gpt-4o-mini`, `EMBED_MODEL=text-embedding-3-small`. Real `.env` is gitignored.
- `config.py` that loads settings from env with sensible defaults.
- `llm.py`: one function `complete(messages, tools=None)` and one `embed(texts)`. Everything else calls these. This is the single place model choices live.
- A `Makefile` with `make ingest`, `make api`, `make eval`, `make test`.
- A `README.md` you fill in as you go.

**Talk about it:** "I kept the provider behind a thin wrapper so swapping models or adding fallbacks is a one-file change, not a refactor."

---

## Phase 1 — Ingestion pipeline

This is where most RAG quality is won or lost. Take it seriously.

**Build (`ingest.py`):**
1. Load every PDF in `data/pdfs/` with `pypdf`, extract text per page, keep page numbers (you'll cite them later).
2. **Chunk** the text. Start with ~500-token chunks and ~50-token overlap, splitting on paragraph boundaries where possible, not mid-sentence. Store metadata with each chunk: source filename, page number, chunk index.
3. **Embed** each chunk via `llm.embed`.
4. **Store** chunks + embeddings + metadata in a persistent Chroma collection under `data/store/`.
5. Make it idempotent: re-running shouldn't duplicate chunks (hash the content or clear-and-rebuild).

**Decisions to understand (interviewers probe these):**
- Why chunk at all? (Context window limits + retrieval precision.)
- Why overlap? (So an answer split across a boundary isn't lost.)
- What breaks with chunks too big? (Noisy context, the model drowns.) Too small? (Lost context, retrieval misses.)

**Talk about it:** "Chunking strategy mattered more than the model choice. I split on paragraph boundaries with overlap and kept page metadata so answers can cite the source page."

---

## Phase 2 — Retrieval (vector + hybrid + rerank)

**Build (`retriever.py`):**
1. **Vector search:** embed the query, pull top-k (say 20) nearest chunks from Chroma.
2. **Keyword search:** BM25 over the same chunks via `rank_bm25`. Pull its top-k.
3. **Hybrid:** merge the two result sets (simple approach: reciprocal rank fusion, or weighted score). Dedup.
4. **Rerank:** run the merged candidates through the cross-encoder, keep the top 5. This is the step that most improves answer quality.
5. Return the final chunks with their metadata.

**Decisions to understand:**
- Why hybrid beats pure vector? (Vectors miss exact terms like error codes, IDs, acronyms; BM25 catches them.)
- What a reranker does? (A cross-encoder scores query+chunk *together*, far more accurate than the bi-encoder used for first-pass retrieval, but too slow to run on everything, so you only rerank the shortlist.)

**Talk about it:** "Pure vector search missed exact identifiers, so I added BM25 and fused the results, then reranked the shortlist with a cross-encoder. That two-stage retrieve-then-rerank pattern is the part that actually moved answer quality."

---

## Phase 3 — Generation with citations

**Build (`generate.py`):**
1. Take a question + the retrieved chunks.
2. Build a prompt: a system instruction ("answer only from the context, cite the source page, say you don't know if it's not there"), then the chunks with their source/page labels, then the question.
3. Call `llm.complete`, return the answer plus the list of sources used.
4. Handle the "context doesn't contain the answer" case explicitly so the model doesn't hallucinate.

**Talk about it:** "I constrained generation to the retrieved context and made 'I don't know' a valid answer, which is the main lever against hallucination in RAG."

---

## Phase 4 — Agentic layer

This is what lifts it above a plain RAG pipeline and hits their "agentic workflows" requirement.

**Build (`agent.py`):**
1. Define tools the model can call, as function schemas:
   - `search_docs(query)` → runs your retriever.
   - At least one more, e.g. `calculator(expression)` or `list_documents()`, so the agent has a real choice to make.
2. Build a loop: send the user question + tool definitions to the LLM. If it returns a tool call, execute it, feed the result back, and loop. If it returns a final answer, stop.
3. Cap the loop (e.g. max 5 iterations) so it can't run away. Handle tool errors gracefully.
4. Build this loop **by hand** first (raw tool-calling), so you understand the mechanics. Then you can *mention* LangGraph / LlamaIndex as the framework version.

**Decisions to understand:**
- Difference between RAG (always retrieve) and an agent (decides whether/what to retrieve).
- The real failure modes: infinite loops, runaway token cost, brittle tool error handling. You guard against all three.

**Talk about it:** "I wrote the tool-calling loop by hand instead of reaching for a framework, so I understand exactly where agents fail: loops, cost, and tool errors. I capped iterations and handled tool failures explicitly."

---

## Phase 5 — FastAPI service

Now it's a service, not a script. Hits "deploy and maintain in production" + the FastAPI nice-to-have.

**Build (`api.py`):**
- `POST /ask` → body `{question}`, returns `{answer, sources, latency_ms}`.
- `POST /ingest` → triggers ingestion (or accepts an uploaded PDF).
- `GET /health` → simple readiness check.
- Add request validation (Pydantic models), error handling (return clean 4xx/5xx, never leak stack traces), and timing on each request.
- Bonus: stream the answer token by token.

**Talk about it:** "I exposed it as a FastAPI service with validation, health checks, and per-request timing, so it behaves like something you'd actually deploy, not a notebook."

---

## Phase 6 — Evaluation harness

This is the single most senior-signalling part, and the part most people skip. Spend real time here.

**Build:**
1. `evals/golden_set.json`: 10–15 question/reference-answer pairs based on your PDFs. Include a few "this isn't in the docs" questions to test that it correctly says "I don't know."
2. `evals/run_eval.py`:
   - For each question, run the full pipeline.
   - Score the answer with **LLM-as-judge**: send the question, the generated answer, and the reference to the LLM and ask it to rate correctness/faithfulness on a 1–5 scale with a reason. Return structured JSON.
   - Also compute a simple **retrieval hit rate**: did the right chunk make it into the context?
   - Print a summary table and an average score.
3. Make it rerunnable so you can change a prompt or chunk size and *see the score move*. That's a regression test for AI.

**Decisions to understand:**
- Why you can't improve what you don't measure.
- Why LLM-as-judge (cheap, scalable, decent) and its weakness (the judge can be wrong; you spot-check it).
- Separating *retrieval* quality from *generation* quality so you know which half to fix.

**Talk about it:** "I built a golden set and an LLM-as-judge eval so every change to chunking or prompts gives me a number, not a vibe. I split retrieval hit-rate from answer quality so I know which half to fix." (If you say one thing on the call, say this.)

---

## Phase 7 — Observability

**Build (`logging_conf.py` + wire it in):**
- Structured (JSON) logs around every LLM and retrieval call: latency, token counts, which chunks were retrieved, tool calls made.
- A simple per-request trace id so you can follow one question end to end.
- Optional: plug in Langfuse for a proper LLM trace UI (free tier).

**Talk about it:** "I traced every LLM call with latency and token counts so I could see cost and find slow steps. In production that's where you catch a retrieval step quietly costing you."

---

## Phase 8 — Modern dev practices (do this throughout, not last)

- Type hints on all functions.
- `tests/`: at least test the retriever (given a known doc, the right chunk comes back) and ingestion (chunk counts, metadata present). Run with pytest.
- A `Dockerfile` so the service runs in a container.
- A clean `README.md`: what it is, how to run it, the architecture diagram, and a short "design decisions" section. The README is what a hiring engineer actually reads.

**Talk about it:** "Typed, tested, containerised, with a README that explains the design decisions, not just the setup commands."

---

## Phase 9 — Stretch (optional): tiny frontend

Only if you have time and want the React/TypeScript nice-to-have on the board.

- A single-page React + TypeScript app: a text box, a submit button, and a results panel showing the answer with its cited sources.
- Calls your `/ask` endpoint.

Keep it minimal. One page. The point is to truthfully say you wired a TS frontend to the service.

---

## Suggested build order for the call

If you're short on time before the call, this order gives the most talking points per hour:

1. Phases 0–3 (working RAG end to end) — the foundation.
2. Phase 6 (evals) — your strongest differentiator, do this before the agent if time is tight.
3. Phase 5 (FastAPI) — makes it a "service."
4. Phase 4 (agent) — the agentic story.
5. Phases 7–9 — polish if time allows.

Even just 0–3 plus 6 is enough to hold a serious conversation.

---

## How to use this with Claude Code

- Open Claude Code in an empty folder.
- Paste this whole file in and say: "Build Phase 0 and Phase 1 from this spec. Stop after Phase 1 and explain what you built and why before continuing."
- Run it, read the code, ask Claude Code to explain anything you don't follow. Drop 2–3 technical PDFs into `data/pdfs/` and ingest them.
- Move through phases one at a time. Don't let it build everything at once, you won't learn it and you won't be able to talk about it.
- After each phase, close the laptop and try to say the "Talk about it" line from memory. If you can, you're ready for that part of the call.
