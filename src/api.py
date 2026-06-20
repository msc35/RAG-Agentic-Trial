"""Phase 5 — FastAPI service.

Endpoints:
    POST /ask        Run the agent pipeline; returns answer + sources + timing.
                     Add ?stream=true for token-by-token streaming (SSE).
    POST /ingest     Re-ingest all PDFs in data/pdfs/, or upload a new PDF first.
    GET  /health     Readiness check — confirms the vector store is reachable.

Design notes:
    - The Retriever (BM25 index + Chroma connection) is created once at startup
      via FastAPI's lifespan hook and reused across requests. Building it per
      request would add ~1s latency per call.
    - All heavy synchronous work (retrieval, LLM calls) runs in a thread-pool
      executor so it doesn't block the async event loop.
    - Errors return clean JSON with a `detail` field; stack traces never reach
      the client.
    - Streaming uses Server-Sent Events (SSE) over the direct generate path.
      Streaming + agent tool-call loops is possible but significantly more
      complex; for the agent path we return a complete response.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent import Agent, AgentResult
from .generate import generate
from .ingest import ingest
from .llm import stream_complete
from .logging_conf import configure_logging, get_logger, new_trace_id, set_trace_id
from .retriever import Retriever

_log = get_logger("api")

# --------------------------------------------------------------------------- #
# App state — initialised once in lifespan, reused across all requests
# --------------------------------------------------------------------------- #

_retriever: Retriever | None = None
_agent: Agent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the BM25 index and open the Chroma connection at startup."""
    configure_logging()
    global _retriever, _agent
    _log.info("startup_begin")
    loop = asyncio.get_event_loop()
    # Run in executor: Retriever.__init__ is synchronous and does real work
    # (loads all corpus into RAM for BM25).
    _retriever = await loop.run_in_executor(None, Retriever)
    _agent = Agent(_retriever)
    _log.info("startup_done", chunks=len(_retriever._corpus))
    yield
    # Nothing to clean up — Chroma persists to disk automatically.


app = FastAPI(
    title="Agentic RAG Service",
    description="Answers questions about ingested technical PDFs.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server and any production frontend origin.
# In production, replace "*" with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """Generate a trace_id per request, log boundaries, return it as a header.

    The ID is stored on request.state so the _require_trace dependency can
    inject it into each handler's own contextvars context. We do NOT call
    set_trace_id() here because Starlette's call_next() runs handlers in a
    separate async task — any ContextVar set in this coroutine won't be
    visible there.
    """
    tid = new_trace_id()
    request.state.trace_id = tid
    t0 = time.perf_counter()
    _log.info("request_start", method=request.method, path=request.url.path, trace_id=tid)
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    _log.info(
        "request_end",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=latency_ms,
        trace_id=tid,
    )
    response.headers["X-Trace-Id"] = tid
    return response


async def _require_trace(request: Request) -> str:
    """FastAPI dependency: sets the trace_id in the handler's own context.

    run_in_executor copies THIS context to worker threads, so every log line
    inside retrieval/LLM/agent calls picks up the correct trace_id.
    """
    tid = getattr(request.state, "trace_id", None) or new_trace_id()
    set_trace_id(tid)
    return tid


def _get_agent() -> Agent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Service is still starting up.")
    return _agent


def _get_retriever() -> Retriever:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Service is still starting up.")
    return _retriever


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The question to answer.")


class SourceModel(BaseModel):
    filename: str
    page: int


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceModel]
    latency_ms: float
    iterations: int
    tool_calls_made: list[str]


class IngestResponse(BaseModel):
    chunks_written: int
    message: str


class HealthResponse(BaseModel):
    status: str
    chunks_in_store: int


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(_tid: str = Depends(_require_trace)):
    """Readiness check — confirms the vector store and BM25 index are loaded."""
    r = _get_retriever()
    return HealthResponse(status="ok", chunks_in_store=len(r._corpus))


@app.post("/ask", response_model=AskResponse, tags=["query"])
async def ask(
    request: AskRequest,
    stream: bool = Query(default=False, description="Stream tokens via SSE instead of returning a complete response."),
    _tid: str = Depends(_require_trace),
):
    """Answer a question using the agentic RAG pipeline.

    The agent decides how many retrieval steps to take (up to 5).
    Use `?stream=true` for token-by-token SSE streaming (bypasses the agent
    loop and uses the direct generate path).
    """
    tid = _tid  # captured before any branch so both paths can use it

    if stream:
        return await _ask_streaming(request.question, tid)

    agent = _get_agent()
    start = time.perf_counter()

    def _run_agent() -> AgentResult:
        set_trace_id(tid)
        return agent.run(request.question)

    try:
        loop = asyncio.get_event_loop()
        result: AgentResult = await loop.run_in_executor(None, _run_agent)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    latency_ms = (time.perf_counter() - start) * 1000
    return AskResponse(
        answer=result.answer,
        sources=[SourceModel(filename=src, page=pg) for src, pg in result.sources],
        latency_ms=round(latency_ms, 1),
        iterations=result.iterations,
        tool_calls_made=result.tool_calls_made,
    )


async def _ask_streaming(question: str, tid: str) -> StreamingResponse:
    """Stream the answer token by token via Server-Sent Events."""
    retriever = _get_retriever()

    async def event_stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()

        def _retrieve():
            set_trace_id(tid)
            return retriever.retrieve(question, top_k=5)

        chunks = await loop.run_in_executor(None, _retrieve)

        if not chunks:
            yield "data: I don't have enough information in the provided documents to answer this question.\n\n"
            yield "data: [DONE]\n\n"
            return

        from .generate import _build_context_block, _SYSTEM_PROMPT

        context_block = _build_context_block(chunks)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{context_block}\n\n---\n\nQuestion: {question}",
            },
        ]

        # stream_complete is a synchronous generator — iterate it in executor.
        gen = stream_complete(messages)
        while True:
            try:
                token = await loop.run_in_executor(None, next, gen)
                yield f"data: {token}\n\n"
            except StopIteration:
                break

        # Emit sources as a final SSE event.
        from .generate import _extract_sources
        import json
        sources = [{"filename": s, "page": p} for s, p in _extract_sources(chunks)]
        yield f"data: [SOURCES]{json.dumps(sources)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/ingest", response_model=IngestResponse, tags=["ops"])
async def ingest_endpoint(
    file: UploadFile | None = File(default=None, description="Optional PDF to upload before re-ingesting."),
    rebuild: bool = Query(default=False, description="Wipe the collection and rebuild from scratch."),
    _tid: str = Depends(_require_trace),
):
    """Re-ingest PDFs from data/pdfs/ into the vector store.

    Optionally upload a new PDF first (it will be saved to data/pdfs/ then included).
    Pass ?rebuild=true to clear stale chunks before re-ingesting.
    """
    from .config import settings

    if file is not None:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
        dest = settings.pdf_dir / file.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)

    try:
        loop = asyncio.get_event_loop()
        def _ingest():
            set_trace_id(_tid)
            return ingest(rebuild=rebuild)
        chunks_written = await loop.run_in_executor(None, _ingest)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest error: {exc}") from exc

    # Reload the retriever so the new chunks are searchable immediately.
    global _retriever, _agent
    def _reload():
        set_trace_id(_tid)
        return Retriever()
    _retriever = await loop.run_in_executor(None, _reload)
    _agent = Agent(_retriever)

    action = "Rebuilt" if rebuild else "Updated"
    return IngestResponse(
        chunks_written=chunks_written,
        message=f"{action} index. Store now has {len(_retriever._corpus)} chunks.",
    )
