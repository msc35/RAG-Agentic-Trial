"""Phase 3 — Generation with citations.

Public API:
    result = generate(question, chunks)
    print(result.answer)
    print(result.sources)   # list of (filename, page) tuples

Pipeline:
  1. Build a prompt that includes every retrieved chunk, labelled with its
     source file and page number.
  2. Instruct the model to answer ONLY from the provided context, to cite the
     source page in its answer, and to say "I don't know" if the context
     doesn't contain the answer.
  3. Call llm.complete and parse the response.
  4. Return the answer text and a deduplicated list of cited sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .llm import complete
from .retriever import RetrievedChunk

_SYSTEM_PROMPT = """\
You are a research assistant that answers questions about technical documents.

Rules you must follow:
1. Answer ONLY using information present in the CONTEXT sections below.
2. For every claim you make, cite the source with (filename, p.N) immediately after it.
3. If the answer is not in the context, say exactly: "I don't have enough information in the provided documents to answer this question." Do NOT guess or use outside knowledge.
4. Be concise and precise. Prefer bullet points for multi-part answers.
"""


@dataclass
class GenerationResult:
    answer: str
    sources: list[tuple[str, int]] = field(default_factory=list)  # (filename, page)


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a labelled context block for the prompt."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[CONTEXT {i}] Source: {chunk.source}, Page {chunk.page}\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_sources(chunks: list[RetrievedChunk]) -> list[tuple[str, int]]:
    """Return unique (source, page) pairs, preserving rank order."""
    seen: set[tuple[str, int]] = set()
    sources: list[tuple[str, int]] = []
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            sources.append(key)
    return sources


def generate(
    question: str,
    chunks: list[RetrievedChunk],
    model: str | None = None,
) -> GenerationResult:
    """Generate an answer grounded in the retrieved chunks.

    Args:
        question: The user's question.
        chunks:   Retrieved chunks from the retrieval pipeline (Phase 2).
        model:    Optional model override (defaults to settings.llm_model).

    Returns:
        GenerationResult with the answer text and deduplicated source list.
    """
    if not chunks:
        return GenerationResult(
            answer="I don't have enough information in the provided documents to answer this question.",
            sources=[],
        )

    context_block = _build_context_block(chunks)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{context_block}\n\n"
                f"---\n\n"
                f"Question: {question}"
            ),
        },
    ]

    message = complete(messages, model=model)
    answer = (message.content or "").strip()

    # If the model returned an empty response, fall back to the "I don't know" path.
    if not answer:
        answer = "I don't have enough information in the provided documents to answer this question."

    sources = _extract_sources(chunks)
    return GenerationResult(answer=answer, sources=sources)
