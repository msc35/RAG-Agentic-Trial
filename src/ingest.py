"""Phase 1 — Ingestion pipeline: load -> chunk -> embed -> store.

Run as a module:

    python -m src.ingest            # add/update chunks (idempotent)
    python -m src.ingest --rebuild  # wipe the collection and rebuild

Pipeline:
  1. Load every PDF in data/pdfs/, extract text per page (keep page numbers).
  2. Chunk each page's text to ~500 tokens with ~50-token overlap, splitting on
     paragraph boundaries where possible so we don't cut mid-sentence.
  3. Embed each chunk via llm.embed.
  4. Upsert chunks + embeddings + metadata into a persistent Chroma collection.

Idempotency: each chunk gets a deterministic id (a content hash). Re-running
upserts the same ids instead of duplicating. --rebuild deletes the collection
first so removed/edited PDFs don't leave stale chunks behind.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from .config import settings
from .llm import embed

# chromadb and pypdf are imported lazily inside the functions that need them, so
# the pure chunking logic stays importable (and unit-testable) without the heavy
# native/ML dependencies installed.

# One tokenizer instance, reused. cl100k_base matches the OpenAI embed/LLM
# models we use, so our token counts line up with what they actually see.
_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """A single retrievable unit of text plus where it came from."""

    id: str
    text: str
    source: str       # PDF filename
    page: int         # 1-based page number, for citations later
    chunk_index: int  # position of this chunk within its page


# --------------------------------------------------------------------------- #
# Step 1: load PDFs, text per page
# --------------------------------------------------------------------------- #
def load_pages(pdf_dir: Path) -> list[tuple[str, int, str]]:
    """Return (filename, page_number, text) for every non-empty page."""
    from pypdf import PdfReader  # lazy: only needed when actually ingesting

    pages: list[tuple[str, int, str]] = []
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {pdf_dir}. Drop some .pdf files there first.")
        return pages

    for path in pdf_paths:
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((path.name, page_number, text))
        print(f"Loaded {path.name}: {len(reader.pages)} pages")
    return pages


# --------------------------------------------------------------------------- #
# Step 2: chunking
# --------------------------------------------------------------------------- #
def _n_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _split_long_paragraph(paragraph: str, max_tokens: int) -> list[str]:
    """Hard-split a single paragraph that is itself larger than max_tokens.

    Falls back to a pure token window. Only used for the rare oversized
    paragraph (e.g. a giant table dumped as one block).
    """
    tokens = _ENCODER.encode(paragraph)
    pieces: list[str] = []
    for start in range(0, len(tokens), max_tokens):
        pieces.append(_ENCODER.decode(tokens[start : start + max_tokens]))
    return pieces


def chunk_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Pack paragraphs into ~max_tokens chunks with token overlap between them.

    Strategy: greedily accumulate whole paragraphs until adding the next one
    would exceed max_tokens, then emit a chunk. Start the next chunk with a
    token-overlap tail of the previous one so an answer spanning the boundary
    isn't lost.
    """
    # Split on blank lines (paragraph boundaries); fall back to the whole text.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0

    for para in paragraphs:
        para_tokens = _n_tokens(para)

        # A paragraph bigger than the budget gets hard-split on its own.
        if para_tokens > max_tokens:
            flush()
            chunks.extend(_split_long_paragraph(para, max_tokens))
            continue

        if current_tokens + para_tokens > max_tokens:
            flush()

        current.append(para)
        current_tokens += para_tokens

    flush()

    if overlap_tokens <= 0 or len(chunks) <= 1:
        return chunks

    # Add an overlap tail from each chunk to the start of the next one.
    overlapped: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tokens = _ENCODER.encode(chunks[i - 1])
        tail = _ENCODER.decode(prev_tokens[-overlap_tokens:])
        overlapped.append(f"{tail}\n\n{chunks[i]}")
    return overlapped


def _chunk_id(source: str, page: int, text: str) -> str:
    """Deterministic id from content + location → makes re-ingest idempotent."""
    digest = hashlib.sha256(f"{source}|{page}|{text}".encode()).hexdigest()
    return digest[:16]


def build_chunks(pages: list[tuple[str, int, str]]) -> list[Chunk]:
    """Turn loaded pages into a flat list of Chunk objects."""
    chunks: list[Chunk] = []
    for source, page, text in pages:
        pieces = chunk_text(
            text,
            max_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        for idx, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=_chunk_id(source, page, piece),
                    text=piece,
                    source=source,
                    page=page,
                    chunk_index=idx,
                )
            )
    return chunks


# --------------------------------------------------------------------------- #
# Steps 3 & 4: embed + store
# --------------------------------------------------------------------------- #
def get_collection(rebuild: bool = False):
    """Open (or create) the persistent Chroma collection."""
    import chromadb  # lazy: heavy native dep, only needed when storing

    settings.store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.store_dir))

    if rebuild:
        try:
            client.delete_collection(settings.collection_name)
            print(f"Rebuild: dropped existing collection '{settings.collection_name}'.")
        except Exception:
            pass  # didn't exist yet — fine

    # We pass our own embeddings, so no embedding_function is configured here.
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def ingest(rebuild: bool = False) -> int:
    """Run the full pipeline. Returns the number of chunks written."""
    pages = load_pages(settings.pdf_dir)
    if not pages:
        return 0

    chunks = build_chunks(pages)

    # Dedup ids within this run (identical text on the same page → one chunk).
    unique: dict[str, Chunk] = {c.id: c for c in chunks}
    chunks = list(unique.values())
    print(f"Built {len(chunks)} chunks from {len(pages)} pages.")

    collection = get_collection(rebuild=rebuild)

    print("Embedding chunks...")
    vectors = embed([c.text for c in chunks])

    # Chroma metadata can't hold the id field; store the rest as metadata.
    metadatas = [
        {"source": c.source, "page": c.page, "chunk_index": c.chunk_index}
        for c in chunks
    ]

    collection.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=metadatas,
    )

    total = collection.count()
    print(f"Done. Collection '{settings.collection_name}' now holds {total} chunks.")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the vector store.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop the existing collection before ingesting (clears stale chunks).",
    )
    args = parser.parse_args()
    ingest(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
