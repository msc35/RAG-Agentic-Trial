"""Tests for Phase 1 ingestion — pure unit tests, no network or file I/O."""

import pytest
from src.ingest import (
    _chunk_id,
    _n_tokens,
    _split_long_paragraph,
    build_chunks,
    chunk_text,
)


# --------------------------------------------------------------------------- #
# chunk_text
# --------------------------------------------------------------------------- #

def _make_text(n_paras: int, tokens_per_para: int = 80) -> str:
    """Build synthetic multi-paragraph text with a predictable token count."""
    para = ("word " * tokens_per_para).strip()
    return "\n\n".join(f"Paragraph {i}. {para}" for i in range(n_paras))


def test_chunk_text_respects_max_tokens():
    text = _make_text(n_paras=20, tokens_per_para=60)
    chunks = chunk_text(text, max_tokens=500, overlap_tokens=0)
    for chunk in chunks:
        assert _n_tokens(chunk) <= 500, "A chunk exceeded max_tokens"


def test_chunk_text_produces_multiple_chunks():
    text = _make_text(n_paras=20, tokens_per_para=60)
    chunks = chunk_text(text, max_tokens=500, overlap_tokens=0)
    assert len(chunks) > 1, "Long text should produce more than one chunk"


def test_chunk_text_overlap_present():
    """The tail of chunk N should appear somewhere near the start of chunk N+1."""
    text = _make_text(n_paras=20, tokens_per_para=60)
    chunks = chunk_text(text, max_tokens=500, overlap_tokens=50)
    if len(chunks) < 2:
        pytest.skip("Need at least 2 chunks to test overlap")
    tail_words = chunks[0].split()[-5:]
    assert any(w in chunks[1] for w in tail_words), "Overlap tail not found in next chunk"


def test_chunk_text_no_content_lost():
    """Every paragraph should appear in at least one chunk."""
    paras = [f"UniqueMarker{i} sentence about topic." for i in range(10)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, max_tokens=200, overlap_tokens=20)
    joined = " ".join(chunks)
    for i, para in enumerate(paras):
        marker = f"UniqueMarker{i}"
        assert marker in joined, f"Paragraph {i} content missing from chunks"


def test_chunk_text_empty_input():
    assert chunk_text("", max_tokens=500, overlap_tokens=50) == []


def test_chunk_text_single_short_paragraph():
    text = "A short paragraph."
    chunks = chunk_text(text, max_tokens=500, overlap_tokens=50)
    assert len(chunks) == 1
    assert chunks[0] == text


# --------------------------------------------------------------------------- #
# _split_long_paragraph
# --------------------------------------------------------------------------- #

def test_split_long_paragraph_each_piece_within_budget():
    long_para = ("token " * 600).strip()
    pieces = _split_long_paragraph(long_para, max_tokens=200)
    for piece in pieces:
        assert _n_tokens(piece) <= 200


def test_split_long_paragraph_nothing_lost():
    long_para = ("alpha beta gamma " * 200).strip()
    pieces = _split_long_paragraph(long_para, max_tokens=100)
    joined_tokens = sum(_n_tokens(p) for p in pieces)
    original_tokens = _n_tokens(long_para)
    # Token counts should be close (slight rounding from decode/encode is ok).
    assert abs(joined_tokens - original_tokens) <= 5


# --------------------------------------------------------------------------- #
# build_chunks — metadata
# --------------------------------------------------------------------------- #

def test_build_chunks_metadata():
    pages = [
        ("paper.pdf", 3, "Introduction paragraph.\n\nSecond paragraph here."),
        ("other.pdf", 1, "Short content."),
    ]
    chunks = build_chunks(pages)

    sources = {c.source for c in chunks}
    assert "paper.pdf" in sources
    assert "other.pdf" in sources

    for c in chunks:
        assert c.page >= 1
        assert c.chunk_index >= 0
        assert len(c.id) == 16  # sha256 hex digest, truncated
        assert c.text  # no empty chunks


def test_build_chunks_page_numbers_preserved():
    pages = [("doc.pdf", 7, "Content on page seven.\n\nMore content.")]
    chunks = build_chunks(pages)
    assert all(c.page == 7 for c in chunks)


def test_build_chunks_chunk_index_sequential():
    long_text = _make_text(n_paras=20, tokens_per_para=60)
    pages = [("big.pdf", 1, long_text)]
    chunks = build_chunks(pages)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


# --------------------------------------------------------------------------- #
# _chunk_id — determinism & uniqueness
# --------------------------------------------------------------------------- #

def test_chunk_id_deterministic():
    id1 = _chunk_id("doc.pdf", 1, "some text")
    id2 = _chunk_id("doc.pdf", 1, "some text")
    assert id1 == id2


def test_chunk_id_differs_by_source():
    assert _chunk_id("a.pdf", 1, "text") != _chunk_id("b.pdf", 1, "text")


def test_chunk_id_differs_by_page():
    assert _chunk_id("a.pdf", 1, "text") != _chunk_id("a.pdf", 2, "text")


def test_chunk_id_differs_by_content():
    assert _chunk_id("a.pdf", 1, "text one") != _chunk_id("a.pdf", 1, "text two")


def test_build_chunks_idempotent_ids():
    """Running build_chunks twice on the same input must produce identical ids."""
    pages = [("doc.pdf", 1, "Paragraph one.\n\nParagraph two.\n\nParagraph three.")]
    run1 = [c.id for c in build_chunks(pages)]
    run2 = [c.id for c in build_chunks(pages)]
    assert run1 == run2
