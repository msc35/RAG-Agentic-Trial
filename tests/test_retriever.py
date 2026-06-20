"""Tests for Phase 2 retrieval — pure logic only, no network/model calls."""

import pytest
from src.retriever import Retriever


def test_rrf_merge_deduplication():
    """A chunk that appears in both lists should appear only once after merge."""
    vector_hits = [
        {"id": "a", "text": "alpha", "source": "x.pdf", "page": 1, "chunk_index": 0},
        {"id": "b", "text": "beta", "source": "x.pdf", "page": 1, "chunk_index": 1},
    ]
    bm25_hits = [
        {"id": "b", "text": "beta", "source": "x.pdf", "page": 1, "chunk_index": 1},
        {"id": "c", "text": "gamma", "source": "x.pdf", "page": 2, "chunk_index": 0},
    ]
    merged = Retriever._rrf_merge(vector_hits, bm25_hits)
    ids = [h["id"] for h in merged]
    assert len(ids) == len(set(ids)), "Duplicate ids after merge"
    assert set(ids) == {"a", "b", "c"}


def test_rrf_merge_cross_list_bonus():
    """A chunk in both lists should outscore a chunk only in one list."""
    shared = {"id": "shared", "text": "x", "source": "f.pdf", "page": 1, "chunk_index": 0}
    only_vector = {"id": "only_v", "text": "y", "source": "f.pdf", "page": 2, "chunk_index": 0}
    only_bm25 = {"id": "only_b", "text": "z", "source": "f.pdf", "page": 3, "chunk_index": 0}

    vector_hits = [shared, only_vector]
    bm25_hits = [shared, only_bm25]
    merged = Retriever._rrf_merge(vector_hits, bm25_hits)

    # 'shared' is rank-1 in both lists; it must come first.
    assert merged[0]["id"] == "shared"


def test_rrf_merge_preserves_order_single_list():
    """With only one list contributing, output order matches that list's rank."""
    hits = [
        {"id": str(i), "text": "t", "source": "f.pdf", "page": i, "chunk_index": 0}
        for i in range(5)
    ]
    merged = Retriever._rrf_merge(hits, [])
    ids = [h["id"] for h in merged]
    assert ids == [str(i) for i in range(5)]
