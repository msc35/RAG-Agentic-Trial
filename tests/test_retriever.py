"""Tests for Phase 2 retrieval.

Unit tests (fast, no network):   test_rrf_*
Integration tests (need store):  test_integration_*

Integration tests are skipped automatically if data/store/ is empty.
Run the full suite:  pytest -q
Run only units:      pytest -q -m "not integration"
"""

import pytest
from src.retriever import Retriever


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def retriever():
    """Shared Retriever instance for integration tests.

    Marked scope="module" so we pay the BM25-index-build cost once per
    test file, not once per test function.
    """
    r = Retriever()
    if len(r._corpus) == 0:
        pytest.skip("No chunks in the vector store — run `make ingest` first.")
    return r


# --------------------------------------------------------------------------- #
# Unit tests — pure RRF logic, no network or model calls
# --------------------------------------------------------------------------- #

def test_rrf_merge_deduplication():
    """A chunk that appears in both lists should appear only once after merge."""
    vector_hits = [
        {"id": "a", "text": "alpha", "source": "x.pdf", "page": 1, "chunk_index": 0},
        {"id": "b", "text": "beta",  "source": "x.pdf", "page": 1, "chunk_index": 1},
    ]
    bm25_hits = [
        {"id": "b", "text": "beta",  "source": "x.pdf", "page": 1, "chunk_index": 1},
        {"id": "c", "text": "gamma", "source": "x.pdf", "page": 2, "chunk_index": 0},
    ]
    merged = Retriever._rrf_merge(vector_hits, bm25_hits)
    ids = [h["id"] for h in merged]
    assert len(ids) == len(set(ids)), "Duplicate ids after merge"
    assert set(ids) == {"a", "b", "c"}


def test_rrf_merge_cross_list_bonus():
    """A chunk in both lists should outscore a chunk only in one list."""
    shared    = {"id": "shared",  "text": "x", "source": "f.pdf", "page": 1, "chunk_index": 0}
    only_v    = {"id": "only_v",  "text": "y", "source": "f.pdf", "page": 2, "chunk_index": 0}
    only_b    = {"id": "only_b",  "text": "z", "source": "f.pdf", "page": 3, "chunk_index": 0}

    merged = Retriever._rrf_merge([shared, only_v], [shared, only_b])
    assert merged[0]["id"] == "shared", "Shared top-1 chunk should win"


def test_rrf_merge_preserves_order_single_list():
    """With only one list, output order must match that list's rank."""
    hits = [
        {"id": str(i), "text": "t", "source": "f.pdf", "page": i, "chunk_index": 0}
        for i in range(5)
    ]
    merged = Retriever._rrf_merge(hits, [])
    assert [h["id"] for h in merged] == [str(i) for i in range(5)]


def test_rrf_merge_empty_inputs():
    assert Retriever._rrf_merge([], []) == []


def test_rrf_merge_one_empty_list():
    hits = [{"id": "x", "text": "t", "source": "f.pdf", "page": 1, "chunk_index": 0}]
    assert len(Retriever._rrf_merge(hits, [])) == 1
    assert len(Retriever._rrf_merge([], hits)) == 1


# --------------------------------------------------------------------------- #
# Integration tests — require the live vector store
# --------------------------------------------------------------------------- #

@pytest.mark.integration
def test_integration_corpus_loaded(retriever):
    """The corpus should have a non-trivial number of chunks."""
    assert len(retriever._corpus) > 100, "Expected many chunks from ingested PDFs"


@pytest.mark.integration
def test_integration_correct_source_retrieved(retriever):
    """A query about MentorNet should surface the MentorNet paper."""
    results = retriever.retrieve("MentorNet curriculum learning noisy labels", top_k=5)
    assert results, "Expected at least one result"
    sources = [r.source for r in results]
    assert any("MentorNet" in s for s in sources), (
        f"MentorNet paper not in top-5. Got: {sources}"
    )


@pytest.mark.integration
def test_integration_sgd_paper_retrieved(retriever):
    """A query about SGD random shuffling should surface the right paper."""
    results = retriever.retrieve("random shuffling SGD convergence rate", top_k=5)
    sources = [r.source for r in results]
    assert any("SGD" in s or "sgd" in s.lower() or "Shuffling" in s for s in sources), (
        f"Expected an SGD paper in results, got: {sources}"
    )


@pytest.mark.integration
def test_integration_metadata_present(retriever):
    """Every returned chunk must carry source, page, chunk_index, and a score."""
    results = retriever.retrieve("stochastic gradient descent", top_k=3)
    for chunk in results:
        assert chunk.source, "Missing source"
        assert chunk.page >= 1, "Page number should be ≥ 1"
        assert chunk.chunk_index >= 0
        assert isinstance(chunk.score, float)


@pytest.mark.integration
def test_integration_scores_descending(retriever):
    """Cross-encoder scores should be in descending order."""
    results = retriever.retrieve("robust optimization corrupted gradients", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "Scores not sorted descending"


@pytest.mark.integration
def test_integration_top_k_respected(retriever):
    """retrieve(top_k=N) should return at most N chunks."""
    for k in (1, 3, 5):
        results = retriever.retrieve("SGD", top_k=k)
        assert len(results) <= k, f"Got {len(results)} results for top_k={k}"
