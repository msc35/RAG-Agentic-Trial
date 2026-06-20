"""Phase 2 — Retrieval: vector search + BM25 hybrid + cross-encoder rerank.

Public API:
    retriever = Retriever()
    chunks = retriever.retrieve("what is SGD?", top_k=5)

Pipeline:
  1. Vector search   — embed the query, pull top-k_candidates from Chroma.
  2. BM25 search     — keyword search over the same corpus, pull top-k_candidates.
  3. Hybrid merge    — combine both result sets via Reciprocal Rank Fusion (RRF),
                       dedup by chunk id.
  4. Rerank          — score the merged shortlist with a cross-encoder, keep top_k.

Why each step:
  • Vector misses exact terms (error codes, acronyms, "SGD" vs "gradient descent").
    BM25 catches them. Hybrid gets both.
  • RRF is a simple, parameter-light way to merge ranked lists without needing
    score normalization (scores from two different models aren't comparable).
  • The cross-encoder reads query+chunk together, much more accurate than the
    bi-encoder used for first-pass retrieval, but too slow to run on all chunks.
    Running it only on the shortlist (top ~40) is the classic two-stage pattern.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from .config import settings
from .llm import embed
from .logging_conf import get_logger

_log = get_logger("retriever")


@dataclass
class RetrievedChunk:
    """A chunk that made it through the full retrieval pipeline."""

    id: str
    text: str
    source: str
    page: int
    chunk_index: int
    score: float  # cross-encoder score (higher = more relevant)


class Retriever:
    """Stateful retriever: loads BM25 corpus once, then answers many queries."""

    # Number of candidates to pull from each first-pass retriever before reranking.
    CANDIDATES = 20

    def __init__(self) -> None:
        self._collection = self._open_collection()
        # Load the full corpus from Chroma once so BM25 covers the same data.
        self._corpus: list[dict[str, Any]] = self._load_corpus()
        self._bm25: BM25Okapi = self._build_bm25()

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _open_collection():
        client = chromadb.PersistentClient(path=str(settings.store_dir))
        return client.get_collection(name=settings.collection_name)

    def _load_corpus(self) -> list[dict[str, Any]]:
        """Pull every document out of Chroma. Used to build the BM25 index."""
        total = self._collection.count()
        if total == 0:
            return []
        result = self._collection.get(
            include=["documents", "metadatas"],
            limit=total,
        )
        docs = []
        for i, doc_id in enumerate(result["ids"]):
            docs.append(
                {
                    "id": doc_id,
                    "text": result["documents"][i],
                    "source": result["metadatas"][i]["source"],
                    "page": result["metadatas"][i]["page"],
                    "chunk_index": result["metadatas"][i]["chunk_index"],
                }
            )
        return docs

    def _build_bm25(self) -> BM25Okapi:
        """Tokenise each document and build the BM25 index."""
        tokenized = [doc["text"].lower().split() for doc in self._corpus]
        return BM25Okapi(tokenized)

    @cached_property
    def _cross_encoder(self):
        """Load the cross-encoder model lazily (slow first call, then cached)."""
        import os
        # Prevent sentence-transformers from trying to import TensorFlow/Keras,
        # which crashes on this Anaconda env (Keras 3 vs tf-keras conflict).
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("USE_TF", "0")
        from sentence_transformers import CrossEncoder

        print("Loading cross-encoder model (first call only)...")
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ------------------------------------------------------------------ #
    # Step 1: vector search
    # ------------------------------------------------------------------ #

    def _vector_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Embed the query and retrieve the k nearest chunks from Chroma."""
        query_vector = embed([query])[0]
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i, doc_id in enumerate(result["ids"][0]):
            hits.append(
                {
                    "id": doc_id,
                    "text": result["documents"][0][i],
                    "source": result["metadatas"][0][i]["source"],
                    "page": result["metadatas"][0][i]["page"],
                    "chunk_index": result["metadatas"][0][i]["chunk_index"],
                    # distance is cosine distance; convert to a similarity rank
                    "vector_rank": i + 1,
                }
            )
        return hits

    # ------------------------------------------------------------------ #
    # Step 2: BM25 keyword search
    # ------------------------------------------------------------------ #

    def _bm25_search(self, query: str, k: int) -> list[dict[str, Any]]:
        """Return the top-k chunks by BM25 score."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        # argsort descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for rank, idx in enumerate(ranked_indices[:k]):
            doc = self._corpus[idx]
            hits.append({**doc, "bm25_rank": rank + 1})
        return hits

    # ------------------------------------------------------------------ #
    # Step 3: Reciprocal Rank Fusion (RRF)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rrf_merge(
        vector_hits: list[dict[str, Any]],
        bm25_hits: list[dict[str, Any]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """Merge two ranked lists with Reciprocal Rank Fusion.

        RRF score = sum of 1/(k + rank_i) across all lists. k=60 is the
        standard constant from the original RRF paper (Cormack et al. 2009).
        A chunk that appears in both lists at high rank scores highest.
        """
        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(vector_hits, start=1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (k + rank)
            by_id[hit["id"]] = hit

        for rank, hit in enumerate(bm25_hits, start=1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (k + rank)
            by_id.setdefault(hit["id"], hit)

        merged = sorted(by_id.values(), key=lambda h: scores[h["id"]], reverse=True)
        return merged

    # ------------------------------------------------------------------ #
    # Step 4: cross-encoder rerank
    # ------------------------------------------------------------------ #

    def _rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int
    ) -> list[RetrievedChunk]:
        """Score every candidate with the cross-encoder, return top_k."""
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._cross_encoder.predict(pairs)

        ranked = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [
            RetrievedChunk(
                id=c["id"],
                text=c["text"],
                source=c["source"],
                page=c["page"],
                chunk_index=c["chunk_index"],
                score=float(s),
            )
            for c, s in ranked
        ]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Run the full retrieve-then-rerank pipeline for a query.

        Args:
            query:  The user's question or search string.
            top_k:  Number of final chunks to return after reranking.

        Returns:
            Chunks sorted by cross-encoder relevance score, descending.
        """
        if self._collection.count() == 0:
            return []

        t0 = time.perf_counter()
        vector_hits = self._vector_search(query, k=self.CANDIDATES)
        bm25_hits = self._bm25_search(query, k=self.CANDIDATES)
        candidates = self._rrf_merge(vector_hits, bm25_hits)
        results = self._rerank(query, candidates, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        _log.info(
            "retrieval",
            query=query[:120],
            n_vector=len(vector_hits),
            n_bm25=len(bm25_hits),
            n_candidates=len(candidates),
            n_returned=len(results),
            top_sources=[f"{c.source}:p{c.page}" for c in results[:3]],
            top_scores=[round(c.score, 3) for c in results[:3]],
            latency_ms=round(latency_ms, 1),
        )
        return results
