# src/fitz_rag/retriever/rag_retriever.py
"""
Embedding-based retriever for fitz-rag.

Supports multiple Qdrant API styles:
- client.query_points(vector=...)               # modern API
- client.query_points(query_vector=...)         # some versions
- client.search(query_vector=...)               # legacy API

Optionally supports a second-stage cross-encoder reranker:
- e.g. CohereRerankClient
"""

from __future__ import annotations

from typing import List, Any, Dict, Optional

from qdrant_client import QdrantClient

from fitz_rag.core import RetrievedChunk
from fitz_rag.retriever.base import BaseRetriever
from fitz_rag.llm.rerank_client import RerankClient

from fitz_rag.config import get_config

_cfg = get_config()


class RAGRetriever(BaseRetriever):
    """
    Standard embedding-based retriever using an embedding client
    and a Qdrant vector store. Works across Qdrant versions.

    If a reranker is provided, retrieval works in two stages:

        1) dense search in Qdrant (top_k candidates)
        2) rerank those candidates using the original query
           -> keep top rerank_k (or all if None)
    """

    def __init__(
        self,
        client: QdrantClient,
        embedder: Any,
        collection: Optional[str] = None,
        top_k: Optional[int] = None,
        reranker: Optional[RerankClient] = None,
        rerank_k: Optional[int] = None,
    ) -> None:
        # load config fallbacks
        retr_cfg = _cfg.get("retriever", {})
        qdrant_cfg = _cfg.get("qdrant", {})

        self.client = client
        self.embedder = embedder
        self.collection = collection or qdrant_cfg.get("collection", "fitz_default")
        self.top_k = top_k or retr_cfg.get("top_k", 10)

        # rerank settings
        self.reranker = reranker
        # if None in config and None in argument, reranking is disabled
        self.rerank_k = rerank_k or retr_cfg.get("rerank_k", None)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        if not query or not query.strip():
            return []

        # 1) Compute embedding
        vector = self.embedder.embed(query)

        # 2) Raw dense retrieval from Qdrant (unordered)
        results = self._search_qdrant(vector)
        if not results:
            return []

        # 3) Optional rerank
        if self.reranker is not None:
            texts = [ (p.payload or {}).get("text", "") for p in results ]
            # If rerank_k is None, keep all
            top_n = self.rerank_k or len(texts)
            # Guard against weird configs
            top_n = max(1, min(top_n, len(texts)))

            order = self.reranker.rerank(query, texts, top_n=top_n)
            # Reorder results by rerank indices
            results = [results[i] for i in order]

        # 4) Convert to RetrievedChunk
        return self._wrap_results(results)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _search_qdrant(self, vector) -> List[Any]:
        """
        Perform the actual Qdrant search, being defensive against different
        client versions / APIs. Returns a list of point-like objects.
        """
        # Try modern "query_points" API first
        if hasattr(self.client, "query_points"):
            # New style: vector=...
            try:
                raw = self.client.query_points(
                    collection_name=self.collection,
                    vector=vector,
                    limit=self.top_k,
                    with_payload=True,
                    with_vectors=False,
                )
                return list(raw.points)
            except TypeError:
                pass

            # Older variant: query_vector=...
            try:
                raw = self.client.query_points(
                    collection_name=self.collection,
                    query_vector=vector,
                    limit=self.top_k,
                    with_payload=True,
                    with_vectors=False,
                )
                return list(raw.points)
            except TypeError:
                pass

        # Legacy .search() API
        if hasattr(self.client, "search"):
            try:
                results = self.client.search(
                    collection_name=self.collection,
                    query_vector=vector,
                    limit=self.top_k,
                    with_payload=True,
                    with_vectors=False,
                )
                return list(results)
            except Exception:
                pass

        # If no path succeeded, error out
        raise RuntimeError(
            "RAGRetriever could not perform search. "
            "Unsupported Qdrant API version."
        )

    def _wrap_results(self, results) -> List[RetrievedChunk]:
        """
        Convert Qdrant results → RetrievedChunk objects.
        """
        chunks: List[RetrievedChunk] = []

        for point in results:
            payload: Dict = point.payload or {}

            chunks.append(
                RetrievedChunk(
                    collection=self.collection,
                    score=float(point.score),
                    text=payload.get("text", ""),
                    metadata=payload,
                    chunk_id=point.id,
                )
            )

        return chunks
