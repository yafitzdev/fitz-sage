# fitz_ai/engines/fitz_krag/retrieval/strategies/code_search.py
"""
Hybrid keyword + semantic search on the symbol index.

Merges keyword matches (symbol name ILIKE) with semantic matches
(summary_vector cosine similarity) using configurable weights.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fitz_ai.engines.fitz_krag.types import Address, AddressKind

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_ai.engines.fitz_krag.ingestion.symbol_store import SymbolStore
    from fitz_ai.llm.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class CodeSearchStrategy:
    """Hybrid keyword + semantic search on symbol_index."""

    def __init__(
        self,
        symbol_store: "SymbolStore",
        embedder: "EmbeddingProvider",
        config: "FitzKragConfig",
    ):
        self._symbol_store = symbol_store
        self._embedder = embedder
        self._config = config

    def retrieve(self, query: str, limit: int) -> list[Address]:
        """
        Retrieve code symbol addresses matching the query.

        1. Keyword search: query words against symbol names
        2. Semantic search: embed query, search summary_vector
        3. Hybrid merge with configurable weights
        """
        fetch_limit = limit * 2

        # 1. Keyword search
        keyword_results = self._symbol_store.search_by_name(query, limit=fetch_limit)

        # 2. Semantic search
        try:
            query_vector = self._embedder.embed(query)
            semantic_results = self._symbol_store.search_by_vector(query_vector, limit=fetch_limit)
        except Exception as e:
            logger.warning(f"Semantic search failed, using keyword only: {e}")
            semantic_results = []

        # 3. Hybrid merge
        merged = self._merge_results(keyword_results, semantic_results)

        # 4. Convert to Address objects
        return [self._to_address(r) for r in merged[:limit]]

    def _merge_results(
        self,
        keyword_results: list[dict[str, Any]],
        semantic_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge keyword and semantic results with weighted scoring."""
        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        kw = self._config.keyword_weight
        sw = self._config.semantic_weight

        # Score keyword results by rank position
        for rank, r in enumerate(keyword_results):
            sid = r["id"]
            rank_score = 1.0 / (rank + 1)
            scores[sid] = scores.get(sid, 0) + kw * rank_score
            by_id[sid] = r

        # Score semantic results by their cosine score
        for rank, r in enumerate(semantic_results):
            sid = r["id"]
            cosine_score = r.get("score", 1.0 / (rank + 1))
            scores[sid] = scores.get(sid, 0) + sw * cosine_score
            by_id[sid] = r

        # Sort by combined score
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        result = []
        for sid in sorted_ids:
            entry = by_id[sid].copy()
            entry["combined_score"] = scores[sid]
            result.append(entry)
        return result

    def _to_address(self, r: dict[str, Any]) -> Address:
        """Convert a symbol store row to an Address."""
        return Address(
            kind=AddressKind.SYMBOL,
            source_id=r["raw_file_id"],
            location=r["qualified_name"],
            summary=r.get("summary") or f"{r['kind']} {r['name']}",
            score=r.get("combined_score", 0.0),
            metadata={
                "symbol_id": r["id"],
                "name": r["name"],
                "qualified_name": r["qualified_name"],
                "kind": r["kind"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "signature": r.get("signature"),
            },
        )
