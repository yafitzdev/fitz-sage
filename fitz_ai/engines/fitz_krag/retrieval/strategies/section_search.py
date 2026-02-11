# fitz_ai/engines/fitz_krag/retrieval/strategies/section_search.py
"""
Section search strategy — BM25 + semantic hybrid for technical documents.

BM25 is weighted higher than semantic (0.6 vs 0.4) because technical
documents are keyword-heavy and full-text search excels at exact term matching.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fitz_ai.engines.fitz_krag.types import Address, AddressKind

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_ai.engines.fitz_krag.ingestion.section_store import SectionStore
    from fitz_ai.llm.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class SectionSearchStrategy:
    """BM25-first retrieval with semantic fallback for technical documents."""

    def __init__(
        self,
        section_store: "SectionStore",
        embedder: "EmbeddingProvider",
        config: "FitzKragConfig",
    ):
        self._section_store = section_store
        self._embedder = embedder
        self._config = config
        self._hyde_generator: Any = None  # Set by engine if HyDE enabled

    def retrieve(self, query: str, limit: int) -> list[Address]:
        """
        Retrieve section addresses matching the query.

        1. BM25 full-text search (PostgreSQL ts_rank)
        2. Semantic search on section summaries
        3. HyDE search (when enabled)
        4. Hybrid merge — BM25 weighted higher
        """
        fetch_limit = limit * 2

        # 1. BM25 search
        bm25_results = self._section_store.search_bm25(query, limit=fetch_limit)

        # 2. Semantic search
        try:
            query_vector = self._embedder.embed(query)
            semantic_results = self._section_store.search_by_vector(query_vector, limit=fetch_limit)
        except Exception as e:
            logger.warning(f"Semantic section search failed, using BM25 only: {e}")
            semantic_results = []

        # 3. HyDE search
        if self._hyde_generator:
            hyde_results = self._run_hyde(query, fetch_limit)
            semantic_results = self._merge_hyde(semantic_results, hyde_results)

        # 4. Hybrid merge
        bm25_weight = self._config.section_bm25_weight
        semantic_weight = self._config.section_semantic_weight
        merged = self._merge_results(bm25_results, semantic_results, bm25_weight, semantic_weight)

        # 5. Convert to Address objects
        return [self._to_address(r) for r in merged[:limit]]

    def _run_hyde(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Generate hypothetical docs via HyDE and search with their embeddings."""
        try:
            hypotheses = self._hyde_generator.generate(query)
            all_results: list[dict[str, Any]] = []
            for hyp in hypotheses:
                hyp_vector = self._embedder.embed(hyp)
                results = self._section_store.search_by_vector(hyp_vector, limit=limit)
                all_results.extend(results)
            return all_results
        except Exception as e:
            logger.warning(f"HyDE section search failed: {e}")
            return []

    def _merge_hyde(
        self,
        semantic_results: list[dict[str, Any]],
        hyde_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge HyDE results into semantic results with lower weight."""
        for r in hyde_results:
            r["score"] = r.get("score", 0.0) * 0.5
        combined = list(semantic_results)
        seen_ids = {r["id"] for r in combined}
        for r in hyde_results:
            if r["id"] not in seen_ids:
                combined.append(r)
                seen_ids.add(r["id"])
        return combined

    def _merge_results(
        self,
        bm25_results: list[dict[str, Any]],
        semantic_results: list[dict[str, Any]],
        bm25_weight: float,
        semantic_weight: float,
    ) -> list[dict[str, Any]]:
        """Merge BM25 and semantic results with weighted scoring."""
        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}

        # Normalize BM25 scores by rank
        for rank, r in enumerate(bm25_results):
            sid = r["id"]
            rank_score = r.get("bm25_score", 1.0 / (rank + 1))
            scores[sid] = scores.get(sid, 0) + bm25_weight * rank_score
            by_id[sid] = r

        # Use cosine scores directly for semantic
        for rank, r in enumerate(semantic_results):
            sid = r["id"]
            cosine_score = r.get("score", 1.0 / (rank + 1))
            scores[sid] = scores.get(sid, 0) + semantic_weight * cosine_score
            by_id[sid] = r

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        result = []
        for sid in sorted_ids:
            entry = by_id[sid].copy()
            entry["combined_score"] = scores[sid]
            result.append(entry)
        return result

    def _to_address(self, section: dict[str, Any]) -> Address:
        """Convert a section store row to an Address."""
        return Address(
            kind=AddressKind.SECTION,
            source_id=section["raw_file_id"],
            location=section["title"],
            summary=section.get("summary") or section["title"],
            score=section.get("combined_score", 0.0),
            metadata={
                "section_id": section["id"],
                "level": section["level"],
                "page_start": section.get("page_start"),
                "page_end": section.get("page_end"),
                "parent_section_id": section.get("parent_section_id"),
            },
        )
