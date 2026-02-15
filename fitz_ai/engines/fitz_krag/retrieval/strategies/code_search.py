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
    """Hybrid keyword + BM25 + semantic search on symbol_index."""

    def __init__(
        self,
        symbol_store: "SymbolStore",
        embedder: "EmbeddingProvider",
        config: "FitzKragConfig",
    ):
        self._symbol_store = symbol_store
        self._embedder = embedder
        self._config = config
        self._hyde_generator: Any = None  # Set by engine if HyDE enabled
        self._raw_store: Any = None  # Set by engine for freshness boosting

    def retrieve(
        self,
        query: str,
        limit: int,
        detection: Any = None,
        *,
        query_vector: list[float] | None = None,
        hyde_vectors: list[list[float]] | None = None,
    ) -> list[Address]:
        """
        Retrieve code symbol addresses matching the query.

        1. Keyword search: query words against symbol names
        2. BM25 full-text search (when content_tsv exists)
        3. Semantic search: embed query, search summary_vector
        4. HyDE search: embed hypothetical docs, search (when enabled)
        5. Hybrid merge with configurable weights

        Args:
            query_vector: Pre-computed query embedding (skips internal embed call).
            hyde_vectors: Pre-computed HyDE hypothesis embeddings (skips HyDE generate + embed).
        """
        fetch_limit = limit * 2

        # 1. Keyword search
        keyword_results = self._symbol_store.search_by_name(query, limit=fetch_limit)

        # 2. BM25 search
        bm25_results: list[dict[str, Any]] = []
        try:
            bm25_results = self._symbol_store.search_bm25(query, limit=fetch_limit)
        except Exception as e:
            logger.debug(f"BM25 search not available: {e}")

        # 3. Semantic search
        try:
            if query_vector is None:
                query_vector = self._embedder.embed(query)
            semantic_results = self._symbol_store.search_by_vector(query_vector, limit=fetch_limit)
        except Exception as e:
            logger.warning(f"Semantic search failed, using keyword only: {e}")
            semantic_results = []

        # 4. HyDE search
        # hyde_vectors=None → not pre-computed, strategy may generate its own
        # hyde_vectors=[] → router intentionally skipped HyDE, don't generate
        # hyde_vectors=[...] → use pre-computed vectors
        skip_hyde = hyde_vectors is not None and len(hyde_vectors) == 0
        if not skip_hyde and (self._hyde_generator or hyde_vectors):
            hyde_results = self._run_hyde(query, fetch_limit, hyde_vectors=hyde_vectors)
            semantic_results = self._merge_hyde(semantic_results, hyde_results)

        # 5. Hybrid merge
        merged = self._merge_results(keyword_results, semantic_results, bm25_results)

        # 6. Keyword enrichment boost (from stored keywords)
        merged = self._apply_keyword_enrichment_boost(query, merged)

        # 7. Freshness boost (when detection signals boost_recency)
        if detection and getattr(detection, "boost_recency", False) and self._raw_store:
            merged = self._apply_recency_boost(merged)

        # 8. Convert to Address objects
        return [self._to_address(r) for r in merged[:limit]]

    def _run_hyde(
        self,
        query: str,
        limit: int,
        *,
        hyde_vectors: list[list[float]] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate hypothetical docs via HyDE and search with their embeddings.

        Args:
            hyde_vectors: Pre-computed hypothesis embeddings. When provided,
                          skips HyDE generation and embedding entirely.
        """
        try:
            if hyde_vectors:
                all_results: list[dict[str, Any]] = []
                for vec in hyde_vectors:
                    results = self._symbol_store.search_by_vector(vec, limit=limit)
                    all_results.extend(results)
                return all_results

            hypotheses = self._hyde_generator.generate(query)
            all_results: list[dict[str, Any]] = []
            for hyp in hypotheses:
                hyp_vector = self._embedder.embed(hyp)
                results = self._symbol_store.search_by_vector(hyp_vector, limit=limit)
                all_results.extend(results)
            return all_results
        except Exception as e:
            logger.warning(f"HyDE search failed: {e}")
            return []

    def _merge_hyde(
        self,
        semantic_results: list[dict[str, Any]],
        hyde_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge HyDE results into semantic results with lower weight."""
        # Discount HyDE scores by 0.5
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
        keyword_results: list[dict[str, Any]],
        semantic_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Merge keyword, BM25, and semantic results with weighted scoring."""
        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        kw = self._config.keyword_weight
        sw = self._config.semantic_weight
        bw = self._config.code_bm25_weight

        # Normalize weights when BM25 results present
        if bm25_results:
            total = kw + sw + bw
            kw, sw, bw = kw / total, sw / total, bw / total

        # Score keyword results by rank position
        for rank, r in enumerate(keyword_results):
            sid = r["id"]
            rank_score = 1.0 / (rank + 1)
            scores[sid] = scores.get(sid, 0) + kw * rank_score
            by_id[sid] = r

        # Score BM25 results
        if bm25_results:
            for rank, r in enumerate(bm25_results):
                sid = r["id"]
                bm25_score = r.get("bm25_score", 1.0 / (rank + 1))
                scores[sid] = scores.get(sid, 0) + bw * bm25_score
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

    def _apply_recency_boost(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Boost results from recently updated files."""
        if not results:
            return results
        file_ids = list({r["raw_file_id"] for r in results})
        try:
            timestamps = self._raw_store.get_updated_timestamps(file_ids)
            if not timestamps:
                return results
            sorted_files = sorted(
                timestamps, key=lambda fid: timestamps[fid] or "", reverse=True
            )
            top_quarter = set(sorted_files[: max(1, len(sorted_files) // 4)])
            top_half = set(sorted_files[: max(1, len(sorted_files) // 2)])
            for r in results:
                fid = r["raw_file_id"]
                if fid in top_quarter:
                    r["combined_score"] = r.get("combined_score", 0) + 0.1
                elif fid in top_half:
                    r["combined_score"] = r.get("combined_score", 0) + 0.05
            results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        except Exception as e:
            logger.debug(f"Recency boost skipped: {e}")
        return results

    def _apply_keyword_enrichment_boost(
        self, query: str, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Boost results that have matching enriched keywords."""
        query_terms = [w.lower().strip("?.,!;:()") for w in query.split() if len(w) >= 3]
        if not query_terms:
            return results
        try:
            keyword_hits = self._symbol_store.search_by_keywords(query_terms, limit=50)
            keyword_ids = {r["id"] for r in keyword_hits}
            if keyword_ids:
                for r in results:
                    if r["id"] in keyword_ids:
                        r["combined_score"] = r.get("combined_score", 0) + 0.1
                results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        except Exception as e:
            logger.debug(f"Keyword enrichment boost skipped: {e}")
        return results

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
