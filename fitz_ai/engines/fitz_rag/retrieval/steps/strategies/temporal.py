# fitz_ai/engines/fitz_rag/retrieval/steps/strategies/temporal.py
"""Temporal search strategy for time-based queries."""

from __future__ import annotations

from typing import Any

from fitz_ai.core.chunk import Chunk

from .base import BaseVectorSearch


class TemporalSearch(BaseVectorSearch):
    """
    Temporal query search strategy.

    Handles time-based comparisons and period filtering with
    temporal-aware query generation and result tagging.
    """

    def execute(
        self,
        query: str,
        chunks: list[Chunk],
        intent: Any,
        references: list,
        temporal_queries: list[str],
    ) -> list[Chunk]:
        """
        Execute search with temporal awareness.

        Args:
            query: Original query
            chunks: Pre-existing chunks
            intent: TemporalIntent enum value
            references: List of TemporalReference objects
            temporal_queries: List of queries to search

        Returns:
            Merged and temporally-aware chunks
        """
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        logger = get_logger(__name__)

        logger.debug(f"{RETRIEVER} TemporalSearch: {len(temporal_queries)} temporal queries")

        # Map chunk_id -> (RRF score, Chunk)
        rrf_scores: dict[str, float] = {}
        chunk_lookup: dict[str, Chunk] = {}

        # Track which temporal references each chunk matches
        chunk_temporal_tags: dict[str, list[str]] = {}

        # Collect all variations with their temporal query index for batch embedding
        all_variations: list[tuple[int, str]] = []  # (temporal_query_idx, variation)
        for idx, tq in enumerate(temporal_queries):
            query_variations = self._get_query_variations(tq)
            for variation in query_variations:
                all_variations.append((idx, variation))

        # Batch embed all variations in one API call
        variation_texts = [v[1] for v in all_variations]
        all_vectors = self._embed_batch(variation_texts) if variation_texts else []

        # Process each variation with its embedding
        for (idx, variation), query_vector in zip(all_variations, all_vectors):
            results = self._hybrid_search(variation, query_vector)

            # Add RRF scores
            for rank, chunk in enumerate(results, start=1):
                rrf_delta = 1.0 / (self.rrf_k + rank)
                if chunk.id in rrf_scores:
                    rrf_scores[chunk.id] += rrf_delta
                else:
                    rrf_scores[chunk.id] = rrf_delta
                    chunk_lookup[chunk.id] = chunk
                    chunk_temporal_tags[chunk.id] = []

                # Tag with temporal reference if this is a focused query
                if idx > 0 and idx <= len(references):
                    ref = references[idx - 1]
                    if ref.text not in chunk_temporal_tags.get(chunk.id, []):
                        chunk_temporal_tags.setdefault(chunk.id, []).append(ref.text)

        # Sort by combined RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build final result list with temporal metadata
        results = []
        for chunk_id in sorted_ids[: self.k]:
            if chunk_id in chunk_lookup:
                chunk = chunk_lookup[chunk_id]
                chunk.metadata["temporal_rrf_score"] = rrf_scores[chunk_id]

                # Add temporal tags if present
                if chunk_id in chunk_temporal_tags and chunk_temporal_tags[chunk_id]:
                    chunk.metadata["temporal_refs"] = chunk_temporal_tags[chunk_id]

                results.append(chunk)

        # Search derived collection
        if self.include_derived:
            query_vector = self._embed(query)
            derived_results = self._search_derived(query_vector)
            results = self._merge_derived_results(results, derived_results)

        # Ensure table chunks are included
        results = self._ensure_table_chunks(results)

        # Expand with entity graph
        results = self._expand_by_entity_graph(results)

        # Apply keyword filtering
        results = self._apply_keyword_filter(results, query)

        logger.debug(
            f"{RETRIEVER} TemporalSearch: {len(temporal_queries)} queries → {len(results)} chunks"
        )

        # Preserve pre-existing chunks
        if chunks:
            return chunks + results

        return results
