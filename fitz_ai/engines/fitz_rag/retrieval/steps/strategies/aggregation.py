# fitz_ai/engines/fitz_rag/retrieval/steps/strategies/aggregation.py
"""Aggregation search strategy for list/count/enumerate queries."""

from __future__ import annotations

from typing import Any

from fitz_ai.core.chunk import Chunk

from .base import BaseVectorSearch


class AggregationSearch(BaseVectorSearch):
    """
    Aggregation query search strategy.

    Handles list all, count, and enumerate queries with comprehensive
    coverage by fetching more results and using multiple variations.
    """

    def execute(self, query: str, chunks: list[Chunk], aggregation_result: Any) -> list[Chunk]:
        """
        Execute search optimized for aggregation queries.

        Args:
            query: Original query
            chunks: Pre-existing chunks
            aggregation_result: AggregationResult from detector

        Returns:
            Comprehensive set of chunks for aggregation
        """
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        logger = get_logger(__name__)

        # Calculate expanded k for comprehensive coverage
        base_k = self.k
        expanded_k = base_k * aggregation_result.fetch_multiplier

        logger.debug(
            f"{RETRIEVER} AggregationSearch: k={base_k}→{expanded_k}, "
            f"target='{aggregation_result.intent.target}'"
        )

        # Temporarily increase k for this search
        original_k = self.k
        self.k = expanded_k

        try:
            # Use augmented query for better retrieval
            search_query = aggregation_result.augmented_query or query

            # Get query variations (synonyms, acronyms)
            query_variations = self._get_query_variations(search_query)

            # Also add variations of the original query
            if search_query != query:
                original_variations = self._get_query_variations(query)
                seen = set(query_variations)
                for var in original_variations:
                    if var not in seen:
                        query_variations.append(var)
                        seen.add(var)

            # Search with all variations and merge with RRF
            results = self._expanded_search(query_variations)

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

            # Tag results with aggregation metadata
            for chunk in results:
                chunk.metadata["aggregation_type"] = aggregation_result.intent.type.name
                chunk.metadata["aggregation_target"] = aggregation_result.intent.target

            logger.debug(
                f"{RETRIEVER} AggregationSearch: {len(query_variations)} variations → "
                f"{len(results)} chunks"
            )

        finally:
            # Restore original k
            self.k = original_k

        # Preserve pre-existing chunks
        if chunks:
            return chunks + results

        return results
