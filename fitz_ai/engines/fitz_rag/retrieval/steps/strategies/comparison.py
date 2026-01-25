# fitz_ai/engines/fitz_rag/retrieval/steps/strategies/comparison.py
"""Comparison search strategy for entity comparison queries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fitz_ai.core.chunk import Chunk
from fitz_ai.engines.fitz_rag.protocols import ChatClient, Embedder, VectorClient
from fitz_ai.engines.fitz_rag.retrieval.steps.utils import parse_json_list

from .base import BaseVectorSearch

if TYPE_CHECKING:
    from fitz_ai.retrieval.detection import DetectionResult


class ComparisonSearch(BaseVectorSearch):
    """
    Comparison query search strategy.

    Handles comparison queries by ensuring both compared entities
    are retrieved through targeted query expansion.
    """

    def __init__(
        self,
        client: VectorClient,
        embedder: Embedder,
        collection: str,
        chat: ChatClient,
        max_queries: int = 5,
        **kwargs,
    ):
        """
        Initialize comparison search strategy.

        Args:
            client: Vector database client
            embedder: Embedding service
            collection: Collection name
            chat: Chat client for query expansion (required)
            max_queries: Maximum number of expanded queries
            **kwargs: Additional arguments for BaseVectorSearch
        """
        super().__init__(client, embedder, collection, **kwargs)
        self.chat = chat
        self.max_queries = max_queries

    def execute(
        self,
        query: str,
        chunks: list[Chunk],
        detection_result: "DetectionResult[Any] | None" = None,
    ) -> list[Chunk]:
        """Handle comparison queries by ensuring both entities are retrieved."""
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        logger = get_logger(__name__)

        # Use comparison_queries from detection result if available
        search_queries = []
        if detection_result and detection_result.transformations:
            search_queries = detection_result.transformations
            logger.info(
                f"{RETRIEVER} ComparisonSearch: using {len(search_queries)} queries from detection"
            )
        else:
            # Fall back to LLM expansion if no detection result
            search_queries = self._expand_comparison_query(query)
            logger.info(
                f"{RETRIEVER} ComparisonSearch: expanded to {len(search_queries)} queries via LLM"
            )

        # Always include the original query
        if query not in search_queries:
            search_queries = [query] + search_queries

        logger.debug(f"{RETRIEVER} ComparisonSearch: search queries: {search_queries}")

        # Batch embed all comparison queries in one API call
        query_vectors = self._embed_batch(search_queries) if search_queries else []

        all_results: list[Chunk] = []
        seen_ids: set[str] = set()

        for sq, query_vector in zip(search_queries, query_vectors):
            hits = self._search(query_vector)
            sub_results = self._hits_to_chunks(hits)

            # Apply keyword filtering per sub-query
            if self.keyword_matcher:
                keywords_in_sq = self.keyword_matcher.find_in_query(sq)
                if keywords_in_sq:
                    sub_results = [
                        c
                        for c in sub_results
                        if self.keyword_matcher.chunk_matches_any(c, keywords_in_sq)
                        or c.metadata.get("is_table_schema")
                    ]

            # Deduplicate across queries
            for chunk in sub_results:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    all_results.append(chunk)

        logger.debug(
            f"{RETRIEVER} ComparisonSearch: retrieved {len(all_results)} unique chunks "
            f"from {len(search_queries)} comparison queries"
        )

        # Search derived collection
        if self.include_derived:
            query_vector = self._embed(query)
            derived_results = self._search_derived(query_vector)
            all_results = self._merge_derived_results(all_results, derived_results)

        # Ensure table chunks are included
        all_results = self._ensure_table_chunks(all_results)

        # Expand with entity graph
        all_results = self._expand_by_entity_graph(all_results)

        # Apply keyword filtering for original query (Bug fix: was missing)
        all_results = self._apply_keyword_filter(all_results, query)

        # Preserve pre-existing chunks
        if chunks:
            logger.debug(
                f"{RETRIEVER} ComparisonSearch: preserving {len(chunks)} pre-existing chunks"
            )
            return chunks + all_results

        return all_results

    def _expand_comparison_query(self, query: str) -> list[str]:
        """Extract comparison entities and generate targeted queries for each."""
        prompt = f"""This is a comparison query. Extract the entities being compared and generate search queries.

Query: {query}

Instructions:
1. Identify the two (or more) things being compared
2. Generate 2-3 search queries for EACH entity to retrieve relevant information
3. Generate 1 query that includes both entities together

Return ONLY a JSON object:
{{"entities": ["entity1", "entity2"], "queries": ["query1", "query2", ...]}}"""

        response = self.chat.chat([{"role": "user", "content": prompt}])
        return self._parse_comparison_response(response, query)

    def _parse_comparison_response(self, response: str, original_query: str) -> list[str]:
        """Parse comparison query expansion response."""
        import json

        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        logger = get_logger(__name__)

        try:
            text = response.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1].strip()
                    if text.startswith("json"):
                        text = text[4:].strip()

            result = json.loads(text)
            if isinstance(result, dict) and "queries" in result:
                queries = result["queries"]
                if isinstance(queries, list):
                    # Allow 2 extra queries for comparison
                    return [str(q) for q in queries[: self.max_queries + 2]]
        except json.JSONDecodeError:
            pass

        # Fallback to generic expansion
        logger.warning(
            f"{RETRIEVER} Failed to parse comparison response, falling back to generic expansion"
        )
        return self._expand_query(original_query)

    def _expand_query(self, query: str) -> list[str]:
        """Use fast LLM to extract key search terms."""
        prompt = f"""Extract the key search terms from this text. Return 3-5 short, focused search queries that would help find relevant documentation.

Text:
{query}

Return ONLY a JSON array of strings, no explanation. Example: ["query 1", "query 2", "query 3"]"""

        response = self.chat.chat([{"role": "user", "content": prompt}])
        return parse_json_list(response, max_items=self.max_queries)
