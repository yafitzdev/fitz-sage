# fitz_ai/engines/fitz_rag/retrieval/steps/strategies/semantic.py
"""Semantic search strategy for standard queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fitz_ai.core.chunk import Chunk
from fitz_ai.engines.fitz_rag.protocols import ChatClient, Embedder, VectorClient
from fitz_ai.engines.fitz_rag.retrieval.steps.utils import parse_json_list

from .base import BaseVectorSearch

if TYPE_CHECKING:
    from fitz_ai.retrieval.hyde import HydeGenerator
    from fitz_ai.retrieval.rewriter.types import RewriteResult


class SemanticSearch(BaseVectorSearch):
    """
    Standard semantic search strategy.

    Handles single and multi-query expansion with hybrid search,
    query expansion, and keyword filtering.
    """

    def __init__(
        self,
        client: VectorClient,
        embedder: Embedder,
        collection: str,
        chat: ChatClient | None = None,
        k: int = 25,
        min_query_length: int = 300,
        max_queries: int = 5,
        hyde_generator: "HydeGenerator | None" = None,
        **kwargs,
    ):
        """
        Initialize semantic search strategy.

        Args:
            client: Vector database client
            embedder: Embedding service
            collection: Collection name
            chat: Chat client for query expansion (optional)
            k: Number of results to retrieve
            min_query_length: Minimum query length for multi-query expansion
            max_queries: Maximum number of expanded queries
            hyde_generator: HyDE generator for hypothetical documents (optional)
            **kwargs: Additional arguments for BaseVectorSearch
        """
        super().__init__(client, embedder, collection, k=k, **kwargs)
        self.chat = chat
        self.min_query_length = min_query_length
        self.max_queries = max_queries
        self.hyde_generator = hyde_generator

    def execute(
        self,
        query: str,
        chunks: list[Chunk],
        rewrite_result: "RewriteResult | None" = None,
    ) -> list[Chunk]:
        """Execute semantic search with optional multi-query expansion."""
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        logger = get_logger(__name__)

        # Determine if we should use multi-query expansion
        use_multi_query = self.chat is not None and len(query) >= self.min_query_length

        if use_multi_query:
            results = self._multi_search(query)
        else:
            results = self._single_search(query, rewrite_result)

        # Preserve pre-existing chunks
        if chunks:
            logger.debug(
                f"{RETRIEVER} SemanticSearch: preserving {len(chunks)} pre-existing chunks"
            )
            return chunks + results

        return results

    def _single_search(
        self,
        query: str,
        rewrite_result: "RewriteResult | None" = None,
    ) -> list[Chunk]:
        """Standard single-query search with expansion and hybrid fusion."""
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        logger = get_logger(__name__)

        logger.debug(
            f"{RETRIEVER} SemanticSearch: single search, k={self.k}, collection={self.collection}"
        )

        # Start with base queries from rewrite result (original + rewritten + decomposed + disambiguated)
        # Check for any variations: rewritten, compound decomposition, or ambiguity
        if rewrite_result and len(rewrite_result.all_query_variations) > 1:
            base_queries = rewrite_result.all_query_variations
            logger.debug(
                f"{RETRIEVER} Query rewrite: using {len(base_queries)} query variations"
            )
        else:
            base_queries = [query]

        # Expand each base query with synonyms/acronyms
        query_variations = []
        for q in base_queries:
            query_variations.extend(self._get_query_variations(q))

        # HyDE: add hypothetical documents for improved recall (use original query)
        if self.hyde_generator:
            hypotheses = self.hyde_generator.generate(query)
            query_variations.extend(hypotheses)
            logger.debug(f"{RETRIEVER} HyDE: added {len(hypotheses)} hypotheses")

        # Search with all query variations and merge with RRF
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

        logger.debug(f"{RETRIEVER} SemanticSearch: retrieved {len(results)} chunks")

        return results

    def _multi_search(self, query: str) -> list[Chunk]:
        """Expand query via LLM, run multiple searches, combine results."""
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        logger = get_logger(__name__)

        search_queries = self._expand_query(query)
        logger.info(
            f"{RETRIEVER} SemanticSearch: expanded to {len(search_queries)} queries: "
            f"{search_queries}"
        )

        all_results: list[Chunk] = []
        seen_ids: set[str] = set()

        for sq in search_queries:
            query_vector = self._embed(sq)
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
            f"{RETRIEVER} SemanticSearch: retrieved {len(all_results)} unique chunks "
            f"from {len(search_queries)} queries"
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

        return all_results

    def _expand_query(self, query: str) -> list[str]:
        """Use fast LLM to extract key search terms."""
        prompt = f"""Extract the key search terms from this text. Return 3-5 short, focused search queries that would help find relevant documentation.

Text:
{query}

Return ONLY a JSON array of strings, no explanation. Example: ["query 1", "query 2", "query 3"]"""

        response = self.chat.chat([{"role": "user", "content": prompt}])
        return parse_json_list(response, max_items=self.max_queries)
