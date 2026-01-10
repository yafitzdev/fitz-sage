# fitz_ai/engines/fitz_rag/retrieval/steps/vector_search.py
"""
Vector Search Step - Intelligent retrieval from vector database.

Embeds query and searches for top-k candidates. Automatically applies:
- Multi-query expansion for long queries (when chat client available)
- Keyword filtering for exact term matching (when keyword_matcher available)
- Deduplication of results from multiple queries

These features are baked in - not configurable via plugins.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from fitz_ai.core.chunk import Chunk
from fitz_ai.engines.fitz_rag.exceptions import EmbeddingError, VectorSearchError
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

from .base import ChatClient, Embedder, KeywordMatcherClient, RetrievalStep, VectorClient

logger = get_logger(__name__)


@dataclass
class VectorSearchStep(RetrievalStep):
    """
    Intelligent vector search step with automatic multi-query expansion.

    This is the core retrieval step that handles:
    - Standard vector search for short queries
    - Automatic query expansion for long queries (> min_query_length)
    - Comparison query handling (ensures both compared entities are retrieved)
    - Keyword filtering for exact term matching
    - Deduplication of results

    These features are always active when dependencies are available.
    No plugin configuration needed - sophistication is baked in.

    Args:
        client: Vector database client
        embedder: Embedding service
        collection: Collection name to search
        chat: Fast-tier chat client for query expansion (optional, enables multi-query)
        keyword_matcher: Keyword matcher for exact term filtering (optional)
        k: Number of candidates to retrieve per query (default: 25)
        min_query_length: Minimum query length to trigger expansion (default: 300)
        max_queries: Maximum number of expanded queries (default: 5)
        filter_conditions: Optional Qdrant-style filter for metadata filtering
    """

    # Comparison patterns - triggers comparison-aware query expansion
    COMPARISON_PATTERNS: ClassVar[tuple[str, ...]] = (
        r"\bvs\.?\b",                    # "vs" or "vs."
        r"\bversus\b",                   # "versus"
        r"\bcompare[ds]?\b",             # "compare", "compared", "compares"
        r"\bcomparing\b",                # "comparing"
        r"\bdifference\s+between\b",     # "difference between"
        r"\bhow\s+does\s+.+\s+compare\b",  # "how does X compare"
        r"\bwhich\s+(?:is|has|one)\s+(?:better|faster|slower|higher|lower)\b",
    )

    client: VectorClient
    embedder: Embedder
    collection: str
    chat: ChatClient | None = None
    keyword_matcher: KeywordMatcherClient | None = None
    k: int = 25
    min_query_length: int = 300
    max_queries: int = 5
    filter_conditions: dict[str, Any] = field(default_factory=dict)

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """
        Execute vector search with automatic query expansion.

        Routing logic:
        1. Comparison queries (detected by pattern) → comparison-aware expansion
        2. Long queries (≥ min_query_length) → generic multi-query expansion
        3. Short queries → single vector search

        Pre-existing chunks (e.g., artifacts) are preserved and prepended.
        """
        # Check for comparison pattern first (regardless of query length)
        if self.chat is not None and self._is_comparison_query(query):
            return self._comparison_search(query, chunks)

        # Existing logic for long queries
        use_multi_query = (
            self.chat is not None and len(query) >= self.min_query_length
        )

        if use_multi_query:
            return self._multi_search(query, chunks)
        else:
            return self._single_search(query, chunks)

    def _single_search(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Standard single-query vector search with keyword filtering."""
        logger.debug(
            f"{RETRIEVER} VectorSearchStep: single search, k={self.k}, "
            f"collection={self.collection}"
        )

        query_vector = self._embed(query)
        hits = self._search(query_vector)
        results = self._hits_to_chunks(hits)

        # Apply keyword filtering if available
        if self.keyword_matcher:
            keywords_in_query = self.keyword_matcher.find_in_query(query)
            if keywords_in_query:
                filtered = [
                    c for c in results
                    if self.keyword_matcher.chunk_matches_any(c, keywords_in_query)
                ]
                logger.debug(
                    f"{RETRIEVER} Keyword filter: {len(results)} → {len(filtered)} chunks "
                    f"(keywords: {[k.id for k in keywords_in_query]})"
                )
                results = filtered

        logger.debug(f"{RETRIEVER} VectorSearchStep: retrieved {len(results)} chunks")

        # Preserve any pre-existing chunks (e.g., artifacts)
        if chunks:
            logger.debug(
                f"{RETRIEVER} VectorSearchStep: preserving {len(chunks)} pre-existing chunks"
            )
            return chunks + results

        return results

    def _multi_search(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Expand query via LLM, run multiple searches, combine results."""
        search_queries = self._expand_query(query)
        logger.info(
            f"{RETRIEVER} VectorSearchStep: expanded to {len(search_queries)} queries: "
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
                    filtered = [
                        c for c in sub_results
                        if self.keyword_matcher.chunk_matches_any(c, keywords_in_sq)
                    ]
                    logger.debug(
                        f"{RETRIEVER} Keyword filter for '{sq}': "
                        f"{len(sub_results)} → {len(filtered)} chunks "
                        f"(keywords: {[k.id for k in keywords_in_sq]})"
                    )
                    sub_results = filtered

            # Deduplicate across queries
            for chunk in sub_results:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    all_results.append(chunk)

        logger.debug(
            f"{RETRIEVER} VectorSearchStep: retrieved {len(all_results)} unique chunks "
            f"from {len(search_queries)} queries"
        )

        # Preserve any pre-existing chunks (e.g., artifacts)
        if chunks:
            logger.debug(
                f"{RETRIEVER} VectorSearchStep: preserving {len(chunks)} pre-existing chunks"
            )
            return chunks + all_results

        return all_results

    def _expand_query(self, query: str) -> list[str]:
        """Use fast LLM to extract key search terms."""
        prompt = f"""Extract the key search terms from this text. Return 3-5 short, focused search queries that would help find relevant documentation.

Text:
{query}

Return ONLY a JSON array of strings, no explanation. Example: ["query 1", "query 2", "query 3"]"""

        response = self.chat.chat([{"role": "user", "content": prompt}])
        return self._parse_json_list(response)

    def _parse_json_list(self, response: str) -> list[str]:
        """Parse JSON list from LLM response, with fallback."""
        try:
            text = response.strip()
            if text.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1].strip()
                    if text.startswith("json"):
                        text = text[4:].strip()
            result = json.loads(text)
            if isinstance(result, list):
                return [str(q) for q in result[: self.max_queries]]
        except json.JSONDecodeError:
            pass

        # Fallback: split by newlines
        lines = [q.strip() for q in response.strip().split("\n") if q.strip()]
        return lines[: self.max_queries]

    # -------------------------------------------------------------------------
    # Comparison Query Handling
    # -------------------------------------------------------------------------

    def _is_comparison_query(self, query: str) -> bool:
        """Detect if query is asking for a comparison between entities."""
        query_lower = query.lower()
        for pattern in self.COMPARISON_PATTERNS:
            if re.search(pattern, query_lower):
                return True
        return False

    def _comparison_search(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Handle comparison queries by ensuring both entities are retrieved."""
        search_queries = self._expand_comparison_query(query)
        logger.info(
            f"{RETRIEVER} VectorSearchStep: comparison query expanded to "
            f"{len(search_queries)} queries: {search_queries}"
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
                    ]

            # Deduplicate across queries
            for chunk in sub_results:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    all_results.append(chunk)

        logger.debug(
            f"{RETRIEVER} VectorSearchStep: retrieved {len(all_results)} unique chunks "
            f"from {len(search_queries)} comparison queries"
        )

        # Preserve any pre-existing chunks (e.g., artifacts)
        if chunks:
            logger.debug(
                f"{RETRIEVER} VectorSearchStep: preserving {len(chunks)} pre-existing chunks"
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
                    # Allow 2 extra queries for comparison (beyond max_queries)
                    return [str(q) for q in queries[: self.max_queries + 2]]
        except json.JSONDecodeError:
            pass

        # Fallback to generic expansion
        logger.warning(
            f"{RETRIEVER} Failed to parse comparison response, falling back to generic expansion"
        )
        return self._expand_query(original_query)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _embed(self, query: str) -> list[float]:
        """Embed query text."""
        try:
            return self.embedder.embed(query)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query: {query!r}") from exc

    def _search(self, query_vector: list[float]) -> list[Any]:
        """Search vector DB."""
        try:
            return self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=self.k,
                with_payload=True,
                query_filter=self.filter_conditions if self.filter_conditions else None,
            )
        except Exception as exc:
            raise VectorSearchError(f"Vector search failed: {exc}") from exc

    def _hits_to_chunks(self, hits: list[Any]) -> list[Chunk]:
        """Convert vector DB hits to Chunk objects."""
        results: list[Chunk] = []
        for idx, hit in enumerate(hits):
            payload = getattr(hit, "payload", None) or getattr(hit, "metadata", None) or {}
            if not isinstance(payload, dict):
                payload = {}

            chunk = Chunk(
                id=str(getattr(hit, "id", idx)),
                doc_id=str(
                    payload.get("doc_id")
                    or payload.get("document_id")
                    or payload.get("source")
                    or "unknown"
                ),
                content=str(payload.get("content") or payload.get("text") or ""),
                chunk_index=int(payload.get("chunk_index", idx)),
                metadata={
                    **payload,
                    "vector_score": getattr(hit, "score", None),
                },
            )
            results.append(chunk)

        return results
