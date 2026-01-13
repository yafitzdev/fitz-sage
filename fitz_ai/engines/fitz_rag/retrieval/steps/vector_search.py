# fitz_ai/engines/fitz_rag/retrieval/steps/vector_search.py
"""
Vector Search Step - Intelligent retrieval from vector database.

Embeds query and searches for top-k candidates. Automatically applies:
- Aggregation query handling (list all, count, enumerate queries)
- Temporal query handling (time-based comparisons and period filtering)
- Query expansion (synonym/acronym variations) for improved recall
- Hybrid search (dense + sparse) with RRF fusion (when sparse index available)
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

from .base import (
    ChatClient,
    Embedder,
    EntityGraphClient,
    KeywordMatcherClient,
    RetrievalStep,
    VectorClient,
)

logger = get_logger(__name__)


@dataclass
class VectorSearchStep(RetrievalStep):
    """
    Intelligent vector search step with automatic query expansion.

    This is the core retrieval step that handles:
    - Temporal query handling (time comparisons, period filtering)
    - Query expansion (synonym/acronym variations for improved recall)
    - Hybrid search (dense + sparse) with RRF fusion
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
        entity_graph: Entity graph for related chunk discovery (optional)
        k: Number of candidates to retrieve per query (default: 25)
        min_query_length: Minimum query length to trigger expansion (default: 300)
        max_queries: Maximum number of expanded queries (default: 5)
        max_entity_expansion: Maximum related chunks to add from entity graph (default: 10)
        filter_conditions: Optional Qdrant-style filter for metadata filtering
        rrf_k: RRF constant for score fusion (default: 60)
    """

    # Comparison patterns - triggers comparison-aware query expansion
    COMPARISON_PATTERNS: ClassVar[tuple[str, ...]] = (
        r"\bvs\.?\b",  # "vs" or "vs."
        r"\bversus\b",  # "versus"
        r"\bcompare[ds]?\b",  # "compare", "compared", "compares"
        r"\bcomparing\b",  # "comparing"
        r"\bdifference\s+between\b",  # "difference between"
        r"\bhow\s+does\s+.+\s+compare\b",  # "how does X compare"
        r"\bwhich\s+(?:is|has|one)\s+(?:better|faster|slower|higher|lower)\b",
    )

    client: VectorClient
    embedder: Embedder
    collection: str
    chat: ChatClient | None = None
    keyword_matcher: KeywordMatcherClient | None = None
    entity_graph: EntityGraphClient | None = None
    k: int = 25
    min_query_length: int = 300
    max_queries: int = 5
    max_entity_expansion: int = 10
    filter_conditions: dict[str, Any] = field(default_factory=dict)
    rrf_k: int = 60  # RRF constant (higher = more weight to lower ranks)

    # Lazy-loaded sparse index for hybrid search
    _sparse_index: Any = field(default=None, init=False, repr=False)
    # Lazy-loaded query expander
    _query_expander: Any = field(default=None, init=False, repr=False)
    # Lazy-loaded temporal detector
    _temporal_detector: Any = field(default=None, init=False, repr=False)
    # Lazy-loaded aggregation detector
    _aggregation_detector: Any = field(default=None, init=False, repr=False)

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """
        Execute vector search with automatic query expansion.

        Routing logic:
        1. Aggregation queries (list all, count, enumerate) → comprehensive retrieval
        2. Temporal queries (time comparisons/periods) → temporal-aware search
        3. Comparison queries (detected by pattern) → comparison-aware expansion
        4. Long queries (≥ min_query_length) → generic multi-query expansion
        5. Short queries → single vector search

        Pre-existing chunks (e.g., artifacts) are preserved and prepended.
        """
        # Check for aggregation query first (list all, count, enumerate)
        aggregation_result = self._check_aggregation_query(query)
        if aggregation_result is not None:
            return self._aggregation_search(query, chunks, aggregation_result)

        # Check for temporal query (time-based comparisons, periods)
        temporal_result = self._check_temporal_query(query)
        if temporal_result is not None:
            intent, references, temporal_queries = temporal_result
            return self._temporal_search(query, chunks, intent, references, temporal_queries)

        # Check for comparison pattern (regardless of query length)
        if self.chat is not None and self._is_comparison_query(query):
            return self._comparison_search(query, chunks)

        # Existing logic for long queries
        use_multi_query = self.chat is not None and len(query) >= self.min_query_length

        if use_multi_query:
            return self._multi_search(query, chunks)
        else:
            return self._single_search(query, chunks)

    def _single_search(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Standard single-query search with query expansion and hybrid fusion."""
        logger.debug(
            f"{RETRIEVER} VectorSearchStep: single search, k={self.k}, collection={self.collection}"
        )

        # Expand query to variations (synonyms, acronyms)
        query_variations = self._get_query_variations(query)

        # Search with all query variations and merge with RRF
        results = self._expanded_search(query_variations)

        # Always include table schema chunks - they may not match semantically
        # but TableQueryStep needs them to execute SQL queries
        results = self._ensure_table_chunks(results)

        # Expand with related chunks via entity graph
        results = self._expand_by_entity_graph(results)

        # Apply keyword filtering if available (use original query for keyword detection)
        if self.keyword_matcher:
            keywords_in_query = self.keyword_matcher.find_in_query(query)
            if keywords_in_query:
                # Keep chunks matching keywords OR table schema chunks
                # (tables don't contain all data, so keyword filtering would wrongly exclude them)
                filtered = [
                    c
                    for c in results
                    if self.keyword_matcher.chunk_matches_any(c, keywords_in_query)
                    or c.metadata.get("is_table_schema")
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
            f"{RETRIEVER} VectorSearchStep: retrieved {len(all_results)} unique chunks "
            f"from {len(search_queries)} queries"
        )

        # Ensure table schema chunks are included
        all_results = self._ensure_table_chunks(all_results)

        # Expand with related chunks via entity graph
        all_results = self._expand_by_entity_graph(all_results)

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
                        or c.metadata.get("is_table_schema")
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

        # Ensure table schema chunks are included
        all_results = self._ensure_table_chunks(all_results)

        # Expand with related chunks via entity graph
        all_results = self._expand_by_entity_graph(all_results)

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

    def _ensure_table_chunks(self, results: list[Chunk]) -> list[Chunk]:
        """
        Ensure table schema chunks are included in results.

        Table schema chunks may have low semantic similarity to queries about
        specific entities (e.g., "how much does Iris Johnson earn?") because
        the schema only contains column names and sample rows, not all data.

        Uses table registry (created at ingestion) to fetch table chunks by ID.
        """
        # Check if we already have table schema chunks
        has_tables = any(c.metadata.get("is_table_schema") for c in results)
        if has_tables:
            return results

        # Get table IDs from registry
        from fitz_ai.tabular.registry import get_table_ids

        table_ids = get_table_ids(self.collection)
        if not table_ids:
            return results

        # Fetch table chunks by ID
        try:
            seen_ids = {c.id for c in results}
            records = self.client.retrieve(
                self.collection,
                ids=table_ids,
                with_payload=True,
            )

            added = 0
            for record in records:
                record_id = (
                    record.get("id") if isinstance(record, dict) else getattr(record, "id", None)
                )
                if record_id and str(record_id) not in seen_ids:
                    payload = (
                        record.get("payload", {})
                        if isinstance(record, dict)
                        else getattr(record, "payload", {})
                    )
                    metadata = payload.get("metadata", {})
                    if isinstance(metadata, dict):
                        flat_metadata = {**payload, **metadata}
                    else:
                        flat_metadata = payload

                    chunk = Chunk(
                        id=str(record_id),
                        doc_id=str(payload.get("doc_id", "unknown")),
                        content=str(payload.get("content", "")),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        metadata=flat_metadata,
                    )
                    results.append(chunk)
                    seen_ids.add(str(record_id))
                    added += 1

            if added:
                logger.debug(f"{RETRIEVER} Added {added} table chunks from registry")

        except Exception as e:
            logger.debug(f"{RETRIEVER} Table fetch failed: {e}")

        return results

    def _expand_by_entity_graph(self, results: list[Chunk]) -> list[Chunk]:
        """
        Expand results with related chunks via shared entities.

        Uses the entity graph to find chunks that share entities with the
        retrieved chunks. This helps answer questions that require information
        from multiple documents connected by shared entities.

        Example:
            Retrieved: "Sarah Chen leads Company XY"
            Entity graph finds: "Company XY manufactures cars"
            → Both chunks now available for answering "What does Sarah's company make?"
        """
        if not self.entity_graph:
            return results

        if not results:
            return results

        # Get IDs of current results
        current_ids = [c.id for c in results]
        seen_ids = set(current_ids)

        # Find related chunks via entity graph
        try:
            related_ids = self.entity_graph.get_related_chunks(
                chunk_ids=current_ids,
                max_total=self.max_entity_expansion,
            )

            if not related_ids:
                return results

            # Fetch related chunks by ID
            records = self.client.retrieve(
                self.collection,
                ids=related_ids,
                with_payload=True,
            )

            added = 0
            for record in records:
                record_id = (
                    record.get("id") if isinstance(record, dict) else getattr(record, "id", None)
                )
                if record_id and str(record_id) not in seen_ids:
                    payload = (
                        record.get("payload", {})
                        if isinstance(record, dict)
                        else getattr(record, "payload", {})
                    )
                    metadata = payload.get("metadata", {})
                    if isinstance(metadata, dict):
                        flat_metadata = {**payload, **metadata, "from_entity_graph": True}
                    else:
                        flat_metadata = {**payload, "from_entity_graph": True}

                    chunk = Chunk(
                        id=str(record_id),
                        doc_id=str(payload.get("doc_id", "unknown")),
                        content=str(payload.get("content", "")),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        metadata=flat_metadata,
                    )
                    results.append(chunk)
                    seen_ids.add(str(record_id))
                    added += 1

            if added:
                logger.debug(f"{RETRIEVER} Entity graph added {added} related chunks")

        except Exception as e:
            logger.debug(f"{RETRIEVER} Entity graph expansion failed: {e}")

        return results

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

            # Flatten nested metadata (ingestion stores chunk.metadata under "metadata" key)
            nested_metadata = payload.get("metadata", {})
            if isinstance(nested_metadata, dict):
                flat_metadata = {
                    **payload,
                    **nested_metadata,
                    "vector_score": getattr(hit, "score", None),
                }
            else:
                flat_metadata = {**payload, "vector_score": getattr(hit, "score", None)}

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
                metadata=flat_metadata,
            )
            results.append(chunk)

        return results

    # -------------------------------------------------------------------------
    # Hybrid Search (Dense + Sparse with RRF Fusion)
    # -------------------------------------------------------------------------

    def _get_sparse_index(self):
        """Lazy-load the sparse index for this collection."""
        if self._sparse_index is None:
            try:
                from fitz_ai.retrieval.sparse import SparseIndex

                self._sparse_index = SparseIndex.load(self.collection)
                if self._sparse_index.is_ready():
                    logger.debug(
                        f"{RETRIEVER} Loaded sparse index: {len(self._sparse_index)} documents"
                    )
                else:
                    logger.debug(f"{RETRIEVER} No sparse index available for {self.collection}")
            except Exception as e:
                logger.debug(f"{RETRIEVER} Failed to load sparse index: {e}")
                # Create empty placeholder to avoid retrying
                from fitz_ai.retrieval.sparse import SparseIndex

                self._sparse_index = SparseIndex(self.collection)

        return self._sparse_index

    def _hybrid_search(self, query: str, query_vector: list[float]) -> list[Chunk]:
        """
        Perform hybrid search combining dense and sparse results with RRF.

        Reciprocal Rank Fusion (RRF) formula:
            score(d) = sum(1 / (k + rank_i(d))) for each retrieval method i

        Args:
            query: Original query text (for sparse search)
            query_vector: Query embedding (for dense search)

        Returns:
            Combined and re-ranked chunks
        """
        sparse_index = self._get_sparse_index()

        # Get dense results
        dense_hits = self._search(query_vector)
        dense_chunks = self._hits_to_chunks(dense_hits)

        # If no sparse index, return dense results only
        if not sparse_index.is_ready():
            return dense_chunks

        # Get sparse results
        sparse_hits = sparse_index.search(query, k=self.k)

        if not sparse_hits:
            return dense_chunks

        # Build RRF scores
        # Map chunk_id -> RRF score
        rrf_scores: dict[str, float] = {}

        # Add dense ranks (1-indexed for RRF formula)
        for rank, chunk in enumerate(dense_chunks, start=1):
            rrf_scores[chunk.id] = 1.0 / (self.rrf_k + rank)

        # Add sparse ranks
        for rank, hit in enumerate(sparse_hits, start=1):
            if hit.chunk_id in rrf_scores:
                rrf_scores[hit.chunk_id] += 1.0 / (self.rrf_k + rank)
            else:
                rrf_scores[hit.chunk_id] = 1.0 / (self.rrf_k + rank)

        # Build chunk lookup from dense results
        chunk_lookup: dict[str, Chunk] = {c.id: c for c in dense_chunks}

        # Fetch any sparse-only chunks from vector DB
        sparse_only_ids = [hit.chunk_id for hit in sparse_hits if hit.chunk_id not in chunk_lookup]
        if sparse_only_ids:
            try:
                records = self.client.retrieve(
                    self.collection,
                    ids=sparse_only_ids,
                    with_payload=True,
                )
                for record in records:
                    record_id = (
                        record.get("id")
                        if isinstance(record, dict)
                        else getattr(record, "id", None)
                    )
                    if record_id:
                        payload = (
                            record.get("payload", {})
                            if isinstance(record, dict)
                            else getattr(record, "payload", {})
                        )
                        metadata = payload.get("metadata", {})
                        if isinstance(metadata, dict):
                            flat_metadata = {**payload, **metadata, "from_sparse": True}
                        else:
                            flat_metadata = {**payload, "from_sparse": True}

                        chunk = Chunk(
                            id=str(record_id),
                            doc_id=str(payload.get("doc_id", "unknown")),
                            content=str(payload.get("content", "")),
                            chunk_index=int(payload.get("chunk_index", 0)),
                            metadata=flat_metadata,
                        )
                        chunk_lookup[str(record_id)] = chunk
            except Exception as e:
                logger.debug(f"{RETRIEVER} Failed to fetch sparse-only chunks: {e}")

        # Sort by RRF score and build result list
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for chunk_id in sorted_ids[: self.k]:
            if chunk_id in chunk_lookup:
                chunk = chunk_lookup[chunk_id]
                # Store RRF score for debugging/logging
                chunk.metadata["rrf_score"] = rrf_scores[chunk_id]
                results.append(chunk)

        sparse_count = sum(1 for c in results if c.metadata.get("from_sparse"))
        logger.debug(
            f"{RETRIEVER} Hybrid search: {len(dense_chunks)} dense + "
            f"{len(sparse_hits)} sparse → {len(results)} merged "
            f"({sparse_count} sparse-only)"
        )

        return results

    # -------------------------------------------------------------------------
    # Query Expansion (Synonym/Acronym Variations)
    # -------------------------------------------------------------------------

    def _get_query_expander(self):
        """Lazy-load the query expander."""
        if self._query_expander is None:
            try:
                from fitz_ai.retrieval.expansion import QueryExpander

                self._query_expander = QueryExpander()
                logger.debug(f"{RETRIEVER} Query expander initialized")
            except Exception as e:
                logger.debug(f"{RETRIEVER} Failed to load query expander: {e}")
                self._query_expander = None

        return self._query_expander

    def _get_query_variations(self, query: str) -> list[str]:
        """
        Get query variations using the expander.

        Returns list with at least the original query.
        """
        expander = self._get_query_expander()
        if expander is None:
            return [query]

        try:
            return expander.expand(query)
        except Exception as e:
            logger.debug(f"{RETRIEVER} Query expansion failed: {e}")
            return [query]

    def _expanded_search(self, query_variations: list[str]) -> list[Chunk]:
        """
        Search with multiple query variations and merge results with RRF.

        For each variation:
        1. Embed the query
        2. Run hybrid search (dense + sparse)
        3. Collect ranked results

        Then merge all results using Reciprocal Rank Fusion.

        Args:
            query_variations: List of query strings (original + expansions)

        Returns:
            Merged and re-ranked chunks
        """
        if len(query_variations) == 1:
            # Single query - just run hybrid search
            query = query_variations[0]
            query_vector = self._embed(query)
            return self._hybrid_search(query, query_vector)

        # Multiple queries - search each and merge with RRF
        # Map chunk_id -> (RRF score, Chunk)
        rrf_scores: dict[str, float] = {}
        chunk_lookup: dict[str, Chunk] = {}

        for variation in query_variations:
            query_vector = self._embed(variation)
            results = self._hybrid_search(variation, query_vector)

            # Add RRF scores for this variation's results
            for rank, chunk in enumerate(results, start=1):
                rrf_delta = 1.0 / (self.rrf_k + rank)
                if chunk.id in rrf_scores:
                    rrf_scores[chunk.id] += rrf_delta
                else:
                    rrf_scores[chunk.id] = rrf_delta
                    chunk_lookup[chunk.id] = chunk

        # Sort by combined RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build final result list
        results = []
        for chunk_id in sorted_ids[: self.k]:
            if chunk_id in chunk_lookup:
                chunk = chunk_lookup[chunk_id]
                chunk.metadata["expansion_rrf_score"] = rrf_scores[chunk_id]
                results.append(chunk)

        logger.debug(
            f"{RETRIEVER} Query expansion: {len(query_variations)} variations → "
            f"{len(results)} chunks (from {len(rrf_scores)} unique)"
        )

        return results

    # -------------------------------------------------------------------------
    # Temporal Query Handling (Time-Based Comparisons and Periods)
    # -------------------------------------------------------------------------

    def _get_temporal_detector(self):
        """Lazy-load the temporal detector."""
        if self._temporal_detector is None:
            try:
                from fitz_ai.retrieval.temporal import TemporalDetector

                self._temporal_detector = TemporalDetector()
                logger.debug(f"{RETRIEVER} Temporal detector initialized")
            except Exception as e:
                logger.debug(f"{RETRIEVER} Failed to load temporal detector: {e}")
                self._temporal_detector = None

        return self._temporal_detector

    def _check_temporal_query(self, query: str):
        """
        Check if query has temporal intent and extract references.

        Returns:
            Tuple of (intent, references, temporal_queries) if temporal query detected,
            None otherwise.
        """
        detector = self._get_temporal_detector()
        if detector is None:
            return None

        try:
            from fitz_ai.retrieval.temporal import TemporalIntent

            intent, references = detector.detect(query)

            if intent == TemporalIntent.NONE:
                return None

            # Generate sub-queries for temporal coverage
            temporal_queries = detector.generate_temporal_queries(query, intent, references)

            logger.debug(
                f"{RETRIEVER} Temporal query detected: intent={intent.value}, "
                f"refs={[r.text for r in references]}, queries={len(temporal_queries)}"
            )

            return intent, references, temporal_queries

        except Exception as e:
            logger.debug(f"{RETRIEVER} Temporal detection failed: {e}")
            return None

    def _temporal_search(
        self,
        query: str,
        chunks: list[Chunk],
        intent: Any,
        references: list,
        temporal_queries: list[str],
    ) -> list[Chunk]:
        """
        Execute search with temporal awareness.

        For temporal queries:
        1. Search with each temporal sub-query
        2. Tag results with their temporal reference
        3. Merge results with RRF
        4. Ensure coverage of all time periods mentioned

        Args:
            query: Original query
            chunks: Pre-existing chunks (artifacts)
            intent: TemporalIntent enum value
            references: List of TemporalReference objects
            temporal_queries: List of queries to search

        Returns:
            Merged and temporally-aware chunks
        """
        logger.debug(
            f"{RETRIEVER} VectorSearchStep: temporal search with "
            f"{len(temporal_queries)} queries"
        )

        # Map chunk_id -> (RRF score, Chunk)
        rrf_scores: dict[str, float] = {}
        chunk_lookup: dict[str, Chunk] = {}

        # Track which temporal references each chunk matches
        chunk_temporal_tags: dict[str, list[str]] = {}

        for idx, tq in enumerate(temporal_queries):
            # Expand each temporal query with synonyms/acronyms
            query_variations = self._get_query_variations(tq)

            for variation in query_variations:
                query_vector = self._embed(variation)
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

        # Ensure table schema chunks are included
        results = self._ensure_table_chunks(results)

        # Expand with related chunks via entity graph
        results = self._expand_by_entity_graph(results)

        # Apply keyword filtering (using original query)
        if self.keyword_matcher:
            keywords_in_query = self.keyword_matcher.find_in_query(query)
            if keywords_in_query:
                filtered = [
                    c
                    for c in results
                    if self.keyword_matcher.chunk_matches_any(c, keywords_in_query)
                    or c.metadata.get("is_table_schema")
                ]
                logger.debug(f"{RETRIEVER} Keyword filter: {len(results)} → {len(filtered)} chunks")
                results = filtered

        logger.debug(
            f"{RETRIEVER} Temporal search: {len(temporal_queries)} queries → "
            f"{len(results)} chunks"
        )

        # Preserve any pre-existing chunks
        if chunks:
            return chunks + results

        return results

    # -------------------------------------------------------------------------
    # Aggregation Query Handling (List All, Count, Enumerate)
    # -------------------------------------------------------------------------

    def _get_aggregation_detector(self):
        """Lazy-load the aggregation detector."""
        if self._aggregation_detector is None:
            try:
                from fitz_ai.retrieval.aggregation import AggregationDetector

                self._aggregation_detector = AggregationDetector()
                logger.debug(f"{RETRIEVER} Aggregation detector initialized")
            except Exception as e:
                logger.debug(f"{RETRIEVER} Failed to load aggregation detector: {e}")
                self._aggregation_detector = None

        return self._aggregation_detector

    def _check_aggregation_query(self, query: str):
        """
        Check if query has aggregation intent (list all, count, enumerate).

        Returns:
            AggregationResult if detected, None otherwise.
        """
        detector = self._get_aggregation_detector()
        if detector is None:
            return None

        try:
            result = detector.detect(query)

            if not result.detected:
                return None

            logger.debug(
                f"{RETRIEVER} Aggregation query detected: type={result.intent.type.name}, "
                f"target='{result.intent.target}', multiplier={result.fetch_multiplier}"
            )

            return result

        except Exception as e:
            logger.debug(f"{RETRIEVER} Aggregation detection failed: {e}")
            return None

    def _aggregation_search(
        self,
        query: str,
        chunks: list[Chunk],
        aggregation_result: Any,
    ) -> list[Chunk]:
        """
        Execute search optimized for aggregation queries.

        Aggregation queries need comprehensive coverage:
        1. Use augmented query for better retrieval
        2. Fetch more chunks (multiplied by fetch_multiplier)
        3. Use multiple query variations for diversity
        4. Tag results with aggregation metadata

        Args:
            query: Original query
            chunks: Pre-existing chunks (artifacts)
            aggregation_result: AggregationResult from detector

        Returns:
            Comprehensive set of chunks for aggregation
        """
        # Calculate expanded k for comprehensive coverage
        base_k = self.k
        expanded_k = base_k * aggregation_result.fetch_multiplier

        logger.debug(
            f"{RETRIEVER} VectorSearchStep: aggregation search, "
            f"k={base_k}→{expanded_k}, target='{aggregation_result.intent.target}'"
        )

        # Temporarily increase k for this search
        original_k = self.k
        self.k = expanded_k

        try:
            # Use augmented query for better retrieval
            search_query = aggregation_result.augmented_query or query

            # Get query variations (synonyms, acronyms)
            query_variations = self._get_query_variations(search_query)

            # Also add variations of the original query to ensure coverage
            if search_query != query:
                original_variations = self._get_query_variations(query)
                # Merge unique variations
                seen = set(query_variations)
                for var in original_variations:
                    if var not in seen:
                        query_variations.append(var)
                        seen.add(var)

            # Search with all variations and merge with RRF
            results = self._expanded_search(query_variations)

            # Always include table schema chunks for potential SQL aggregations
            results = self._ensure_table_chunks(results)

            # Expand with related chunks via entity graph
            # (important for comprehensive aggregation)
            results = self._expand_by_entity_graph(results)

            # Apply keyword filtering if available (use original query)
            if self.keyword_matcher:
                keywords_in_query = self.keyword_matcher.find_in_query(query)
                if keywords_in_query:
                    filtered = [
                        c
                        for c in results
                        if self.keyword_matcher.chunk_matches_any(c, keywords_in_query)
                        or c.metadata.get("is_table_schema")
                    ]
                    logger.debug(
                        f"{RETRIEVER} Keyword filter: {len(results)} → {len(filtered)} chunks"
                    )
                    results = filtered

            # Tag results with aggregation metadata
            for chunk in results:
                chunk.metadata["aggregation_type"] = aggregation_result.intent.type.name
                chunk.metadata["aggregation_target"] = aggregation_result.intent.target

            logger.debug(
                f"{RETRIEVER} Aggregation search: {len(query_variations)} variations → "
                f"{len(results)} chunks"
            )

        finally:
            # Restore original k
            self.k = original_k

        # Preserve any pre-existing chunks
        if chunks:
            return chunks + results

        return results
