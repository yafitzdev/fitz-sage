# fitz_ai/engines/fitz_rag/retrieval/steps/strategies/base.py
"""Base classes for search strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fitz_ai.core.chunk import Chunk
from fitz_ai.engines.fitz_rag.protocols import ChatClient, Embedder, VectorClient


class SearchStrategy(ABC):
    """
    Base strategy for executing vector search.

    Each strategy handles a specific type of query (semantic, temporal,
    aggregation, comparison) with specialized retrieval logic.
    """

    @abstractmethod
    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """
        Execute search for the given query.

        Args:
            query: User's query string
            chunks: Pre-existing chunks to preserve (e.g., from artifacts)

        Returns:
            List of retrieved chunks (pre-existing chunks + new results)
        """
        pass


class BaseVectorSearch(SearchStrategy):
    """
    Base class for vector search strategies with shared utilities.

    Provides common functionality like embedding, vector search,
    hybrid search, query expansion, filtering, etc.
    """

    def __init__(
        self,
        client: VectorClient,
        embedder: Embedder,
        collection: str,
        k: int = 25,
        filter_conditions: dict[str, Any] | None = None,
        rrf_k: int = 60,
        keyword_matcher: Any | None = None,
        entity_graph: Any | None = None,
        max_entity_expansion: int = 10,
        include_derived: bool = True,
    ):
        """
        Initialize base vector search.

        Args:
            client: Vector database client
            embedder: Embedding service
            collection: Collection name to search
            k: Number of candidates to retrieve per query
            filter_conditions: Optional metadata filtering
            rrf_k: RRF constant for score fusion
            keyword_matcher: Optional keyword matcher for filtering
            entity_graph: Optional entity graph for expansion
            max_entity_expansion: Max related chunks from entity graph
            include_derived: Whether to search derived collection
        """
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.k = k
        self.filter_conditions = filter_conditions or {}
        self.rrf_k = rrf_k
        self.keyword_matcher = keyword_matcher
        self.entity_graph = entity_graph
        self.max_entity_expansion = max_entity_expansion
        self.include_derived = include_derived

        # Lazy-loaded components
        self._sparse_index: Any = None
        self._query_expander: Any = None

    def _embed(self, query: str) -> list[float]:
        """Embed query text."""
        from fitz_ai.engines.fitz_rag.exceptions import EmbeddingError

        try:
            return self.embedder.embed(query)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query: {query!r}") from exc

    def _search(self, query_vector: list[float]) -> list[Any]:
        """Search vector DB."""
        from fitz_ai.engines.fitz_rag.exceptions import VectorSearchError

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

            # Flatten nested metadata
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

    def _get_sparse_index(self):
        """Lazy-load sparse index for hybrid search."""
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        logger = get_logger(__name__)

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
                from fitz_ai.retrieval.sparse import SparseIndex

                self._sparse_index = SparseIndex(self.collection)

        return self._sparse_index

    def _get_query_expander(self):
        """Lazy-load query expander."""
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        logger = get_logger(__name__)

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
        """Get query variations (synonyms, acronyms)."""
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        logger = get_logger(__name__)

        expander = self._get_query_expander()
        if expander is None:
            return [query]

        try:
            return expander.expand(query)
        except Exception as e:
            logger.debug(f"{RETRIEVER} Query expansion failed: {e}")
            return [query]

    def _hybrid_search(self, query: str, query_vector: list[float]) -> list[Chunk]:
        """Perform hybrid search combining dense and sparse results with RRF."""
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        from ..utils import normalize_record

        logger = get_logger(__name__)

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
        rrf_scores: dict[str, float] = {}

        # Add dense ranks
        for rank, chunk in enumerate(dense_chunks, start=1):
            rrf_scores[chunk.id] = 1.0 / (self.rrf_k + rank)

        # Add sparse ranks
        for rank, hit in enumerate(sparse_hits, start=1):
            if hit.chunk_id in rrf_scores:
                rrf_scores[hit.chunk_id] += 1.0 / (self.rrf_k + rank)
            else:
                rrf_scores[hit.chunk_id] = 1.0 / (self.rrf_k + rank)

        # Build chunk lookup
        chunk_lookup: dict[str, Chunk] = {c.id: c for c in dense_chunks}

        # Fetch sparse-only chunks
        sparse_only_ids = [hit.chunk_id for hit in sparse_hits if hit.chunk_id not in chunk_lookup]
        if sparse_only_ids:
            try:
                records = self.client.retrieve(
                    self.collection,
                    ids=sparse_only_ids,
                    with_payload=True,
                )
                for record in records:
                    chunk = normalize_record(record, extra_metadata={"from_sparse": True})
                    chunk_lookup[chunk.id] = chunk
            except Exception as e:
                logger.debug(f"{RETRIEVER} Failed to fetch sparse-only chunks: {e}")

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for chunk_id in sorted_ids[: self.k]:
            if chunk_id in chunk_lookup:
                chunk = chunk_lookup[chunk_id]
                chunk.metadata["rrf_score"] = rrf_scores[chunk_id]
                results.append(chunk)

        sparse_count = sum(1 for c in results if c.metadata.get("from_sparse"))
        logger.debug(
            f"{RETRIEVER} Hybrid search: {len(dense_chunks)} dense + "
            f"{len(sparse_hits)} sparse → {len(results)} merged "
            f"({sparse_count} sparse-only)"
        )

        return results

    def _expanded_search(self, query_variations: list[str]) -> list[Chunk]:
        """Search with multiple query variations and merge with RRF."""
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        logger = get_logger(__name__)

        if len(query_variations) == 1:
            # Single query - just run hybrid search
            query = query_variations[0]
            query_vector = self._embed(query)
            return self._hybrid_search(query, query_vector)

        # Multiple queries - search each and merge with RRF
        rrf_scores: dict[str, float] = {}
        chunk_lookup: dict[str, Chunk] = {}

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

    def _search_derived(self, query_vector: list[float]) -> list[Chunk]:
        """Search derived collection for pre-computed structured query results."""
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        logger = get_logger(__name__)

        if not self.include_derived:
            return []

        try:
            from fitz_ai.structured.constants import get_derived_collection
            from fitz_ai.structured.derived import FIELD_CONTENT, FIELD_SOURCE_TABLE
        except ImportError:
            return []

        derived_collection = get_derived_collection(self.collection)

        try:
            hits = self.client.search(
                collection_name=derived_collection,
                query_vector=query_vector,
                limit=self.k,
                with_payload=True,
                query_filter=self.filter_conditions if self.filter_conditions else None,
            )
        except Exception as e:
            logger.debug(f"{RETRIEVER} Derived collection search skipped: {e}")
            return []

        if not hits:
            return []

        # Convert derived hits to Chunk objects
        results: list[Chunk] = []
        for idx, hit in enumerate(hits):
            payload = getattr(hit, "payload", None) or getattr(hit, "metadata", None) or {}
            if not isinstance(payload, dict):
                payload = {}

            source_table = payload.get(FIELD_SOURCE_TABLE, "")
            content = payload.get(FIELD_CONTENT, "")

            metadata = {
                **payload,
                "from_derived": True,
                "is_derived": True,
                "source_table": source_table,
                "vector_score": getattr(hit, "score", None),
            }

            chunk = Chunk(
                id=f"derived_{getattr(hit, 'id', idx)}",
                doc_id=f"table:{source_table}" if source_table else "derived",
                content=content,
                chunk_index=0,
                metadata=metadata,
            )
            results.append(chunk)

        if results:
            logger.debug(
                f"{RETRIEVER} Derived search: found {len(results)} pre-computed results"
            )

        return results

    def _merge_derived_results(
        self, main_results: list[Chunk], derived_results: list[Chunk]
    ) -> list[Chunk]:
        """Merge derived results with main results, deduplicating by content."""
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        logger = get_logger(__name__)

        if not derived_results:
            return main_results

        # Track seen content
        seen_content = set()
        merged = []

        # Add derived results first (higher priority)
        for chunk in derived_results:
            content_key = chunk.content.strip().lower()[:200]
            if content_key not in seen_content:
                seen_content.add(content_key)
                merged.append(chunk)

        # Add main results, skipping duplicates
        for chunk in main_results:
            content_key = chunk.content.strip().lower()[:200]
            if content_key not in seen_content:
                seen_content.add(content_key)
                merged.append(chunk)

        logger.debug(
            f"{RETRIEVER} Merged {len(derived_results)} derived + "
            f"{len(main_results)} main → {len(merged)} total"
        )

        return merged

    def _ensure_table_chunks(self, results: list[Chunk]) -> list[Chunk]:
        """Ensure table schema chunks are included in results."""
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        from ..utils import normalize_record

        logger = get_logger(__name__)

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
                chunk = normalize_record(record)
                if chunk.id not in seen_ids:
                    results.append(chunk)
                    seen_ids.add(chunk.id)
                    added += 1

            if added:
                logger.debug(f"{RETRIEVER} Added {added} table chunks from registry")

        except Exception as e:
            logger.debug(f"{RETRIEVER} Table fetch failed: {e}")

        return results

    def _expand_by_entity_graph(self, results: list[Chunk]) -> list[Chunk]:
        """Expand results with related chunks via shared entities."""
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        from ..utils import normalize_record

        logger = get_logger(__name__)

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
                chunk = normalize_record(record, extra_metadata={"from_entity_graph": True})
                if chunk.id not in seen_ids:
                    results.append(chunk)
                    seen_ids.add(chunk.id)
                    added += 1

            if added:
                logger.debug(f"{RETRIEVER} Entity graph added {added} related chunks")

        except Exception as e:
            logger.debug(f"{RETRIEVER} Entity graph expansion failed: {e}")

        return results

    def _apply_keyword_filter(self, results: list[Chunk], query: str) -> list[Chunk]:
        """Apply keyword filtering if matcher is available."""
        from fitz_ai.logging.logger import get_logger
        from fitz_ai.logging.tags import RETRIEVER

        logger = get_logger(__name__)

        if not self.keyword_matcher:
            return results

        keywords_in_query = self.keyword_matcher.find_in_query(query)
        if not keywords_in_query:
            return results

        # Keep chunks matching keywords OR table schema chunks
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

        return filtered
