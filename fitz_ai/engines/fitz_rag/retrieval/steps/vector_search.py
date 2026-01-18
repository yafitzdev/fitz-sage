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

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from fitz_ai.core.chunk import Chunk
from fitz_ai.engines.fitz_rag.protocols import ChatClient, Embedder, VectorClient
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

from .base import RetrievalStep
from .strategies import (
    AggregationSearch,
    ComparisonSearch,
    SemanticSearch,
    TemporalSearch,
)

# Check if derived collection support is available
try:
    from fitz_ai.structured.constants import get_derived_collection
    from fitz_ai.structured.derived import FIELD_CONTENT, FIELD_DERIVED, FIELD_SOURCE_TABLE

    DERIVED_AVAILABLE = True
except ImportError:
    DERIVED_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class VectorSearchStep(RetrievalStep):
    """
    Intelligent vector search step with automatic query routing.

    Routes queries to specialized strategies based on query type:
    - Aggregation queries → AggregationSearch
    - Temporal queries → TemporalSearch
    - Comparison queries → ComparisonSearch
    - Standard queries → SemanticSearch

    Args:
        client: Vector database client
        embedder: Embedding service
        collection: Collection name to search
        chat: Fast-tier chat client for query expansion (optional)
        keyword_matcher: Keyword matcher for filtering (optional)
        entity_graph: Entity graph for expansion (optional)
        k: Number of candidates to retrieve per query (default: 25)
        min_query_length: Minimum query length for multi-query expansion (default: 300)
        max_queries: Maximum number of expanded queries (default: 5)
        max_entity_expansion: Maximum related chunks from entity graph (default: 10)
        filter_conditions: Optional metadata filtering
        rrf_k: RRF constant for score fusion (default: 60)
        include_derived: Include derived sentences from structured queries (default: True)
    """

    # Comparison patterns
    COMPARISON_PATTERNS: ClassVar[tuple[str, ...]] = (
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bcompare[ds]?\b",
        r"\bcomparing\b",
        r"\bdifference\s+between\b",
        r"\bhow\s+does\s+.+\s+compare\b",
        r"\bwhich\s+(?:is|has|one)\s+(?:better|faster|slower|higher|lower)\b",
    )

    client: VectorClient
    embedder: Embedder
    collection: str
    chat: ChatClient | None = None
    keyword_matcher: Any | None = None
    entity_graph: Any | None = None
    k: int = 25
    min_query_length: int = 300
    max_queries: int = 5
    max_entity_expansion: int = 10
    filter_conditions: dict[str, Any] = field(default_factory=dict)
    rrf_k: int = 60
    include_derived: bool = True

    # Lazy-loaded detectors
    _temporal_detector: Any = field(default=None, init=False, repr=False)
    _aggregation_detector: Any = field(default=None, init=False, repr=False)

    # Lazy-loaded strategies
    _semantic_strategy: SemanticSearch | None = field(default=None, init=False, repr=False)
    _aggregation_strategy: AggregationSearch | None = field(default=None, init=False, repr=False)
    _temporal_strategy: TemporalSearch | None = field(default=None, init=False, repr=False)
    _comparison_strategy: ComparisonSearch | None = field(default=None, init=False, repr=False)

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """
        Execute vector search with automatic query routing.

        Routing logic:
        1. Aggregation queries → AggregationSearch
        2. Temporal queries → TemporalSearch
        3. Comparison queries → ComparisonSearch
        4. Standard queries → SemanticSearch
        """
        # Check for aggregation query first
        aggregation_result = self._check_aggregation_query(query)
        if aggregation_result is not None:
            strategy = self._get_aggregation_strategy()
            return strategy.execute(query, chunks, aggregation_result)

        # Check for temporal query
        temporal_result = self._check_temporal_query(query)
        if temporal_result is not None:
            intent, references, temporal_queries = temporal_result
            strategy = self._get_temporal_strategy()
            return strategy.execute(query, chunks, intent, references, temporal_queries)

        # Check for comparison query
        if self.chat is not None and self._is_comparison_query(query):
            strategy = self._get_comparison_strategy()
            return strategy.execute(query, chunks)

        # Default: semantic search
        strategy = self._get_semantic_strategy()
        return strategy.execute(query, chunks)

    def _get_semantic_strategy(self) -> SemanticSearch:
        """Lazy-load semantic search strategy."""
        if self._semantic_strategy is None:
            self._semantic_strategy = SemanticSearch(
                client=self.client,
                embedder=self.embedder,
                collection=self.collection,
                chat=self.chat,
                k=self.k,
                min_query_length=self.min_query_length,
                max_queries=self.max_queries,
                filter_conditions=self.filter_conditions,
                rrf_k=self.rrf_k,
                keyword_matcher=self.keyword_matcher,
                entity_graph=self.entity_graph,
                max_entity_expansion=self.max_entity_expansion,
                include_derived=self.include_derived,
            )
        return self._semantic_strategy

    def _get_aggregation_strategy(self) -> AggregationSearch:
        """Lazy-load aggregation search strategy."""
        if self._aggregation_strategy is None:
            self._aggregation_strategy = AggregationSearch(
                client=self.client,
                embedder=self.embedder,
                collection=self.collection,
                k=self.k,
                filter_conditions=self.filter_conditions,
                rrf_k=self.rrf_k,
                keyword_matcher=self.keyword_matcher,
                entity_graph=self.entity_graph,
                max_entity_expansion=self.max_entity_expansion,
                include_derived=self.include_derived,
            )
        return self._aggregation_strategy

    def _get_temporal_strategy(self) -> TemporalSearch:
        """Lazy-load temporal search strategy."""
        if self._temporal_strategy is None:
            self._temporal_strategy = TemporalSearch(
                client=self.client,
                embedder=self.embedder,
                collection=self.collection,
                k=self.k,
                filter_conditions=self.filter_conditions,
                rrf_k=self.rrf_k,
                keyword_matcher=self.keyword_matcher,
                entity_graph=self.entity_graph,
                max_entity_expansion=self.max_entity_expansion,
                include_derived=self.include_derived,
            )
        return self._temporal_strategy

    def _get_comparison_strategy(self) -> ComparisonSearch:
        """Lazy-load comparison search strategy."""
        if self._comparison_strategy is None:
            self._comparison_strategy = ComparisonSearch(
                client=self.client,
                embedder=self.embedder,
                collection=self.collection,
                chat=self.chat,
                max_queries=self.max_queries,
                k=self.k,
                filter_conditions=self.filter_conditions,
                rrf_k=self.rrf_k,
                keyword_matcher=self.keyword_matcher,
                entity_graph=self.entity_graph,
                max_entity_expansion=self.max_entity_expansion,
                include_derived=self.include_derived,
            )
        return self._comparison_strategy

    def _is_comparison_query(self, query: str) -> bool:
        """Detect if query is asking for a comparison."""
        query_lower = query.lower()
        for pattern in self.COMPARISON_PATTERNS:
            if re.search(pattern, query_lower):
                return True
        return False

    def _get_temporal_detector(self):
        """Lazy-load temporal detector."""
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
        """Check if query has temporal intent."""
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

    def _get_aggregation_detector(self):
        """Lazy-load aggregation detector."""
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
        """Check if query has aggregation intent."""
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

    # -------------------------------------------------------------------------
    # Test compatibility methods - delegate to strategies
    # -------------------------------------------------------------------------

    def _search_derived(self, query_vector: list[float]) -> list[Chunk]:
        """Delegate to semantic strategy for backward compatibility with tests."""
        strategy = self._get_semantic_strategy()
        return strategy._search_derived(query_vector)

    def _merge_derived_results(
        self, main_results: list[Chunk], derived_results: list[Chunk]
    ) -> list[Chunk]:
        """Delegate to semantic strategy for backward compatibility with tests."""
        strategy = self._get_semantic_strategy()
        return strategy._merge_derived_results(main_results, derived_results)

    def _single_search(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Delegate to semantic strategy for backward compatibility with tests."""
        strategy = self._get_semantic_strategy()
        return strategy._single_search(query)

    def _expand_comparison_query(self, query: str) -> list[str]:
        """Delegate to comparison strategy for backward compatibility with tests."""
        if self.chat is None:
            return [query]
        strategy = self._get_comparison_strategy()
        return strategy._expand_comparison_query(query)
