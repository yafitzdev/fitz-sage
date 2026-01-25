# fitz_ai/engines/fitz_rag/retrieval/steps/vector_search.py
"""
Vector Search Step - Intelligent retrieval from vector database.

Embeds query and searches for top-k candidates. Automatically applies:
- Query rewriting (conversational context, clarity, retrieval optimization)
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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fitz_ai.core.chunk import Chunk
from fitz_ai.engines.fitz_rag.protocols import Embedder, VectorClient
from fitz_ai.llm.factory import ChatFactory
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

from .base import RetrievalStep
from .strategies import (
    AggregationSearch,
    ComparisonSearch,
    SemanticSearch,
    TemporalSearch,
)

if TYPE_CHECKING:
    from fitz_ai.retrieval.detection import DetectionOrchestrator, DetectionSummary

# Check if derived collection support is available
try:
    from fitz_ai.structured import constants as _struct_constants  # noqa: F401

    DERIVED_AVAILABLE = True
    del _struct_constants  # Clean up namespace
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
        chat_factory: Chat factory for per-task tier selection (optional)
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

    client: VectorClient
    embedder: Embedder
    collection: str
    chat_factory: ChatFactory | None = None
    keyword_matcher: Any | None = None
    entity_graph: Any | None = None
    k: int = 25
    min_query_length: int = 300
    max_queries: int = 5
    max_entity_expansion: int = 10
    filter_conditions: dict[str, Any] = field(default_factory=dict)
    rrf_k: int = 60
    include_derived: bool = True

    # Conversation context for query rewriting (optional)
    conversation_context: Any | None = None

    # Lazy-loaded unified detection orchestrator
    _detection_orchestrator: "DetectionOrchestrator | None" = field(
        default=None, init=False, repr=False
    )

    # Lazy-loaded strategies
    _semantic_strategy: SemanticSearch | None = field(default=None, init=False, repr=False)
    _aggregation_strategy: AggregationSearch | None = field(default=None, init=False, repr=False)
    _temporal_strategy: TemporalSearch | None = field(default=None, init=False, repr=False)
    _comparison_strategy: ComparisonSearch | None = field(default=None, init=False, repr=False)

    # Lazy-loaded HyDE generator
    _hyde_generator: Any = field(default=None, init=False, repr=False)

    # Lazy-loaded query rewriter
    _query_rewriter: Any = field(default=None, init=False, repr=False)

    # Current rewrite result (for strategies to access)
    _current_rewrite: Any = field(default=None, init=False, repr=False)

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """
        Execute vector search with automatic query routing.

        Flow:
        0. Query rewriting (conversational, clarity, retrieval optimization)
        1. Run unified detection
        2. Route to specialized strategy based on detection
        """
        # Step 0: Rewrite query if rewriter is available
        rewrite_result = self._rewrite_query(query)
        effective_query = rewrite_result.rewritten_query
        self._current_rewrite = rewrite_result

        # Step 1: Run unified detection
        detection = self._get_detection_summary(effective_query)

        # Step 2: Route to strategy based on detection
        # Check aggregation first (highest priority for list/count queries)
        if detection.has_aggregation_intent:
            strategy = self._get_aggregation_strategy()
            return strategy.execute(effective_query, chunks, detection.aggregation)

        # Check temporal queries
        if detection.has_temporal_intent:
            strategy = self._get_temporal_strategy()
            return strategy.execute(effective_query, chunks, detection.temporal)

        # Check comparison queries (requires chat_factory for LLM expansion)
        if self.chat_factory is not None and detection.has_comparison_intent:
            strategy = self._get_comparison_strategy()
            return strategy.execute(effective_query, chunks, detection.comparison)

        # Default: semantic search (pass rewrite result and detection for enhancements)
        strategy = self._get_semantic_strategy()
        return strategy.execute(query, chunks, rewrite_result=rewrite_result, detection=detection)

    def _get_hyde_generator(self):
        """Lazy-load HyDE generator when chat_factory is available."""
        if self._hyde_generator is None and self.chat_factory is not None:
            try:
                from fitz_ai.retrieval.hyde import HydeGenerator

                self._hyde_generator = HydeGenerator(chat_factory=self.chat_factory)
                logger.debug(f"{RETRIEVER} HyDE generator initialized")
            except Exception as e:
                logger.debug(f"{RETRIEVER} Failed to load HyDE generator: {e}")
                self._hyde_generator = None

        return self._hyde_generator

    def _get_query_rewriter(self):
        """Lazy-load query rewriter when chat_factory is available."""
        if self._query_rewriter is None and self.chat_factory is not None:
            try:
                from fitz_ai.retrieval.rewriter import QueryRewriter

                self._query_rewriter = QueryRewriter(chat_factory=self.chat_factory)
                logger.debug(f"{RETRIEVER} Query rewriter initialized")
            except Exception as e:
                logger.debug(f"{RETRIEVER} Failed to load query rewriter: {e}")
                self._query_rewriter = None

        return self._query_rewriter

    def _rewrite_query(self, query: str):
        """Rewrite query if rewriter is available."""
        rewriter = self._get_query_rewriter()
        if rewriter is None:
            # Return a no-op result
            from fitz_ai.retrieval.rewriter.types import RewriteResult, RewriteType

            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                rewrite_type=RewriteType.NONE,
                confidence=1.0,
            )

        return rewriter.rewrite(query, self.conversation_context)

    def _get_detection_orchestrator(self) -> "DetectionOrchestrator":
        """Lazy-load the unified detection orchestrator with LLM classification."""
        if self._detection_orchestrator is None:
            from fitz_ai.retrieval.detection import DetectionOrchestrator

            # Pass chat factory for LLM-based detection
            self._detection_orchestrator = DetectionOrchestrator(chat_factory=self.chat_factory)
            logger.debug(
                f"{RETRIEVER} Detection orchestrator initialized (LLM: {self.chat_factory is not None})"
            )

        return self._detection_orchestrator

    def _get_detection_summary(self, query: str) -> "DetectionSummary":
        """Run unified detection on the query."""
        orchestrator = self._get_detection_orchestrator()
        return orchestrator.detect_for_retrieval(query)

    def _get_semantic_strategy(self) -> SemanticSearch:
        """Lazy-load semantic search strategy."""
        if self._semantic_strategy is None:
            self._semantic_strategy = SemanticSearch(
                client=self.client,
                embedder=self.embedder,
                collection=self.collection,
                chat_factory=self.chat_factory,
                k=self.k,
                min_query_length=self.min_query_length,
                max_queries=self.max_queries,
                filter_conditions=self.filter_conditions,
                rrf_k=self.rrf_k,
                keyword_matcher=self.keyword_matcher,
                entity_graph=self.entity_graph,
                max_entity_expansion=self.max_entity_expansion,
                include_derived=self.include_derived,
                hyde_generator=self._get_hyde_generator(),
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
                chat_factory=self.chat_factory,
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
        if self.chat_factory is None:
            return [query]
        strategy = self._get_comparison_strategy()
        return strategy._expand_comparison_query(query)
