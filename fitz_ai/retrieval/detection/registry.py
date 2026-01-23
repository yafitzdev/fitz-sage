# fitz_ai/retrieval/detection/registry.py
"""
Detection orchestrator and registry.

Uses LLM classification for robust query detection instead of brittle regex patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from fitz_ai.logging.logger import get_logger

from .llm_classifier import LLMClassifier
from .protocol import DetectionCategory, DetectionResult

if TYPE_CHECKING:
    from .llm_classifier import ChatProtocol

logger = get_logger(__name__)


class TemporalIntent(Enum):
    """Temporal query intent types."""

    COMPARISON = "COMPARISON"
    TREND = "TREND"
    POINT_IN_TIME = "POINT_IN_TIME"
    RANGE = "RANGE"
    SEQUENCE = "SEQUENCE"


class AggregationType(Enum):
    """Aggregation query types."""

    LIST = "LIST"
    COUNT = "COUNT"
    UNIQUE = "UNIQUE"


@dataclass
class DetectionSummary:
    """
    Summary of all detections for retrieval routing.

    Provides convenient accessors for VectorSearchStep routing decisions.
    """

    temporal: DetectionResult[Any]
    aggregation: DetectionResult[Any]
    comparison: DetectionResult[Any]
    freshness: DetectionResult[Any]
    expansion: DetectionResult[Any]
    rewriter: DetectionResult[Any]
    vocabulary: DetectionResult[Any]

    @property
    def has_temporal_intent(self) -> bool:
        """True if temporal intent detected."""
        return self.temporal.detected

    @property
    def has_aggregation_intent(self) -> bool:
        """True if aggregation intent detected."""
        return self.aggregation.detected

    @property
    def has_comparison_intent(self) -> bool:
        """True if comparison intent detected."""
        return self.comparison.detected

    @property
    def boost_recency(self) -> bool:
        """True if recency boosting should be applied."""
        return self.freshness.detected and self.freshness.metadata.get(
            "boost_recency", False
        )

    @property
    def boost_authority(self) -> bool:
        """True if authority boosting should be applied."""
        return self.freshness.detected and self.freshness.metadata.get(
            "boost_authority", False
        )

    @property
    def query_variations(self) -> list[str]:
        """Query variations from expansion detector."""
        if self.expansion.detected:
            return self.expansion.transformations
        return []

    @property
    def temporal_intent(self) -> TemporalIntent | None:
        """Get the temporal intent enum value."""
        if self.temporal.detected:
            return self.temporal.intent
        return None

    @property
    def aggregation_type(self) -> AggregationType | None:
        """Get the aggregation type enum value."""
        if self.aggregation.detected:
            return self.aggregation.intent
        return None

    @property
    def fetch_multiplier(self) -> int:
        """Get fetch multiplier from aggregation if detected."""
        if self.aggregation.detected:
            return self.aggregation.metadata.get("fetch_multiplier", 1)
        return 1

    @property
    def needs_rewriting(self) -> bool:
        """True if query might benefit from rewriting."""
        if not self.rewriter.detected:
            return False
        return self.rewriter.metadata.get("needs_context", False) or self.rewriter.metadata.get(
            "is_compound", False
        )

    @property
    def comparison_entities(self) -> list[str]:
        """Get entities being compared."""
        if self.comparison.detected:
            return self.comparison.metadata.get("entities", [])
        return []

    @property
    def comparison_queries(self) -> list[str]:
        """Get expanded comparison queries."""
        if self.comparison.detected:
            return self.comparison.transformations
        return []


@dataclass
class DetectionOrchestrator:
    """
    Orchestrates detection using LLM classification.

    Usage:
        orchestrator = DetectionOrchestrator(chat_client=chat)
        summary = orchestrator.detect_for_retrieval(query)

        if summary.has_temporal_intent:
            # Route to temporal strategy
            ...
    """

    chat_client: ChatProtocol | None = None

    # Lazy-loaded classifier and expansion detector
    _classifier: LLMClassifier | None = field(default=None, init=False, repr=False)
    _expansion_detector: Any = field(default=None, init=False, repr=False)

    def _ensure_classifier(self) -> LLMClassifier | None:
        """Lazy-load LLM classifier."""
        if self._classifier is None and self.chat_client is not None:
            self._classifier = LLMClassifier(chat_client=self.chat_client)
        return self._classifier

    def _get_expansion_detector(self) -> Any:
        """Lazy-load expansion detector."""
        if self._expansion_detector is None:
            from .detectors.expansion import ExpansionDetector

            self._expansion_detector = ExpansionDetector()
        return self._expansion_detector

    def detect_for_retrieval(self, query: str) -> DetectionSummary:
        """
        Run detection optimized for retrieval routing.

        Uses LLM classification for robust detection, with dict-based
        expansion for query variations.

        Args:
            query: User's query string

        Returns:
            DetectionSummary with routing information
        """
        classifier = self._ensure_classifier()

        if classifier:
            classification = classifier.classify(query)
        else:
            classification = {}
            logger.debug("No chat client available, skipping LLM classification")

        # Get expansion result from dict-based detector
        expansion_detector = self._get_expansion_detector()
        expansion_result = expansion_detector.detect(query)

        return DetectionSummary(
            temporal=self._build_temporal_result(classification.get("temporal", {})),
            aggregation=self._build_aggregation_result(classification.get("aggregation", {})),
            comparison=self._build_comparison_result(classification.get("comparison", {})),
            freshness=self._build_freshness_result(classification.get("freshness", {})),
            expansion=expansion_result,
            rewriter=self._build_rewriter_result(classification.get("rewriter", {})),
            vocabulary=DetectionResult.not_detected(DetectionCategory.VOCABULARY),
        )

    def _build_temporal_result(self, data: dict[str, Any]) -> DetectionResult[TemporalIntent]:
        """Build temporal detection result from LLM classification."""
        if not data.get("detected", False):
            return DetectionResult.not_detected(DetectionCategory.TEMPORAL)

        # Parse intent
        intent_str = data.get("intent")
        intent = None
        if intent_str:
            try:
                intent = TemporalIntent(intent_str)
            except ValueError:
                pass

        return DetectionResult(
            detected=True,
            category=DetectionCategory.TEMPORAL,
            confidence=0.9,
            intent=intent,
            matches=[],
            metadata={
                "references": data.get("references", []),
            },
            transformations=data.get("time_focused_queries", []),
        )

    def _build_aggregation_result(
        self, data: dict[str, Any]
    ) -> DetectionResult[AggregationType]:
        """Build aggregation detection result from LLM classification."""
        if not data.get("detected", False):
            return DetectionResult.not_detected(DetectionCategory.AGGREGATION)

        # Parse type
        type_str = data.get("type")
        agg_type = None
        if type_str:
            try:
                agg_type = AggregationType(type_str)
            except ValueError:
                pass

        return DetectionResult(
            detected=True,
            category=DetectionCategory.AGGREGATION,
            confidence=0.9,
            intent=agg_type,
            matches=[],
            metadata={
                "target": data.get("target"),
                "fetch_multiplier": data.get("fetch_multiplier", 3),
            },
            transformations=[],
        )

    def _build_comparison_result(self, data: dict[str, Any]) -> DetectionResult[None]:
        """Build comparison detection result from LLM classification."""
        if not data.get("detected", False):
            return DetectionResult.not_detected(DetectionCategory.COMPARISON)

        return DetectionResult(
            detected=True,
            category=DetectionCategory.COMPARISON,
            confidence=0.9,
            intent=None,
            matches=[],
            metadata={
                "entities": data.get("entities", []),
            },
            transformations=data.get("comparison_queries", []),
        )

    def _build_freshness_result(self, data: dict[str, Any]) -> DetectionResult[None]:
        """Build freshness detection result from LLM classification."""
        boost_recency = data.get("boost_recency", False)
        boost_authority = data.get("boost_authority", False)

        if not boost_recency and not boost_authority:
            return DetectionResult.not_detected(DetectionCategory.FRESHNESS)

        return DetectionResult(
            detected=True,
            category=DetectionCategory.FRESHNESS,
            confidence=0.9,
            intent=None,
            matches=[],
            metadata={
                "boost_recency": boost_recency,
                "boost_authority": boost_authority,
            },
            transformations=[],
        )

    def _build_rewriter_result(self, data: dict[str, Any]) -> DetectionResult[None]:
        """Build rewriter detection result from LLM classification."""
        needs_context = data.get("needs_context", False)
        is_compound = data.get("is_compound", False)

        if not needs_context and not is_compound:
            return DetectionResult.not_detected(DetectionCategory.REWRITER)

        return DetectionResult(
            detected=True,
            category=DetectionCategory.REWRITER,
            confidence=0.9,
            intent=None,
            matches=[],
            metadata={
                "needs_context": needs_context,
                "is_compound": is_compound,
            },
            transformations=data.get("decomposed_queries", []),
        )
