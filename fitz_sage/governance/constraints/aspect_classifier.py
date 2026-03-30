# fitz_sage/governance/constraints/aspect_classifier.py
"""
Query and chunk aspect classification for intent alignment.

Aspects represent different facets of an entity:
- CAUSE: Why something happens, root causes
- SYMPTOM: Observable effects, manifestations
- TREATMENT: Solutions, interventions, fixes
- DEFINITION: What something is, core explanation
- PROCESS: How something works, steps involved
- PRICING: Cost, financial information
- COMPARISON: Evaluation against alternatives
- TIMELINE: When something happened, temporal sequence

Classification is backed by SemanticMatcher (embedding similarity).
When no matcher is available, both methods return GENERAL — the safe
neutral default that skips aspect-based filtering entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fitz_sage.governance.constraints.semantic import SemanticMatcher


class QueryAspect(Enum):
    """Query intent aspects."""

    CAUSE = "cause"
    EFFECT = "effect"  # Consequences, outcomes, results
    SYMPTOM = "symptom"
    TREATMENT = "treatment"
    DEFINITION = "definition"
    PROCESS = "process"  # How it works, mechanism, manufacturing
    APPLICATION = "application"  # Use cases, applications, purposes
    PRICING = "pricing"
    COMPARISON = "comparison"
    TIMELINE = "timeline"
    PROOF = "proof"  # Mathematical proofs, evidence, verification
    GENERAL = "general"  # Catch-all / no matcher available


@dataclass
class AspectMatch:
    """Result of aspect compatibility check."""

    compatible: bool
    query_aspect: QueryAspect
    chunk_aspects: list[QueryAspect]
    reason: str


class AspectClassifier:
    """Classifies query and chunk aspects for intent alignment.

    Delegates entirely to SemanticMatcher for embedding-based classification.
    When no matcher is provided, returns GENERAL (neutral) — aspect-based
    filtering is simply skipped rather than approximated with heuristics.
    """

    def __init__(self, semantic_matcher: SemanticMatcher | None = None) -> None:
        self._semantic_matcher = semantic_matcher

    def classify_query(self, query: str) -> QueryAspect:
        """Classify query into aspect category via SemanticMatcher.

        Returns GENERAL when no matcher is configured.
        """
        if self._semantic_matcher is not None:
            return self._semantic_matcher.classify_query_aspect(query)
        return QueryAspect.GENERAL

    def extract_chunk_aspects(self, chunk_content: str) -> list[QueryAspect]:
        """Extract aspects present in chunk content via SemanticMatcher.

        Returns [GENERAL] when no matcher is configured.
        """
        if self._semantic_matcher is not None:
            return self._semantic_matcher.classify_chunk_aspects(chunk_content)
        return [QueryAspect.GENERAL]

    def check_compatibility(self, query: str, chunk_content: str) -> AspectMatch:
        """Check if query aspect is compatible with chunk aspects."""
        query_aspect = self.classify_query(query)
        chunk_aspects = self.extract_chunk_aspects(chunk_content)

        if query_aspect == QueryAspect.GENERAL:
            return AspectMatch(
                compatible=True,
                query_aspect=query_aspect,
                chunk_aspects=chunk_aspects,
                reason="General query matches any content",
            )

        if QueryAspect.GENERAL in chunk_aspects:
            return AspectMatch(
                compatible=True,
                query_aspect=query_aspect,
                chunk_aspects=chunk_aspects,
                reason="General content matches any query",
            )

        if query_aspect in chunk_aspects:
            return AspectMatch(
                compatible=True,
                query_aspect=query_aspect,
                chunk_aspects=chunk_aspects,
                reason=f"Query aspect {query_aspect.value} found in chunk",
            )

        return AspectMatch(
            compatible=False,
            query_aspect=query_aspect,
            chunk_aspects=chunk_aspects,
            reason=f"Query asks about {query_aspect.value}, chunk discusses {[a.value for a in chunk_aspects]}",
        )


__all__ = ["AspectClassifier", "AspectMatch", "QueryAspect"]
