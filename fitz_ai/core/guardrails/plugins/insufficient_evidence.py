# fitz_ai/core/guardrails/plugins/insufficient_evidence.py
"""
Insufficient Evidence Constraint - Default guardrail for evidence coverage.

This constraint prevents the system from giving confident answers when
there is not enough direct evidence in the retrieved chunks.

Uses semantic matching for language-agnostic evidence detection.

It does NOT:
- Rank authority
- Resolve ambiguity
- Guess intent
- Use LLM calls

It only enforces: "Is there explicit evidence to justify a decisive answer?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from fitz_ai.core.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from ..base import ConstraintResult
from ..semantic import SemanticMatcher

logger = get_logger(__name__)


# =============================================================================
# Constraint Implementation
# =============================================================================


@dataclass
class InsufficientEvidenceConstraint:
    """
    Constraint that prevents confident answers without sufficient evidence.

    This constraint checks:
    1. Are there any chunks at all?
    2. For causal questions: is there causal language?
    3. For fact questions: are there direct assertions?

    Uses semantic embedding similarity for language-agnostic detection.

    Attributes:
        semantic_matcher: SemanticMatcher instance for embedding-based detection
        enabled: Whether this constraint is active (default: True)
        min_evidence_count: Minimum chunks with evidence required (default: 1)
    """

    semantic_matcher: SemanticMatcher
    enabled: bool = True
    min_evidence_count: int = 1

    @property
    def name(self) -> str:
        return "insufficient_evidence"

    def apply(
        self,
        query: str,
        chunks: Sequence[Chunk],
    ) -> ConstraintResult:
        """
        Check if there is sufficient evidence to answer the query.

        Args:
            query: The user's question
            chunks: Retrieved chunks

        Returns:
            ConstraintResult - denies if insufficient evidence
        """
        if not self.enabled:
            return ConstraintResult.allow()

        # Rule 1: Empty context
        if not chunks:
            logger.info(f"{PIPELINE} InsufficientEvidenceConstraint: no chunks retrieved")
            return ConstraintResult.deny(
                reason="No evidence retrieved",
                signal="abstain",
                evidence_count=0,
            )

        # Determine query type using semantic matching
        is_causal = self.semantic_matcher.is_causal_query(query)
        is_fact = self.semantic_matcher.is_fact_query(query)

        # Rule 2: Causal queries need causal evidence
        if is_causal:
            evidence_count = self.semantic_matcher.count_causal_chunks(chunks)

            if evidence_count < self.min_evidence_count:
                logger.info(
                    f"{PIPELINE} InsufficientEvidenceConstraint: "
                    f"causal query but no causal evidence (found {evidence_count})"
                )
                return ConstraintResult.deny(
                    reason="No explicit causal evidence found",
                    signal="abstain",
                    evidence_count=evidence_count,
                    query_type="causal",
                )

        # Rule 3: Fact queries need assertions
        elif is_fact:
            evidence_count = self.semantic_matcher.count_assertion_chunks(chunks)

            if evidence_count < self.min_evidence_count:
                logger.info(
                    f"{PIPELINE} InsufficientEvidenceConstraint: "
                    f"fact query but no direct assertion (found {evidence_count})"
                )
                return ConstraintResult.deny(
                    reason="No direct assertion found in retrieved evidence",
                    signal="abstain",
                    evidence_count=evidence_count,
                    query_type="fact",
                )

        # For other query types (or if evidence found), allow
        logger.debug(f"{PIPELINE} InsufficientEvidenceConstraint: sufficient evidence")
        return ConstraintResult.allow()


__all__ = ["InsufficientEvidenceConstraint"]
