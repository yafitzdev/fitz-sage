# fitz_ai/core/guardrails/plugins/causal_attribution.py
"""
Causal Attribution Constraint - Prevents implicit causality claims.

This constraint prevents the system from synthesizing causal explanations
when documents only describe outcomes without explicit causal language.

Uses semantic matching for language-agnostic causal detection.

It enforces: "Don't invent causality that isn't explicitly stated."

This is NOT reasoning suppression. It's epistemic honesty enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from ..base import ConstraintResult
from ..semantic import SemanticMatcher

if TYPE_CHECKING:
    from fitz_ai.core.chunk import ChunkLike

logger = get_logger(__name__)


# =============================================================================
# Constraint Implementation
# =============================================================================


@dataclass
class CausalAttributionConstraint:
    """
    Constraint that prevents implicit causal synthesis.

    When a query requests causal explanation (why, what caused, etc.),
    this constraint verifies that retrieved documents contain explicit
    causal language before allowing a causal answer.

    Uses semantic embedding similarity for language-agnostic detection.

    This prevents the LLM from inventing causal relationships that
    aren't explicitly stated in the evidence.

    Attributes:
        semantic_matcher: SemanticMatcher instance for embedding-based detection
        enabled: Whether this constraint is active (default: True)
    """

    semantic_matcher: SemanticMatcher
    enabled: bool = True

    @property
    def name(self) -> str:
        return "causal_attribution"

    def apply(
        self,
        query: str,
        chunks: Sequence["ChunkLike"],
    ) -> ConstraintResult:
        """
        Check if causal queries have explicit causal evidence.

        Args:
            query: The user's question
            chunks: Retrieved chunks

        Returns:
            ConstraintResult - denies causal synthesis if no explicit causal evidence
        """
        if not self.enabled:
            return ConstraintResult.allow()

        # Only applies to causal queries
        if not self.semantic_matcher.is_causal_query(query):
            return ConstraintResult.allow()

        # Empty chunks - defer to InsufficientEvidenceConstraint
        if not chunks:
            return ConstraintResult.allow()

        # Check for explicit causal language using semantic matching
        causal_count = self.semantic_matcher.count_causal_chunks(chunks)

        if causal_count > 0:
            logger.debug(f"{PIPELINE} CausalAttributionConstraint: causal evidence found")
            return ConstraintResult.allow()

        # Causal query but no explicit causal language in evidence
        logger.info(
            f"{PIPELINE} CausalAttributionConstraint: causal query but no explicit "
            f"causal language in {len(chunks)} chunks"
        )

        return ConstraintResult.deny(
            reason="No explicit causal language found in sources",
            signal="qualified",  # Not abstain - we have evidence, just not causal
            causal_chunks=causal_count,
            total_chunks=len(chunks),
        )


__all__ = ["CausalAttributionConstraint"]
