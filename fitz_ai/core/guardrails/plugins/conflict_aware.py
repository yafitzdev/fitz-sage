# fitz_ai/core/guardrails/plugins/conflict_aware.py
"""
Conflict-Aware Constraint - Default guardrail for contradiction detection.

This constraint detects conflicting claims in retrieved chunks using
semantic embedding similarity. When conflicts exist, it prevents the
system from giving a confident answer.

Uses semantic matching for language-agnostic conflict detection.

This constraint does NOT:
- Resolve conflicts
- Choose sides
- Apply authority hierarchies

It only prevents confident collapse when evidence disagrees.
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
class ConflictAwareConstraint:
    """
    Constraint that detects conflicting claims using semantic similarity.

    When retrieved chunks contain mutually exclusive claims (e.g., one says
    "security incident" and another says "operational incident", or one says
    "improved" and another says "declined"), this constraint prevents the
    system from confidently asserting either.

    This constraint is language-agnostic - it works across any language
    supported by the embedding model.

    This does NOT resolve conflicts - it only prevents confident collapse.

    Attributes:
        semantic_matcher: SemanticMatcher instance for embedding-based detection
        enabled: Whether this constraint is active (default: True)
    """

    semantic_matcher: SemanticMatcher
    enabled: bool = True

    @property
    def name(self) -> str:
        return "conflict_aware"

    def apply(
        self,
        query: str,
        chunks: Sequence[Chunk],
    ) -> ConstraintResult:
        """
        Check for conflicting claims in retrieved chunks.

        Args:
            query: The user's question
            chunks: Retrieved chunks

        Returns:
            ConstraintResult - denies decisive answer if conflicts detected
        """
        if not self.enabled:
            return ConstraintResult.allow()

        if not chunks:
            return ConstraintResult.allow()

        # If query explicitly asks for resolution, allow decisive answer
        if self.semantic_matcher.is_resolution_query(query):
            logger.debug(f"{PIPELINE} ConflictAwareConstraint: resolution query detected, allowing")
            return ConstraintResult.allow()

        # Detect conflicts using semantic matching
        conflicts = self.semantic_matcher.find_conflicts(chunks)

        if not conflicts:
            return ConstraintResult.allow()

        # Format conflict description
        conflict_descriptions = []
        for chunk1_id, chunk2_id, conflict_type in conflicts[:3]:  # Limit to 3
            conflict_descriptions.append(f"[{chunk1_id}] vs [{chunk2_id}]: {conflict_type}")

        reason = f"Conflicting claims detected: {'; '.join(conflict_descriptions)}"

        logger.info(f"{PIPELINE} ConflictAwareConstraint: {reason}")

        return ConstraintResult.deny(
            reason=reason,
            signal="disputed",
            conflicts=conflicts,
            conflict_count=len(conflicts),
        )


__all__ = ["ConflictAwareConstraint"]
