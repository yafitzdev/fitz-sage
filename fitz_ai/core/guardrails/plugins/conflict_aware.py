# fitz_ai/core/guardrails/plugins/conflict_aware.py
"""
Conflict-Aware Constraint - Default guardrail for contradiction detection.

This constraint detects explicitly conflicting claims in retrieved chunks.
When conflicts exist, it prevents the system from giving a confident answer.

Uses core/conflicts.py for conflict detection logic, ensuring consistency
across the platform (query-time and ingest-time).

This constraint does NOT:
- Resolve conflicts
- Choose sides
- Apply authority hierarchies

It only prevents confident collapse when evidence disagrees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from fitz_ai.core.conflicts import find_conflicts
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from ..base import ConstraintResult

if TYPE_CHECKING:
    from fitz_ai.core.conflicts import ChunkLike

logger = get_logger(__name__)


def _is_resolution_query(query: str) -> bool:
    """
    Detect if query explicitly asks for conflict resolution.

    Queries like "Which classification should be considered authoritative?"
    should be allowed to give decisive answers even with conflicts.
    """
    resolution_patterns = (
        r"\bwhich\b.*\bauthoritative\b",
        r"\bwhich\b.*\bcorrect\b",
        r"\bwhich\b.*\btrust\b",
        r"\bwhich\b.*\bbelieve\b",
        r"\bresolve\b.*\bconflict",
        r"\breconcile\b",
        r"\bwhy\s+(?:do|does|are)\b.*\bdisagree\b",
        r"\bwhy\s+(?:the)?\s*difference\b",
    )

    query_lower = query.lower()
    for pattern in resolution_patterns:
        if re.search(pattern, query_lower):
            return True

    return False


# =============================================================================
# Constraint Implementation
# =============================================================================


@dataclass
class ConflictAwareConstraint:
    """
    Default constraint that detects conflicting claims.

    When retrieved chunks contain mutually exclusive claims (e.g., one says
    "security incident" and another says "operational incident", or one says
    "improved" and another says "declined"), this constraint prevents the
    system from confidently asserting either.

    This does NOT resolve conflicts - it only prevents confident collapse.

    Attributes:
        enabled: Whether this constraint is active (default: True)
    """

    enabled: bool = True

    @property
    def name(self) -> str:
        return "conflict_aware"

    def apply(
        self,
        query: str,
        chunks: Sequence["ChunkLike"],
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
        if _is_resolution_query(query):
            logger.debug(f"{PIPELINE} ConflictAwareConstraint: resolution query detected, allowing")
            return ConstraintResult.allow()

        # Detect conflicts using core logic
        conflicts = find_conflicts(chunks)

        if not conflicts:
            return ConstraintResult.allow()

        # Format conflict description
        conflict_descriptions = []
        for chunk1_id, class1, chunk2_id, class2 in conflicts[:3]:  # Limit to 3
            conflict_descriptions.append(f"'{class1}' vs '{class2}'")

        reason = f"Conflicting claims detected: {', '.join(conflict_descriptions)}"

        logger.info(f"{PIPELINE} ConflictAwareConstraint: {reason}")

        return ConstraintResult.deny(
            reason=reason,
            signal="disputed",
            conflicts=conflicts,
            conflict_count=len(conflicts),
        )


__all__ = ["ConflictAwareConstraint"]
