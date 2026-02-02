# fitz_ai/core/guardrails/plugins/conflict_aware.py
"""
Conflict-Aware Constraint - Default guardrail for contradiction detection.

This constraint detects conflicting claims in retrieved chunks using
LLM-based semantic analysis. When conflicts exist, it prevents the
system from giving a confident answer.

Uses a fast LLM model for efficient conflict detection.

This constraint does NOT:
- Resolve conflicts
- Choose sides
- Apply authority hierarchies

It only prevents confident collapse when evidence disagrees.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from fitz_ai.core.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from ..base import ConstraintResult
from ..semantic import SemanticMatcher


def _extract_json(text: str) -> dict | None:
    """
    Extract JSON from LLM response, handling common issues.

    Handles:
    - Markdown code blocks (```json ... ```)
    - Extra text before/after JSON
    - Minor formatting issues
    """
    text = text.strip()

    # Remove markdown code blocks
    if "```" in text:
        # Try to extract content between code blocks
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()

    # Try to find JSON object in the text
    # Look for {...} pattern
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Direct parse attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return None

if TYPE_CHECKING:
    from fitz_ai.llm.providers.base import ChatProvider

logger = get_logger(__name__)


# =============================================================================
# Prompts
# =============================================================================

CONFLICT_DETECTION_PROMPT = """Do these chunks CONTRADICT each other? Look for:
- Success vs Failure
- Approved vs Rejected
- Increased vs Decreased
- Yes vs No
- Different numbers for same thing
- Opposite outcomes

Chunks:
{chunks}

If chunks contradict, respond:
{{"has_conflict": true, "conflicts": [{{"chunk_a": "FIRST_CHUNK_ID", "chunk_b": "SECOND_CHUNK_ID", "description": "brief reason"}}]}}

If NO contradiction:
{{"has_conflict": false, "conflicts": []}}

JSON only:"""


# =============================================================================
# Constraint Implementation
# =============================================================================


@dataclass
class ConflictAwareConstraint:
    """
    Constraint that detects conflicting claims using LLM analysis.

    When retrieved chunks contain mutually exclusive claims (e.g., one says
    "revenue increased" and another says "revenue decreased"), this constraint
    prevents the system from confidently asserting either.

    Uses a fast LLM model for efficient conflict detection.

    This does NOT resolve conflicts - it only prevents confident collapse.

    Attributes:
        semantic_matcher: SemanticMatcher for resolution query detection
        chat: Optional ChatProvider for LLM-based conflict detection
        enabled: Whether this constraint is active (default: True)
    """

    semantic_matcher: SemanticMatcher
    chat: "ChatProvider | None" = None
    enabled: bool = True
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def name(self) -> str:
        return "conflict_aware"

    def _detect_conflicts_llm(
        self, chunks: Sequence[Chunk]
    ) -> list[tuple[str, str, str]]:
        """
        Use LLM to detect conflicts between chunks.

        Returns:
            List of (chunk_a_id, chunk_b_id, conflict_description) tuples
        """
        if not self.chat:
            return []

        # Format chunks for the prompt
        chunk_texts = []
        for i, chunk in enumerate(chunks[:10]):  # Limit to 10 chunks for efficiency
            chunk_texts.append(f"[{chunk.id}]: {chunk.content[:500]}")  # Truncate long content

        chunks_str = "\n\n".join(chunk_texts)
        prompt = CONFLICT_DETECTION_PROMPT.format(chunks=chunks_str)

        try:
            response = self.chat.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic
                max_tokens=256,  # Ensure response isn't truncated
            )

            logger.debug(f"{PIPELINE} ConflictAwareConstraint: LLM response: {response[:200]}")

            # Parse JSON response with robust extraction
            result = _extract_json(response)

            if result is None:
                logger.warning(
                    f"{PIPELINE} ConflictAwareConstraint: could not extract JSON from: {response[:100]}"
                )
                return []

            logger.debug(f"{PIPELINE} ConflictAwareConstraint: parsed result: {result}")

            if not result.get("has_conflict", False):
                logger.debug(f"{PIPELINE} ConflictAwareConstraint: LLM found no conflict")
                return []

            conflicts = []
            for conflict in result.get("conflicts", [])[:3]:
                if isinstance(conflict, dict):
                    chunk_a = str(conflict.get("chunk_a", "unknown"))
                    chunk_b = str(conflict.get("chunk_b", "unknown"))
                    description = str(conflict.get("description", "conflicting claims"))
                    conflicts.append((chunk_a, chunk_b, description))

            logger.info(f"{PIPELINE} ConflictAwareConstraint: LLM detected {len(conflicts)} conflicts")
            return conflicts

        except Exception as e:
            logger.warning(f"{PIPELINE} ConflictAwareConstraint: LLM call failed: {e}")
            return []

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

        # Skip conflict detection if no LLM available
        if not self.chat:
            logger.warning(f"{PIPELINE} ConflictAwareConstraint: no chat provider, skipping conflict detection")
            return ConstraintResult.allow()

        # Need at least 2 chunks to have a conflict
        if len(chunks) < 2:
            return ConstraintResult.allow()

        # Detect conflicts using LLM
        conflicts = self._detect_conflicts_llm(chunks)

        if conflicts:
            # Format conflict description
            conflict_descriptions = []
            for chunk1_id, chunk2_id, conflict_type in conflicts[:3]:
                conflict_descriptions.append(f"[{chunk1_id}] vs [{chunk2_id}]: {conflict_type}")

            reason = f"Conflicting claims detected: {'; '.join(conflict_descriptions)}"
            logger.info(f"{PIPELINE} ConflictAwareConstraint: {reason}")

            return ConstraintResult.deny(
                reason=reason,
                signal="disputed",
                conflicts=conflicts,
                conflict_count=len(conflicts),
            )

        return ConstraintResult.allow()


__all__ = ["ConflictAwareConstraint"]
