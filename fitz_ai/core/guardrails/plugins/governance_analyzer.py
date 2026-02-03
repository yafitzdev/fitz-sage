# fitz_ai/core/guardrails/plugins/governance_analyzer.py
"""
Unified Governance Analyzer - LLM-based epistemic analysis.

This single constraint replaces the three embedding-based constraints:
- ConflictAwareConstraint (conflict detection)
- CausalAttributionConstraint (causal evidence detection)
- InsufficientEvidenceConstraint (relevance/evidence detection)

Why unified?
- Embedding similarity measures topical relatedness, not semantic nuance
- LLM can distinguish "Ford earnings" from "Tesla stock" (same domain, wrong entity)
- LLM can distinguish "Amazon rainforest" from "Amazon return policy" (keyword overlap)
- LLM can distinguish correlation from causation
- One LLM call is more efficient than 3 separate calls

The analyzer determines the appropriate governance mode:
- ABSTAIN: Context is irrelevant or insufficient
- DISPUTED: Context contains contradicting claims
- QUALIFIED: Context exists but evidence is insufficient (causal, correlation, small sample)
- CONFIDENT: Context is relevant and sufficient for a confident answer
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

if TYPE_CHECKING:
    from fitz_ai.llm.providers.base import ChatProvider

logger = get_logger(__name__)


def _extract_json(text: str) -> dict | None:
    """Extract JSON from LLM response, handling common issues."""
    text = text.strip()

    # Remove markdown code blocks
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()

    # Find all potential JSON objects (non-greedy approach)
    # This handles cases where the model outputs reasoning before JSON
    json_candidates = []
    brace_count = 0
    start_idx = None

    for i, char in enumerate(text):
        if char == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                json_candidates.append(text[start_idx : i + 1])
                start_idx = None

    # Try each candidate, preferring later ones (model often outputs JSON at end)
    for candidate in reversed(json_candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Direct parse attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return None


# =============================================================================
# Prompt
# =============================================================================

GOVERNANCE_ANALYSIS_PROMPT = """Output JSON only. No explanation. No reasoning.

{{"mode":"MODE","reason":"WHY"}}

MODE options:
- disputed: chunks contradict (yes vs no, success vs failure)
- abstain: wrong topic (query asks X, context discusses Y)
- qualified: asks "why" but context lacks causal explanation
- confident: context answers the query

Query: {query}

Context:
{context}

JSON:"""


# =============================================================================
# Constraint Implementation
# =============================================================================


@dataclass
class GovernanceAnalyzer:
    """
    Unified LLM-based governance analyzer.

    Replaces embedding-based constraints with a single LLM call that analyzes:
    - Relevance (is context about the right entity/time/aspect?)
    - Conflicts (do sources contradict?)
    - Evidence quality (causal evidence, sample size, completeness)

    Attributes:
        chat: ChatProvider for LLM-based analysis (required)
        enabled: Whether this constraint is active (default: True)
    """

    chat: "ChatProvider"
    enabled: bool = True
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def name(self) -> str:
        return "governance_analyzer"

    def _analyze(self, query: str, chunks: Sequence[Chunk]) -> dict[str, Any]:
        """
        Use LLM to analyze governance issues.

        Returns:
            Dict with analysis results
        """
        # Format context for prompt
        context_texts = []
        for i, chunk in enumerate(chunks[:8]):  # Limit to 8 chunks
            context_texts.append(f"[{i+1}]: {chunk.content[:400]}")

        context_str = "\n\n".join(context_texts)
        prompt = GOVERNANCE_ANALYSIS_PROMPT.format(query=query, context=context_str)

        try:
            response = self.chat.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )

            logger.debug(f"{PIPELINE} GovernanceAnalyzer: LLM response: {response[:300]}")

            result = _extract_json(response)

            if result is None:
                logger.warning(
                    f"{PIPELINE} GovernanceAnalyzer: could not extract JSON from: {response[:100]}"
                )
                # Default to allowing (fail-safe)
                return {"recommended_mode": "confident", "reason": "JSON parse error"}

            return result

        except Exception as e:
            logger.warning(f"{PIPELINE} GovernanceAnalyzer: LLM call failed: {e}")
            # Default to allowing (fail-safe)
            return {"recommended_mode": "confident", "reason": f"LLM error: {e}"}

    def apply(
        self,
        query: str,
        chunks: Sequence[Chunk],
    ) -> ConstraintResult:
        """
        Analyze query and chunks for governance issues.

        Args:
            query: The user's question
            chunks: Retrieved chunks

        Returns:
            ConstraintResult with appropriate signal
        """
        if not self.enabled:
            return ConstraintResult.allow()

        # Empty chunks = abstain
        if not chunks:
            logger.info(f"{PIPELINE} GovernanceAnalyzer: no chunks retrieved")
            return ConstraintResult.deny(
                reason="No evidence retrieved",
                signal="abstain",
            )

        # Analyze using LLM
        analysis = self._analyze(query, chunks)

        # Get mode from simplified response format
        mode = analysis.get("mode", "confident")
        reason = analysis.get("reason", "")

        # Map mode to constraint result
        if mode == "confident":
            logger.debug(f"{PIPELINE} GovernanceAnalyzer: allowing confident answer")
            return ConstraintResult.allow()

        elif mode == "abstain":
            logger.info(f"{PIPELINE} GovernanceAnalyzer: abstain - {reason}")
            return ConstraintResult.deny(
                reason=f"Context not relevant: {reason}",
                signal="abstain",
            )

        elif mode == "disputed":
            logger.info(f"{PIPELINE} GovernanceAnalyzer: disputed - {reason}")
            return ConstraintResult.deny(
                reason=f"Conflicting claims: {reason}",
                signal="disputed",
            )

        elif mode == "qualified":
            logger.info(f"{PIPELINE} GovernanceAnalyzer: qualified - {reason}")
            return ConstraintResult.deny(
                reason=f"Insufficient evidence: {reason}",
                signal="qualified",
            )

        # Unknown mode - default to allow
        logger.warning(f"{PIPELINE} GovernanceAnalyzer: unknown mode '{mode}'")
        return ConstraintResult.allow()


__all__ = ["GovernanceAnalyzer"]
