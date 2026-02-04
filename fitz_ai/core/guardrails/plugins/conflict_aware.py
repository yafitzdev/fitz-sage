# fitz_ai/core/guardrails/plugins/conflict_aware.py
"""
Conflict-Aware Constraint - Default guardrail for contradiction detection.

This constraint detects conflicting claims in retrieved chunks using
simple LLM-based YES/NO classification per chunk.

The key insight: contradiction detection is the ONE thing keywords can't do.
A chunk saying "revenue increased" and one saying "revenue decreased" have
similar keywords but opposite stances.

This uses a simple per-chunk stance classification:
1. For each chunk: "Does this answer the query with YES or NO?"
2. If some chunks say YES and others say NO → contradiction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from fitz_ai.core.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from ..base import ConstraintResult
from ..numerical_detector import NumericalConflictDetector

if TYPE_CHECKING:
    from fitz_ai.llm.providers.base import ChatProvider

logger = get_logger(__name__)


STANCE_PROMPT = """Does this text answer the question with YES or NO?
If unclear or not applicable, say UNCLEAR.

Question: {query}
Text: {text}

Reply with ONLY one word: YES, NO, or UNCLEAR"""

# Single-call contradiction detection prompt
CONTRADICTION_PROMPT = """Do these texts CONTRADICT each other about the question?

Question: {query}

Text 1: {text1}

Text 2: {text2}

If they say OPPOSITE things (one says yes, one says no), answer CONTRADICT.
If they agree or are compatible, answer AGREE.
If unclear, answer UNCLEAR.

Reply with ONLY one word: CONTRADICT, AGREE, or UNCLEAR"""

# Fusion prompts - same question asked 3 different ways
# Helps reduce variance through majority voting
FUSION_PROMPTS = [
    # Direct contradiction check
    """Do these two texts CONTRADICT each other regarding the question?

Question: {query}
Text A: {text1}
Text B: {text2}

Answer CONTRADICT if they make opposite claims, AGREE if compatible, UNCLEAR if uncertain.
Reply with ONE word: CONTRADICT, AGREE, or UNCLEAR""",
    # Consistency framing (inverted)
    """Are these two texts CONSISTENT with each other regarding the question?

Question: {query}
First text: {text1}
Second text: {text2}

Answer YES if they agree or are compatible, NO if they contradict, UNCLEAR if uncertain.
Reply with ONE word: YES, NO, or UNCLEAR""",
    # Logical compatibility framing
    """If the first statement is true, could the second statement also be true?

Question context: {query}
Statement 1: {text1}
Statement 2: {text2}

Answer YES if both could be true, NO if they are mutually exclusive, UNCLEAR if uncertain.
Reply with ONE word: YES, NO, or UNCLEAR""",
]


# Keywords that indicate user is asking for conflict resolution
RESOLUTION_KEYWORDS = (
    "authoritative",
    "which source",
    "trust",
    "resolve",
    "reconcile",
    "correct version",
    "which is right",
    "which is correct",
)

# Keywords that indicate uncertainty/causal queries (use conservative detection)
UNCERTAINTY_QUERY_PATTERNS = (
    "why ",
    "why?",
    "what caused",
    "what led to",
    "how come",
    "what made",
    "might",
    "could",
    "possibly",
    "potentially",
    "likely",
    "unlikely",
)


def _is_resolution_query(query: str) -> bool:
    """Check if query is asking to resolve conflicts (keyword-based)."""
    q = query.lower()
    return any(kw in q for kw in RESOLUTION_KEYWORDS)


def _is_uncertainty_query(query: str) -> bool:
    """Check if query involves uncertainty/causality (use conservative detection)."""
    q = query.lower().strip()
    return any(pattern in q for pattern in UNCERTAINTY_QUERY_PATTERNS)


@dataclass
class ConflictAwareConstraint:
    """
    Constraint that detects conflicting claims using pairwise LLM comparison.

    When retrieved chunks contain mutually exclusive claims (e.g., one says
    "revenue increased" and another says "revenue decreased"), this constraint
    prevents the system from confidently asserting either.

    Three modes:
    - Standard: Single pairwise comparison prompt (aggressive, high recall)
    - Fusion: Ask 3 different ways, majority vote (conservative, high precision)
    - Adaptive: Auto-select based on query type:
        - Uncertainty/causal queries → Fusion (fewer false disputes)
        - Factual queries → Standard (catch more contradictions)

    Numerical variance filtering:
    - Before LLM check, extracts numeric mentions from both chunks
    - If same unit + same direction + ≤25% relative difference → variance (not contradiction)
    - Prevents "10% growth" vs "12% growth" from being flagged as dispute

    Attributes:
        chat: ChatProvider for contradiction detection
        enabled: Whether this constraint is active (default: True)
        use_fusion: If True, always use 3-prompt fusion (default: False)
        adaptive: If True, select method based on query type (default: False)
    """

    chat: "ChatProvider | None" = None
    enabled: bool = True
    use_fusion: bool = False
    adaptive: bool = False
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _numerical_detector: NumericalConflictDetector = field(
        default_factory=NumericalConflictDetector, repr=False, compare=False
    )

    @property
    def name(self) -> str:
        return "conflict_aware"

    def _get_chunk_stance(self, query: str, chunk: Chunk) -> str:
        """
        Get chunk's stance on the query: YES, NO, or UNCLEAR.

        Uses a simple single-word response format that fast models handle well.
        Uses raw content (not summary) because contradiction detection needs
        the full detail - summaries often lose the nuance needed for YES/NO.
        """
        if not self.chat:
            return "UNCLEAR"

        # Use raw content - summaries lose nuance needed for stance detection
        text = chunk.content[:500] if len(chunk.content) > 500 else chunk.content

        prompt = STANCE_PROMPT.format(query=query, text=text)

        try:
            response = self.chat.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )

            # Parse single-word response
            word = response.strip().upper()

            if "YES" in word:
                return "YES"
            elif "NO" in word:
                return "NO"
            return "UNCLEAR"

        except Exception as e:
            logger.warning(f"{PIPELINE} ConflictAwareConstraint: stance check failed: {e}")
            return "UNCLEAR"

    def _check_pairwise_contradiction(self, query: str, chunk1: Chunk, chunk2: Chunk) -> bool:
        """
        Check if two chunks contradict each other about the query.

        First checks for numerical variance (same direction, similar values).
        If variance detected, skips LLM call and returns False.

        Uses a single LLM call to compare both chunks together,
        which is more reliable than separate stance detection.
        """
        if not self.chat:
            return False

        text1 = chunk1.content[:400] if len(chunk1.content) > 400 else chunk1.content
        text2 = chunk2.content[:400] if len(chunk2.content) > 400 else chunk2.content

        # Check for numerical variance BEFORE LLM call
        is_variance, reason = self._numerical_detector.check_chunk_pair_variance(text1, text2)
        if is_variance:
            logger.debug(f"{PIPELINE} ConflictAwareConstraint: skipping LLM check - {reason}")
            return False  # Variance, not contradiction

        prompt = CONTRADICTION_PROMPT.format(query=query, text1=text1, text2=text2)

        try:
            response = self.chat.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )

            word = response.strip().upper()
            return "CONTRADICT" in word

        except Exception as e:
            logger.warning(f"{PIPELINE} ConflictAwareConstraint: contradiction check failed: {e}")
            return False

    def _check_pairwise_fusion(self, query: str, chunk1: Chunk, chunk2: Chunk) -> bool:
        """
        Check for contradiction using 3-prompt fusion with majority voting.

        First checks for numerical variance (same direction, similar values).
        If variance detected, skips LLM calls and returns False.

        Asks the same question 3 different ways:
        1. Direct: "Do these CONTRADICT?"
        2. Inverted: "Are these CONSISTENT?" (NO = contradict)
        3. Logical: "If A is true, can B be true?" (NO = contradict)

        Returns True if 2+ prompts indicate contradiction.
        This reduces variance by requiring consensus.
        """
        if not self.chat:
            return False

        text1 = chunk1.content[:400] if len(chunk1.content) > 400 else chunk1.content
        text2 = chunk2.content[:400] if len(chunk2.content) > 400 else chunk2.content

        # Check for numerical variance BEFORE LLM calls
        is_variance, reason = self._numerical_detector.check_chunk_pair_variance(text1, text2)
        if is_variance:
            logger.debug(f"{PIPELINE} ConflictAwareConstraint: skipping fusion check - {reason}")
            return False  # Variance, not contradiction

        contradict_votes = 0
        responses = []

        for i, prompt_template in enumerate(FUSION_PROMPTS):
            prompt = prompt_template.format(query=query, text1=text1, text2=text2)

            try:
                response = self.chat.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
                word = response.strip().upper()
                responses.append(word)

                # Interpret response based on prompt framing
                if i == 0:  # Direct: CONTRADICT means contradiction
                    if "CONTRADICT" in word:
                        contradict_votes += 1
                elif i == 1:  # Inverted: NO means contradiction (not consistent)
                    if word.startswith("NO"):
                        contradict_votes += 1
                elif i == 2:  # Logical: NO means contradiction (mutually exclusive)
                    if word.startswith("NO"):
                        contradict_votes += 1

            except Exception as e:
                logger.warning(f"{PIPELINE} ConflictAwareConstraint: fusion prompt {i} failed: {e}")
                responses.append("ERROR")

        logger.debug(
            f"{PIPELINE} ConflictAwareConstraint fusion: votes={contradict_votes}/3, responses={responses}"
        )

        # Majority vote: 2+ must indicate contradiction
        return contradict_votes >= 2

    def apply(
        self,
        query: str,
        chunks: Sequence[Chunk],
    ) -> ConstraintResult:
        """
        Check for conflicting claims in retrieved chunks.

        Uses pairwise contradiction detection - comparing chunks together
        rather than classifying each independently.

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

        # Need at least 2 chunks to have a conflict
        if len(chunks) < 2:
            return ConstraintResult.allow()

        # If query explicitly asks for resolution, allow decisive answer
        if _is_resolution_query(query):
            logger.debug(f"{PIPELINE} ConflictAwareConstraint: resolution query detected, allowing")
            return ConstraintResult.allow()

        # Skip conflict detection if no LLM available
        if not self.chat:
            logger.debug(
                f"{PIPELINE} ConflictAwareConstraint: no chat provider, skipping conflict detection"
            )
            return ConstraintResult.allow()

        # Use pairwise contradiction detection (more reliable than per-chunk stance)
        # Check first chunk against all others (limit to 4 comparisons)
        chunks_to_check = list(chunks[:5])
        first_chunk = chunks_to_check[0]

        # Choose detection method based on settings
        if self.adaptive:
            # Adaptive mode: use fusion for uncertainty queries, standard for factual
            use_fusion_for_query = _is_uncertainty_query(query)
            check_method = (
                self._check_pairwise_fusion
                if use_fusion_for_query
                else self._check_pairwise_contradiction
            )
            method_name = "adaptive-fusion" if use_fusion_for_query else "adaptive-pairwise"
            logger.debug(
                f"{PIPELINE} ConflictAwareConstraint: adaptive mode selected {method_name} "
                f"(uncertainty_query={use_fusion_for_query})"
            )
        else:
            # Fixed mode: use fusion if explicitly enabled
            check_method = (
                self._check_pairwise_fusion
                if self.use_fusion
                else self._check_pairwise_contradiction
            )
            method_name = "fusion" if self.use_fusion else "pairwise"

        for other_chunk in chunks_to_check[1:]:
            if check_method(query, first_chunk, other_chunk):
                logger.info(
                    f"{PIPELINE} ConflictAwareConstraint ({method_name}): contradiction detected "
                    f"between chunks"
                )
                return ConstraintResult.deny(
                    reason="Retrieved chunks contain contradictory information",
                    signal="disputed",
                    method=method_name,
                )

        logger.debug(f"{PIPELINE} ConflictAwareConstraint: no contradiction detected")
        return ConstraintResult.allow()


__all__ = ["ConflictAwareConstraint"]
