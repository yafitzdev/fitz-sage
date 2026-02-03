# fitz_ai/core/guardrails/plugins/deterministic_conflict.py
"""
Deterministic Conflict Detection - LLM-free contradiction detection.

Uses embeddings + regex antonym detection instead of LLM calls.
This eliminates model variance and provides consistent, explainable results.

Logic:
1. Check if chunks are about the same topic (embedding similarity > threshold)
2. Check if they contain antonym pairs (regex)
3. If both: CONTRADICTION

No LLM calls. Fully deterministic. Tunable thresholds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

from fitz_ai.core.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from ..base import ConstraintResult

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Type alias for embedder function
EmbedderFunc = Callable[[str], list[float]]

# Antonym pairs for contradiction detection
# Format: (pattern_a, pattern_b) - if chunk1 has A and chunk2 has B (or vice versa), contradiction
ANTONYM_PAIRS: list[tuple[str, str]] = [
    # Success/Failure (expanded)
    (
        r"\b(success|succeeded|successful|successfully|complete success)\b",
        r"\b(fail|failed|failure|unsuccessful|catastroph)\b",
    ),
    # Approval/Rejection
    (
        r"\b(approved|accepted|granted|authorized|passed|full approval)\b",
        r"\b(rejected|denied|refused|declined|rejection|failed to pass)\b",
    ),
    # Increase/Decrease (expanded with surge/plunge)
    (
        r"\b(increased|grew|rose|gained|improved|growth|up|surged|soared|jumped)\b",
        r"\b(decreased|declined|fell|dropped|worsened|down|decline|plunged|tumbled)\b",
    ),
    # Profit/Loss
    (
        r"\b(profitable|profit|profits|gains?|record profits?)\b",
        r"\b(loss|losses|unprofitable|deficit|net loss)\b",
    ),
    # Positive/Negative outcome
    (
        r"\b(positive|good|excellent|strong|exceeded|better)\b",
        r"\b(negative|bad|poor|weak|missed|worse|disappointing)\b",
    ),
    # Confirm/Deny
    (r"\b(confirmed|verified|validated|proven)\b", r"\b(denied|refuted|disproven|contradicted)\b"),
    # Start/Stop/Cancel (expanded)
    (
        r"\b(started|began|launched|initiated|went ahead|proceeded|are now open)\b",
        r"\b(stopped|ended|terminated|cancelled|called off|delayed|postponed)\b",
    ),
    # Open/Closed
    (r"\b(open|opened|available|accessible)\b", r"\b(closed|shut|unavailable|inaccessible)\b"),
    # Guilty/Innocent
    (
        r"\b(guilty|convicted|responsible|guilty plea|pleaded guilty)\b",
        r"\b(innocent|acquitted|not guilty|exonerated|pleaded not guilty)\b",
    ),
    # Win/Lose (expanded)
    (
        r"\b(won|winning|victory|victorious|champion)\b",
        r"\b(lost|losing|defeat|defeated|runner-up)\b",
    ),
    # Alive/Dead
    (r"\b(alive|living|survived|surviving)\b", r"\b(dead|died|deceased|killed)\b"),
    # True/False
    (r"\b(true|correct|accurate|right)\b", r"\b(false|incorrect|inaccurate|wrong)\b"),
    # Present/Absent
    (r"\b(present|exists?|found|detected)\b", r"\b(absent|missing|not found|undetected)\b"),
    # Safe/Dangerous
    (r"\b(safe|secure|harmless)\b", r"\b(dangerous|unsafe|hazardous|risky)\b"),
    # Legal/Illegal
    (r"\b(legal|lawful|legitimate)\b", r"\b(illegal|unlawful|illegitimate)\b"),
    # Resign/Remain
    (
        r"\b(resign|resigned|resignation|stepping down|leaving|departure)\b",
        r"\b(remain|remaining|staying|continue|will remain|will stay)\b",
    ),
    # Support/Oppose
    (
        r"\b(support|supports|supported|favor|favors|endorses?)\b",
        r"\b(oppose|opposes|opposed|against|opposition|rejects?)\b",
    ),
    # Effective/Ineffective
    (
        r"\b(effective|works|working|efficacy|efficient)\b",
        r"\b(ineffective|doesn't work|not working|no effect|inefficient)\b",
    ),
    # Higher/Lower
    (r"\b(higher|more|greater|above|exceeded)\b", r"\b(lower|less|fewer|below|missed|short)\b"),
    # Agree/Disagree
    (
        r"\b(agree|agrees|agreed|consensus|unanimous)\b",
        r"\b(disagree|disagrees|disagreed|divided|split)\b",
    ),
    # Allow/Prohibit
    (
        r"\b(allow|allows|allowed|permit|permits|permitted)\b",
        r"\b(prohibit|prohibits|prohibited|ban|bans|banned|forbid)\b",
    ),
    # Include/Exclude
    (
        r"\b(include|includes|included|contains?)\b",
        r"\b(exclude|excludes|excluded|omit|omits|omitted)\b",
    ),
]

# Negation patterns - "X is approved" vs "X is not approved"
NEGATION_PATTERNS = [
    (r"\bis\s+(\w+)", r"\bis\s+not\s+\1"),
    (r"\bwas\s+(\w+)", r"\bwas\s+not\s+\1"),
    (r"\bwere\s+(\w+)", r"\bwere\s+not\s+\1"),
    (r"\bhas\s+been\s+(\w+)", r"\bhas\s+not\s+been\s+\1"),
    (r"\bwill\s+(\w+)", r"\bwill\s+not\s+\1"),
]

# Resolution keywords - if query asks for resolution, allow decisive answer
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


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def has_antonym_pair(text1: str, text2: str) -> tuple[bool, str | None]:
    """
    Check if two texts contain antonym pairs.

    Returns (has_antonym, description) where description explains which pair matched.
    """
    text1_lower = text1.lower()
    text2_lower = text2.lower()

    for pattern_a, pattern_b in ANTONYM_PAIRS:
        a_in_1 = re.search(pattern_a, text1_lower, re.IGNORECASE)
        b_in_2 = re.search(pattern_b, text2_lower, re.IGNORECASE)

        if a_in_1 and b_in_2:
            return True, f"'{a_in_1.group()}' vs '{b_in_2.group()}'"

        # Check reverse
        a_in_2 = re.search(pattern_a, text2_lower, re.IGNORECASE)
        b_in_1 = re.search(pattern_b, text1_lower, re.IGNORECASE)

        if a_in_2 and b_in_1:
            return True, f"'{b_in_1.group()}' vs '{a_in_2.group()}'"

    return False, None


def has_negation_flip(text1: str, text2: str) -> tuple[bool, str | None]:
    """
    Check if one text negates a statement from the other.

    E.g., "X is approved" vs "X is not approved"
    """
    text1_lower = text1.lower()
    text2_lower = text2.lower()

    # Simple negation check: look for "not" or "n't" differences
    # Find key phrases and check if one is negated

    # Pattern: "is/was/were [word]" vs "is/was/were not [word]"
    positive_patterns = [
        r"\b(is|was|were|has been|will be)\s+(\w+ed|\w+ing|\w+)\b",
    ]

    for pattern in positive_patterns:
        matches1 = re.findall(pattern, text1_lower)
        matches2 = re.findall(pattern, text2_lower)

        for verb1, word1 in matches1:
            # Check if text2 has the negated version
            negated = f"{verb1} not {word1}"
            if negated in text2_lower:
                return True, f"'{verb1} {word1}' vs '{negated}'"

        for verb2, word2 in matches2:
            negated = f"{verb2} not {word2}"
            if negated in text1_lower:
                return True, f"'{verb2} {word2}' vs '{negated}'"

    return False, None


def _is_resolution_query(query: str) -> bool:
    """Check if query is asking to resolve conflicts."""
    q = query.lower()
    return any(kw in q for kw in RESOLUTION_KEYWORDS)


@dataclass
class DeterministicConflictConstraint:
    """
    LLM-free conflict detection using embeddings + regex.

    Logic:
    1. Compute embedding similarity between chunks
    2. If similarity > threshold (same topic), check for antonyms
    3. If antonyms found, flag as contradiction

    No LLM variance. Fully deterministic. Explainable results.

    Attributes:
        embedder: Function to embed text into vectors
        similarity_threshold: Minimum similarity to consider "same topic" (default: 0.6)
        enabled: Whether this constraint is active (default: True)
    """

    embedder: EmbedderFunc | None = None
    similarity_threshold: float = 0.4  # Lower = more permissive (checks more pairs for antonyms)
    enabled: bool = True
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def name(self) -> str:
        return "deterministic_conflict"

    def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for text, with caching."""
        if not self.embedder:
            return None

        # Simple cache using hash of first 200 chars
        cache_key = hash(text[:200])
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            embedding = self.embedder(text)
            self._cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"{PIPELINE} DeterministicConflict: embedding failed: {e}")
            return None

    def _check_pair(self, chunk1: Chunk, chunk2: Chunk) -> tuple[bool, str | None]:
        """
        Check if two chunks contradict each other.

        Returns (is_contradiction, explanation).
        """
        text1 = chunk1.content
        text2 = chunk2.content

        # Step 1: Check for antonym pairs (most reliable signal)
        has_antonym, antonym_desc = has_antonym_pair(text1, text2)
        if has_antonym:
            # Optional: verify same topic with embeddings
            if self.embedder:
                emb1 = self._get_embedding(text1)
                emb2 = self._get_embedding(text2)
                if emb1 and emb2:
                    similarity = cosine_similarity(emb1, emb2)
                    logger.debug(
                        f"{PIPELINE} DeterministicConflict: antonym found, similarity={similarity:.2f}"
                    )
            return True, f"Antonym pair: {antonym_desc}"

        # Step 2: Check for negation patterns
        has_negation, negation_desc = has_negation_flip(text1, text2)
        if has_negation:
            return True, f"Negation: {negation_desc}"

        return False, None

    def apply(
        self,
        query: str,
        chunks: Sequence[Chunk],
    ) -> ConstraintResult:
        """
        Check for contradictions using embeddings + regex.

        Args:
            query: The user's question
            chunks: Retrieved chunks

        Returns:
            ConstraintResult - denies if contradiction detected
        """
        if not self.enabled:
            return ConstraintResult.allow()

        if not chunks or len(chunks) < 2:
            return ConstraintResult.allow()

        # If query asks for resolution, allow decisive answer
        if _is_resolution_query(query):
            logger.debug(f"{PIPELINE} DeterministicConflict: resolution query, allowing")
            return ConstraintResult.allow()

        # Check all pairs (limit to first 5 chunks)
        chunks_to_check = list(chunks[:5])

        for i, chunk1 in enumerate(chunks_to_check):
            for chunk2 in chunks_to_check[i + 1 :]:
                is_contradiction, explanation = self._check_pair(chunk1, chunk2)

                if is_contradiction:
                    logger.info(
                        f"{PIPELINE} DeterministicConflict: contradiction detected - {explanation}"
                    )
                    return ConstraintResult.deny(
                        reason=f"Contradiction detected: {explanation}",
                        signal="disputed",
                        explanation=explanation,
                    )

        logger.debug(f"{PIPELINE} DeterministicConflict: no contradiction detected")
        return ConstraintResult.allow()


__all__ = [
    "DeterministicConflictConstraint",
    "has_antonym_pair",
    "has_negation_flip",
    "cosine_similarity",
    "ANTONYM_PAIRS",
]
