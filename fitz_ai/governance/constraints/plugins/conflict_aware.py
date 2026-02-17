# fitz_ai/governance/constraints/plugins/conflict_aware.py
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

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

from fitz_ai.governance.protocol import EvidenceItem
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from ..base import ConstraintResult, FeatureSpec
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

CONTRADICT = they make OPPOSITE or INCOMPATIBLE claims. This includes:
- Direct opposites (one says yes, the other says no)
- Competing explanations for the same thing
- Different conclusions from the same evidence
- Mutually exclusive claims (if one is true, the other cannot be)
AGREE = they are genuinely compatible and could both be true simultaneously.
UNCLEAR = cannot determine.

Reply with ONLY one word: CONTRADICT, AGREE, or UNCLEAR"""

# Fusion prompts - same question asked 3 different ways
# Helps reduce variance through majority voting
FUSION_PROMPTS = [
    # Direct contradiction check
    """Do these two texts CONTRADICT each other regarding the question?

Question: {query}
Text A: {text1}
Text B: {text2}

CONTRADICT = they make opposite or incompatible claims (including competing explanations, different conclusions from the same evidence, or mutually exclusive claims).
AGREE = they are genuinely compatible and could both be true simultaneously.
Reply with ONE word: CONTRADICT, AGREE, or UNCLEAR""",
    # Consistency framing (inverted)
    """Are these two texts CONSISTENT with each other regarding the question?

Question: {query}
First text: {text1}
Second text: {text2}

YES = they genuinely agree and could both be true simultaneously.
NO = they make opposite or incompatible claims (including competing explanations or mutually exclusive conclusions).
Reply with ONE word: YES, NO, or UNCLEAR""",
    # Logical compatibility framing
    """If the first statement is true, could the second statement also be true?

Question context: {query}
Statement 1: {text1}
Statement 2: {text2}

YES = both could genuinely be true at the same time.
NO = they are mutually exclusive or incompatible (including competing explanations for the same thing).
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

# Keywords that indicate uncertainty/causal/opinion queries (use conservative detection)
# These query types often have legitimately mixed evidence that should NOT trigger
# dispute mode. Routing them through fusion (3-prompt majority vote) reduces false
# disputes while preserving true contradiction detection for factual queries.
UNCERTAINTY_QUERY_PATTERNS = (
    # Causal/explanatory
    "why ",
    "why?",
    "what caused",
    "what led to",
    "how come",
    "what made",
    # Uncertainty language
    "might",
    "could",
    "possibly",
    "potentially",
    "likely",
    "unlikely",
    # Impact/effect queries (often have legitimately varied evidence)
    "what impact",
    "what effect",
    "how does",
    "how do ",
    # Prediction phrases (specific, not just "will")
    "will be in ",
    "predict",
    "forecast",
    # Subjective evaluation
    "is it worth",
    "do you think",
    "should we",
    "should i",
)


# ---------------------------------------------------------------------------
# Evidence character classification (deterministic, no LLM)
# ---------------------------------------------------------------------------
# Hedging markers: language indicating uncertainty, preliminary findings, tentative claims
_HEDGE_PATTERNS = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bsuggests?\b",
    r"\bpreliminary\b",
    r"\blimited evidence\b",
    r"\bmore research\b",
    r"\bcannot conclude\b",
    r"\binconclusive\b",
    r"\bassociated with\b",
    r"\bcorrelated\b",
    r"\bobservational\b",
    r"\btentative\b",
    r"\bappears? to\b",
    r"\bseems? to\b",
    r"\bpossibly\b",
    r"\bpotentially\b",
    r"\bunclear\b",
    r"\bnot (yet |fully )?established\b",
    r"\bearly\b.{0,20}\b(results?|findings?|data|studies?|evidence|trial)\b",
    r"\bwarrants? further\b",
    r"\bnot definitive\b",
    r"\bcannot\b.{0,20}\b(definitive|conclusive|confirm|conclude)\b",
    r"\brequires? (more|further|additional)\b",
    r"\blikely\b",
    r"\buncertain\b",
    r"\bevolve[ds]?\b",
]

# Assertive markers: language indicating firm claims, established facts, confirmed findings
_ASSERT_PATTERNS = [
    r"\bconfirmed?\b",
    r"\bproven\b",
    r"\bestablished\b",
    r"\bdefinitely\b",
    r"\bclearly\b",
    r"\bFDA approved\b",
    r"\bbanned\b",
    r"\brequires?\b",
    r"\bmust\b",
    r"\bcauses?\b",
    r"\bproves?\b",
    r"\b\d+(\.\d+)?%\b",  # Specific percentages
    r"\bn\s*=\s*\d+\b",  # Sample sizes
    r"\bp\s*[<>=]\s*0?\.\d+\b",  # p-values
]

_HEDGE_RE = [re.compile(p, re.IGNORECASE) for p in _HEDGE_PATTERNS]
_ASSERT_RE = [re.compile(p, re.IGNORECASE) for p in _ASSERT_PATTERNS]


def _classify_evidence_character(text: str) -> str:
    """
    Classify a chunk's evidence character as 'assertive', 'hedged', or 'mixed'.

    Uses weighted regex pattern matching. No LLM call.
    Default is 'assertive' — hedged classification requires positive evidence
    of uncertainty language. This prevents short factual statements from being
    classified as 'mixed' due to lack of explicit assertive markers.

    Returns:
        'assertive' - firm claims, established facts, or neutral factual statements
        'hedged' - preliminary, uncertain, tentative language dominates
        'mixed' - both assertive and hedged language present in significant amounts
    """
    hedge_count = sum(1 for pat in _HEDGE_RE if pat.search(text))
    assert_count = sum(1 for pat in _ASSERT_RE if pat.search(text))

    # Hedged: multiple hedge markers and few/no assertive markers
    if hedge_count >= 3 and assert_count <= 1:
        return "hedged"

    # Mixed: significant presence of both hedge and assertive language
    if hedge_count >= 2 and assert_count >= 2:
        return "mixed"

    # Default: assertive. Factual statements without hedge language are assertive
    # even if they lack explicit assertive markers (e.g., "Revenue was $5M").
    return "assertive"


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

    Relevance gate:
    - If embedder is provided, skips chunk pairs where either chunk has
      low similarity to the query. Prevents false disputes from irrelevant content.
    - Uses cosine similarity with configurable threshold (default: 0.3)

    Attributes:
        chat: ChatProvider for contradiction detection
        enabled: Whether this constraint is active (default: True)
        use_fusion: If True, always use 3-prompt fusion (default: False)
        adaptive: If True, select method based on query type (default: False)
        embedder: Optional embedding function for relevance gating
        relevance_threshold: Min query-chunk similarity to check for conflicts (default: 0.3)
    """

    chat: "ChatProvider | None" = None
    enabled: bool = True
    use_fusion: bool = False
    adaptive: bool = False
    embedder: Callable[[str], list[float]] | None = None
    relevance_threshold: float = 0.45
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _numerical_detector: NumericalConflictDetector = field(
        default_factory=NumericalConflictDetector, repr=False, compare=False
    )

    @property
    def name(self) -> str:
        return "conflict_aware"

    @staticmethod
    def feature_schema() -> list[FeatureSpec]:
        return [
            FeatureSpec("ca_fired", "bool", default=None),
            FeatureSpec("ca_signal", "categorical", default=None),
            FeatureSpec("ca_numerical_variance_detected", "bool", default=None),
            FeatureSpec("ca_pairs_checked", "float", default=0),
            FeatureSpec("ca_first_evidence_char", "categorical", default=None),
            FeatureSpec("ca_evidence_characters", "categorical", default=None),
            FeatureSpec("ca_is_uncertainty_query", "bool", default=None),
            FeatureSpec("ca_relevance_filtered_count", "float", default=0),
        ]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _is_chunk_relevant(self, query_embedding: list[float], chunk: EvidenceItem) -> bool:
        """Check if a chunk is relevant to the query via embedding similarity."""
        if not self.embedder:
            return True  # No embedder = assume relevant

        try:
            chunk_embedding = self.embedder(chunk.content[:500])
            sim = self._cosine_similarity(query_embedding, chunk_embedding)
            return sim >= self.relevance_threshold
        except Exception as e:
            logger.warning(f"{PIPELINE} ConflictAwareConstraint: relevance check failed: {e}")
            return True  # Fail open

    def _get_chunk_stance(self, query: str, chunk: EvidenceItem) -> str:
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

    def _check_pairwise_contradiction(
        self, query: str, chunk1: EvidenceItem, chunk2: EvidenceItem
    ) -> bool:
        """
        Check if two chunks contradict each other about the query.

        First checks for numerical variance (same direction, similar values).
        If variance detected, skips LLM call and returns False.

        Uses a single LLM call to compare both chunks together,
        which is more reliable than separate stance detection.
        """
        if not self.chat:
            return False

        text1 = chunk1.content[:800] if len(chunk1.content) > 800 else chunk1.content
        text2 = chunk2.content[:800] if len(chunk2.content) > 800 else chunk2.content

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

    def _check_pairwise_fusion(
        self, query: str, chunk1: EvidenceItem, chunk2: EvidenceItem
    ) -> bool:
        """
        Check for contradiction using 3-prompt fusion with majority voting.

        First checks for numerical variance (same direction, similar values).
        If variance detected, skips LLM calls and returns False.

        Asks the same question 3 different ways:
        1. Direct: "Do these CONTRADICT?"
        2. Inverted: "Are these CONSISTENT?" (NO = contradict)
        3. Logical: "If A is true, can B be true?" (NO = contradict)

        Returns True if 2+ prompts indicate contradiction (majority vote).
        This reduces variance by requiring consensus across differently-framed prompts.
        """
        if not self.chat:
            return False

        text1 = chunk1.content[:800] if len(chunk1.content) > 800 else chunk1.content
        text2 = chunk2.content[:800] if len(chunk2.content) > 800 else chunk2.content

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
        chunks: Sequence[EvidenceItem],
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

        # Diagnostics for classifier feature extraction
        ca_diag = {
            "ca_numerical_variance_detected": False,
            "ca_is_uncertainty_query": False,
            "ca_skipped_hedged_pairs": 0,
            "ca_relevance_filtered_count": 0,
            "ca_pairs_checked": 0,
            "ca_first_evidence_char": None,
        }

        # Skip conflict detection if no LLM available
        if not self.chat:
            logger.debug(
                f"{PIPELINE} ConflictAwareConstraint: no chat provider, skipping conflict detection"
            )
            return ConstraintResult.allow(**ca_diag)

        # Relevance gate: filter out chunks not relevant to the query
        # This prevents false disputes from irrelevant content pairs
        chunks_to_check = list(chunks[:5])
        if self.embedder:
            try:
                query_embedding = self.embedder(query)
                relevant_chunks = [
                    c for c in chunks_to_check if self._is_chunk_relevant(query_embedding, c)
                ]
                filtered = len(chunks_to_check) - len(relevant_chunks)
                ca_diag["ca_relevance_filtered_count"] = filtered
                if filtered > 0:
                    logger.debug(
                        f"{PIPELINE} ConflictAwareConstraint: relevance gate filtered "
                        f"{filtered}/{len(chunks_to_check)} chunks"
                    )
                chunks_to_check = relevant_chunks
            except Exception as e:
                logger.warning(f"{PIPELINE} ConflictAwareConstraint: relevance gate failed: {e}")

        if len(chunks_to_check) < 2:
            return ConstraintResult.allow(**ca_diag)

        first_chunk = chunks_to_check[0]

        # Classify evidence character for the first chunk (reused across pairs)
        first_char = _classify_evidence_character(first_chunk.content)
        ca_diag["ca_first_evidence_char"] = first_char

        # Choose base detection method based on settings
        if self.adaptive:
            use_fusion_for_query = _is_uncertainty_query(query)
            ca_diag["ca_is_uncertainty_query"] = use_fusion_for_query
            base_method = (
                self._check_pairwise_fusion
                if use_fusion_for_query
                else self._check_pairwise_contradiction
            )
            base_method_name = "adaptive-fusion" if use_fusion_for_query else "adaptive-pairwise"
            logger.debug(
                f"{PIPELINE} ConflictAwareConstraint: adaptive mode selected {base_method_name} "
                f"(uncertainty_query={use_fusion_for_query})"
            )
        else:
            base_method = (
                self._check_pairwise_fusion
                if self.use_fusion
                else self._check_pairwise_contradiction
            )
            base_method_name = "fusion" if self.use_fusion else "pairwise"

        evidence_chars_seen = []

        for other_chunk in chunks_to_check[1:]:
            other_char = _classify_evidence_character(other_chunk.content)
            evidence_chars_seen.append(f"{first_char}_vs_{other_char}")

            # Pair-level evidence character: combine both chunks' text
            # to catch hedging distributed across chunks (e.g., "may" in
            # chunk A + "preliminary" in chunk B = hedged pair)
            pair_char = _classify_evidence_character(
                first_chunk.content + " " + other_chunk.content
            )

            # Evidence character gating (pair-conditioned truth table):
            # hedged vs hedged OR pair-level hedged → skip
            if (first_char == "hedged" and other_char == "hedged") or pair_char == "hedged":
                ca_diag["ca_skipped_hedged_pairs"] += 1
                logger.debug(
                    f"{PIPELINE} ConflictAwareConstraint: skipping hedged pair "
                    f"({first_char} vs {other_char}, pair={pair_char})"
                )
                continue

            ca_diag["ca_pairs_checked"] += 1

            # Track numerical variance (checked again inside pairwise methods, but
            # tracked here for classifier feature extraction)
            text1 = (
                first_chunk.content[:800] if len(first_chunk.content) > 800 else first_chunk.content
            )
            text2 = (
                other_chunk.content[:800] if len(other_chunk.content) > 800 else other_chunk.content
            )
            is_var, _ = self._numerical_detector.check_chunk_pair_variance(text1, text2)
            if is_var:
                ca_diag["ca_numerical_variance_detected"] = True

            # assertive vs assertive → standard method (likely real contradictions)
            # any hedged/mixed involved → use fusion for higher bar
            if first_char == "assertive" and other_char == "assertive" and pair_char == "assertive":
                check_method = base_method
                method_name = base_method_name
            else:
                check_method = self._check_pairwise_fusion
                method_name = f"{base_method_name}+evidence-gate-fusion"

            if check_method(query, first_chunk, other_chunk):
                logger.info(
                    f"{PIPELINE} ConflictAwareConstraint ({method_name}): contradiction detected "
                    f"between chunks (chars: {first_char} vs {other_char})"
                )
                return ConstraintResult.deny(
                    reason="Retrieved chunks contain contradictory information",
                    signal="disputed",
                    method=method_name,
                    ca_evidence_characters=f"{first_char}_vs_{other_char}",
                    **ca_diag,
                )

        # Expose evidence characters even on allow path for classifier features
        ca_diag["ca_evidence_characters"] = (
            evidence_chars_seen[0] if evidence_chars_seen else "none"
        )

        logger.debug(f"{PIPELINE} ConflictAwareConstraint: no contradiction detected")
        return ConstraintResult.allow(**ca_diag)


__all__ = ["ConflictAwareConstraint"]
