# fitz_ai/governance/constraints/plugins/answer_verification.py
"""
Answer Verification Constraint - Citation-grounded verification.

Instead of asking "is this relevant?", asks "cite the passage that answers
this question." Verifies the citation with fuzzy string matching. Produces
features for the ML classifier AND acts as a post-classifier veto.

Flow:
  1. For each chunk (up to 3), ask the LLM to quote the exact passage
  2. Verify the quote via substring match or SequenceMatcher
  3. If any chunk produces a valid citation → evidence answers the question
  4. If no citation found → fire "qualified" signal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Sequence

from fitz_ai.governance.protocol import EvidenceItem
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from ..base import ConstraintResult, FeatureSpec

if TYPE_CHECKING:
    from fitz_ai.llm.providers.base import ChatProvider

logger = get_logger(__name__)

# Similarity threshold for fuzzy citation matching
_STRONG_MATCH_THRESHOLD = 0.9
_WEAK_MATCH_THRESHOLD = 0.8

CITATION_PROMPT = """Given this question and context, quote the EXACT passage (verbatim, word-for-word) from the context that answers the question. If no passage answers it, say NONE.

Question: {query}
Context: {context}

Quote:"""


def _score_citation(citation: str, context: str) -> float:
    """Score a citation against the source context.

    Returns:
        Float 0-1 where 1.0 = exact substring, >0.8 = fuzzy match, 0 = no match.
    """
    if not citation or citation.strip().upper() == "NONE":
        return 0.0

    citation_clean = citation.strip().strip('"').strip("'").strip()
    if not citation_clean:
        return 0.0

    context_lower = context.lower()
    citation_lower = citation_clean.lower()

    # Exact substring match
    if citation_lower in context_lower:
        return 1.0

    # Fuzzy match via SequenceMatcher
    ratio = SequenceMatcher(None, citation_lower, context_lower).ratio()

    # The citation is typically much shorter than context, so use a
    # find_longest_match approach for partial overlap
    matcher = SequenceMatcher(None, citation_lower, context_lower)
    match = matcher.find_longest_match(0, len(citation_lower), 0, len(context_lower))
    if match.size > 0:
        # Coverage: how much of the citation is covered by the best match
        coverage = match.size / len(citation_lower)
        # Use the better of full ratio or coverage-based score
        ratio = max(ratio, coverage)

    return ratio


def _citations_contradict(citation_a: str, citation_b: str) -> bool:
    """Check if two citations contradict each other using simple heuristics.

    Two citations contradict if they are from different chunks and have
    low textual similarity (different passages about the same question).
    """
    if not citation_a or not citation_b:
        return False

    a_lower = citation_a.strip().lower()
    b_lower = citation_b.strip().lower()

    # If citations are very similar, they agree
    similarity = SequenceMatcher(None, a_lower, b_lower).ratio()
    if similarity > 0.7:
        return False

    # Both are non-trivial citations but say different things → contradiction
    if len(a_lower) > 20 and len(b_lower) > 20:
        return True

    return False


@dataclass
class AnswerVerificationConstraint:
    """
    Verifies chunks actually answer the query using citation-grounded verification.

    Asks the LLM to quote the exact passage that answers the question, then
    verifies the quote with string matching. Produces rich features for the
    ML classifier.

    Attributes:
        chat: ChatProvider for citation extraction
        enabled: Whether this constraint is active (default: True)
    """

    name: str = "answer_verification"
    chat: "ChatProvider | None" = None
    enabled: bool = True
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @staticmethod
    def feature_schema() -> list[FeatureSpec]:
        return [
            FeatureSpec("av_fired", "bool", default=None),
            FeatureSpec("av_citation_found", "bool", default=None),
            FeatureSpec("av_citation_quality", "float", default=None),
            FeatureSpec("av_citations_count", "float", default=None),
            FeatureSpec("av_contradicting_citations", "bool", default=False),
        ]

    def _extract_citation(self, query: str, context: str) -> tuple[str, float]:
        """Ask LLM to cite the passage that answers the query, then verify.

        Returns:
            Tuple of (citation_text, match_score) where match_score is 0-1.
        """
        if not self.chat:
            return "", 0.0

        prompt = CITATION_PROMPT.format(query=query, context=context)

        try:
            response = self.chat.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            citation = response.strip()
            score = _score_citation(citation, context)
            return citation, score
        except Exception as e:
            logger.warning(f"{PIPELINE} AnswerVerificationConstraint: citation extraction failed: {e}")
            return "", 0.0

    def apply(self, query: str, chunks: Sequence[EvidenceItem]) -> ConstraintResult:
        """
        Check if chunks answer the query using citation-grounded verification.

        Flow:
          1. For each chunk (up to 3), ask LLM to quote the answering passage
          2. Verify quote with string matching
          3. If any valid citation found → allow (early exit on strong match)
          4. If no citation found → fire "qualified" signal

        Args:
            query: User query
            chunks: Retrieved chunks

        Returns:
            ConstraintResult with citation features (always emitted)
        """
        if not self.enabled:
            return ConstraintResult.allow()

        if not chunks:
            return ConstraintResult.allow()

        if not self.chat:
            logger.debug(f"{PIPELINE} AnswerVerificationConstraint: no chat provider, skipping")
            return ConstraintResult.allow()

        citations: list[tuple[str, float]] = []  # (citation_text, score)

        for idx, chunk in enumerate(chunks[:3]):
            citation, score = self._extract_citation(query, chunk.content)
            citations.append((citation, score))

            logger.debug(
                f"{PIPELINE} AnswerVerificationConstraint citation (chunk {idx + 1}): "
                f"score={score:.3f}, citation={citation[:80]!r}"
            )

            # Early exit on strong match — evidence clearly answers the question
            if score >= _STRONG_MATCH_THRESHOLD:
                return ConstraintResult.allow(
                    av_citation_found=True,
                    av_citation_quality=score,
                    av_citations_count=1,
                    av_contradicting_citations=False,
                )

        # Evaluate all citations
        valid_citations = [(text, score) for text, score in citations if score >= _WEAK_MATCH_THRESHOLD]
        best_score = max((s for _, s in citations), default=0.0)
        citation_found = len(valid_citations) > 0

        # Check for contradictions between valid citations
        has_contradiction = False
        if len(valid_citations) >= 2:
            for i in range(len(valid_citations)):
                for j in range(i + 1, len(valid_citations)):
                    if _citations_contradict(valid_citations[i][0], valid_citations[j][0]):
                        has_contradiction = True
                        break
                if has_contradiction:
                    break

        if citation_found:
            logger.debug(
                f"{PIPELINE} AnswerVerificationConstraint: {len(valid_citations)} valid citation(s), "
                f"best_score={best_score:.3f}, contradiction={has_contradiction}"
            )
            return ConstraintResult.allow(
                av_citation_found=True,
                av_citation_quality=best_score,
                av_citations_count=len(valid_citations),
                av_contradicting_citations=has_contradiction,
            )
        else:
            logger.info(
                f"{PIPELINE} AnswerVerificationConstraint: no valid citations from "
                f"{len(citations)} chunk(s), best_score={best_score:.3f} -> firing"
            )
            return ConstraintResult.deny(
                reason="Retrieved content may not directly answer the question",
                signal="qualified",
                av_citation_found=False,
                av_citation_quality=best_score,
                av_citations_count=0,
                av_contradicting_citations=False,
            )


__all__ = ["AnswerVerificationConstraint"]
