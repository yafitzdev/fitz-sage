# fitz_ai/governance/constraints/plugins/answer_verification.py
"""
Answer Verification Constraint - LLM jury with balanced-tier confirmation.

Uses 3 fast-tier prompts to screen for irrelevant context, then confirms
with a single balanced-tier call when 2+/3 fast prompts agree.

Flow:
  1. Run 3 fast-tier prompts (lenient relevance checks)
  2. If 2+/3 say NO (context not relevant) -> run 1 balanced-tier confirmation
  3. If balanced also says NO -> fire "qualified" signal

This gives high recall on true abstain cases (~55%) with very low false
positives on disputed (~0%) and trustworthy (~4%).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from fitz_ai.governance.protocol import EvidenceItem
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from ..base import ConstraintResult

if TYPE_CHECKING:
    from fitz_ai.llm.providers.base import ChatProvider

logger = get_logger(__name__)


# Fast-tier jury prompts: lenient relevance checks
# All use YES = relevant, NO = not relevant. "When in doubt, answer YES."
JURY_PROMPTS = [
    # P1: Mentions any relevant entity/concept
    """Read the question and the context below.

Question: {query}
Context: {context}

Does the context mention any person, place, thing, or concept that the question asks about? Even a single mention counts as YES.
Answer NO only if the context does not mention anything the question refers to.
When in doubt, answer YES.

ONE word: YES or NO""",
    # P2: Information presence
    """Read the question and the context below.

Question: {query}
Context: {context}

Does the context contain any information that could help answer this question? Even partial or indirect information counts as YES.
Only answer NO if the context has zero relevant information.
When in doubt, answer YES.

ONE word: YES or NO""",
    # P3: Meaningful connection
    """Read the question and the context below.

Question: {query}
Context: {context}

Is there a meaningful connection between the question and the context? If the context mentions anything related to what the question asks about, answer YES.
Only answer NO if someone reading the context would learn nothing useful about the question.
When in doubt, answer YES.

ONE word: YES or NO""",
]

# Balanced-tier confirmation prompt (runs only when 2+/3 fast prompts say NO)
BALANCED_CONFIRM_PROMPT = """You are verifying whether retrieved context is relevant to a user's question.

Question: {query}
Context: {context}

Does the context contain information that is relevant and useful for answering this question?

Answer YES if the context has any relevant information, even partial.
Answer NO if the context is completely irrelevant and contains nothing useful for the question.

Reply with ONE word: YES or NO"""


@dataclass
class AnswerVerificationConstraint:
    """
    Verifies chunks actually answer the query using fast-tier jury + balanced confirmation.

    Two-stage design:
      Stage 1: 3 fast-tier prompts check relevance (lenient, low cost)
      Stage 2: 1 balanced-tier call confirms when 2+/3 fast say NO

    This keeps cost low (balanced only runs ~20% of the time) while
    achieving high accuracy through the smarter model's confirmation.

    Attributes:
        chat: Fast-tier ChatProvider for jury calls
        chat_balanced: Balanced-tier ChatProvider for confirmation (optional)
        enabled: Whether this constraint is active (default: True)
        max_context_chars: Max characters to include from context (default: 1000)
    """

    name: str = "answer_verification"
    chat: "ChatProvider | None" = None
    chat_balanced: "ChatProvider | None" = None
    enabled: bool = True
    max_context_chars: int = 1000
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def _run_jury(self, query: str, context: str) -> tuple[int, list[str]]:
        """
        Run 3 fast-tier prompts to check if context is relevant to query.

        Returns:
            Tuple of (no_votes, responses) where no_votes is count of
            prompts that said context is not relevant.
        """
        if not self.chat:
            return 0, ["NO_CHAT"] * 3

        no_votes = 0
        responses = []

        for i, prompt_template in enumerate(JURY_PROMPTS):
            prompt = prompt_template.format(query=query, context=context)

            try:
                response = self.chat.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
                word = response.strip().upper()
                responses.append(word)

                # All prompts: NO means context not relevant
                if word.startswith("NO"):
                    no_votes += 1

            except Exception as e:
                logger.warning(
                    f"{PIPELINE} AnswerVerificationConstraint: jury prompt {i} failed: {e}"
                )
                responses.append("ERROR")

        return no_votes, responses

    def _run_balanced_confirm(self, query: str, context: str) -> bool:
        """
        Run a single balanced-tier confirmation call.

        Returns:
            True if balanced model also says context is NOT relevant (confirms denial).
        """
        if not self.chat_balanced:
            return True  # No balanced model = trust fast jury

        prompt = BALANCED_CONFIRM_PROMPT.format(query=query, context=context)

        try:
            response = self.chat_balanced.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            word = response.strip().upper()
            return word.startswith("NO")
        except Exception as e:
            logger.warning(f"{PIPELINE} AnswerVerificationConstraint: balanced confirm failed: {e}")
            return False  # On error, don't fire (conservative)

    def apply(self, query: str, chunks: Sequence[EvidenceItem]) -> ConstraintResult:
        """
        Check if chunks are relevant to query using fast jury + balanced confirmation.

        Flow:
          1. Run 3 fast-tier prompts
          2. If 2+/3 say NO -> run balanced confirmation
          3. If balanced also says NO -> fire "qualified" signal

        Args:
            query: User query
            chunks: Retrieved chunks

        Returns:
            ConstraintResult with jury_votes metadata (always emitted)
        """
        if not self.enabled:
            return ConstraintResult.allow()

        if not chunks:
            return ConstraintResult.allow()

        if not self.chat:
            logger.debug(f"{PIPELINE} AnswerVerificationConstraint: no chat provider, skipping")
            return ConstraintResult.allow()

        # Combine top chunks for verification
        context_parts = []
        total_chars = 0
        for chunk in chunks[:3]:
            remaining = self.max_context_chars - total_chars
            if remaining <= 0:
                break
            content = chunk.content[:remaining]
            context_parts.append(content)
            total_chars += len(content)

        context = "\n\n---\n\n".join(context_parts)

        # Stage 1: Fast jury
        no_votes, responses = self._run_jury(query, context)

        logger.debug(
            f"{PIPELINE} AnswerVerificationConstraint jury: "
            f"no_votes={no_votes}/3, responses={responses}"
        )

        # Stage 2: Balanced confirmation when 2+/3 fast say NO
        if no_votes >= 2:
            confirmed = self._run_balanced_confirm(query, context)

            if confirmed:
                logger.info(
                    f"{PIPELINE} AnswerVerificationConstraint: fast jury {no_votes}/3 NO, "
                    f"balanced confirmed -> firing"
                )
                return ConstraintResult.deny(
                    reason="Retrieved content may not directly answer the question",
                    signal="qualified",
                    jury_votes=no_votes,
                    jury_responses=responses,
                    balanced_confirmed=True,
                )
            else:
                logger.debug(
                    f"{PIPELINE} AnswerVerificationConstraint: fast jury {no_votes}/3 NO, "
                    f"balanced rejected -> allowing"
                )
                return ConstraintResult.allow(
                    jury_votes=no_votes,
                    jury_responses=responses,
                    balanced_confirmed=False,
                )

        # Less than 2/3 fast NO -> allow
        return ConstraintResult.allow(
            jury_votes=no_votes,
            jury_responses=responses,
        )


__all__ = ["AnswerVerificationConstraint"]
