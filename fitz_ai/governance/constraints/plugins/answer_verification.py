# fitz_ai/governance/constraints/plugins/answer_verification.py
"""
Answer Verification Constraint - LLM jury for positive confidence confirmation.

Uses 3-prompt fusion with majority voting to verify chunks actually answer
the query. Prevents false trustworthy answers when context is semantically
relevant but doesn't contain the requested information.

The jury approach reduces LLM variance by requiring 2+ NO votes to trigger.
This is conservative - benefit of doubt goes to allowing trustworthy answers.
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


# Jury prompts - same question asked 3 different ways
# Reduces variance through majority voting
JURY_PROMPTS = [
    # Direct answerability check
    """Can this question be answered using the provided context?

Question: {query}
Context: {context}

Answer YES if the context contains information to answer the question.
Answer NO if the context doesn't address what the question asks.

Reply with ONE word: YES or NO""",
    # Sufficiency framing (inverted logic)
    """Is the context INSUFFICIENT to answer the question?

Question: {query}
Context: {context}

Answer YES if the context lacks the needed information.
Answer NO if the context has enough to answer.

Reply with ONE word: YES or NO""",
    # Completeness framing
    """Could someone write a complete answer to this question using only this context?

Question: {query}
Context: {context}

Answer YES if the context provides what's needed.
Answer NO if critical information is missing.

Reply with ONE word: YES or NO""",
]


@dataclass
class AnswerVerificationConstraint:
    """
    Verifies chunks actually answer the query using 3-prompt LLM jury.

    Prevents false trustworthy answers when context is semantically relevant
    but doesn't contain the requested information.

    Example failure mode this solves:
    - Query: "What is the capital of France?"
    - Context: "France has 67 million people and is famous for wine."
    - Without verification: TRUSTWORTHY (no constraint triggered)
    - With verification: fires "qualified" signal (jury agrees context doesn't answer)

    Jury voting:
    - 3 prompts ask "does this answer?" in different ways
    - Require 3/3 NO votes to fire (very conservative - unanimous jury)
    - 0-2 NO votes = allow trustworthy (benefit of doubt)

    This is conservative - we only fire when the jury unanimously agrees
    the context clearly doesn't answer.

    Attributes:
        chat: ChatProvider for LLM jury calls
        enabled: Whether this constraint is active (default: True)
        max_context_chars: Max characters to include from context (default: 1000)
    """

    name: str = "answer_verification"
    chat: "ChatProvider | None" = None
    enabled: bool = True
    max_context_chars: int = 1000
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def _run_jury(self, query: str, context: str) -> tuple[int, list[str]]:
        """
        Run 3-prompt jury to check if context answers query.

        Returns:
            Tuple of (no_votes, responses) where no_votes is count of
            prompts that said context doesn't answer.
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

                # Interpret response based on prompt framing
                if i == 0:  # Direct: NO means doesn't answer
                    if word.startswith("NO"):
                        no_votes += 1
                elif i == 1:  # Inverted: YES means insufficient (doesn't answer)
                    if word.startswith("YES"):
                        no_votes += 1
                elif i == 2:  # Completeness: NO means can't write answer
                    if word.startswith("NO"):
                        no_votes += 1

            except Exception as e:
                logger.warning(
                    f"{PIPELINE} AnswerVerificationConstraint: jury prompt {i} failed: {e}"
                )
                responses.append("ERROR")

        return no_votes, responses

    def apply(self, query: str, chunks: Sequence[EvidenceItem]) -> ConstraintResult:
        """
        Check if chunks actually answer the query using LLM jury.

        Args:
            query: User query
            chunks: Retrieved chunks

        Returns:
            ConstraintResult - denies confident if jury agrees context doesn't answer
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

        # Run jury
        no_votes, responses = self._run_jury(query, context)

        logger.debug(
            f"{PIPELINE} AnswerVerificationConstraint jury: "
            f"no_votes={no_votes}/3, responses={responses}"
        )

        # Require 3/3 NO votes to qualify (very conservative - unanimous jury)
        if no_votes >= 3:
            logger.info(
                f"{PIPELINE} AnswerVerificationConstraint: jury ruled context "
                f"doesn't answer ({no_votes}/3 NO votes)"
            )
            return ConstraintResult.deny(
                reason="Retrieved content may not directly answer the question",
                signal="qualified",
                jury_votes=no_votes,
                jury_responses=responses,
            )

        return ConstraintResult.allow()


__all__ = ["AnswerVerificationConstraint"]
