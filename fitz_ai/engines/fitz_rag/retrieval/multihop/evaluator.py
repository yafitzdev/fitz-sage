# fitz_ai/engines/fitz_rag/retrieval/multihop/evaluator.py
"""
Evidence evaluator for multi-hop retrieval.

Uses LLM to judge whether retrieved evidence is sufficient to answer the query.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

if TYPE_CHECKING:
    from fitz_ai.core.chunk import Chunk

logger = get_logger(__name__)


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for chat services."""

    def chat(self, messages: list[dict[str, Any]]) -> str: ...


class EvidenceEvaluator:
    """
    Evaluates if retrieved evidence is sufficient to answer a query.

    Uses a fast LLM to judge whether the current evidence contains enough
    information to answer the original question, or if more retrieval is needed.

    The evaluator is intentionally conservative - it will request more retrieval
    when uncertain, as getting more context is cheap compared to hallucination.
    """

    def __init__(self, chat: ChatClient, max_context_chars: int = 5000):
        """
        Initialize the evaluator.

        Args:
            chat: Fast-tier chat client for evaluation
            max_context_chars: Maximum characters to include in evaluation prompt
        """
        self.chat = chat
        self.max_context_chars = max_context_chars

    def evaluate(self, query: str, chunks: list["Chunk"]) -> bool:
        """
        Determine if chunks contain enough evidence to answer query.

        Args:
            query: Original user query
            chunks: All retrieved chunks so far

        Returns:
            True if sufficient, False if more retrieval needed
        """
        if not chunks:
            logger.debug(f"{RETRIEVER} Evidence: INSUFFICIENT (no chunks)")
            return False

        # Build context from chunks (truncate to fit)
        context = self._build_context(chunks)

        prompt = f"""Given this question and retrieved evidence, determine if there is enough information to answer the question.

Question: {query}

Retrieved evidence:
{context}

Respond with ONLY "SUFFICIENT" or "INSUFFICIENT".

SUFFICIENT means: The evidence contains the key facts needed to answer the question directly.
INSUFFICIENT means: The evidence is missing critical information, or only contains partial/indirect information.
"""

        response = self.chat.chat([{"role": "user", "content": prompt}])
        # Check for exact word match to avoid "INSUFFICIENT" matching "SUFFICIENT"
        response_upper = response.upper()
        is_sufficient = "SUFFICIENT" in response_upper and "INSUFFICIENT" not in response_upper

        logger.debug(
            f"{RETRIEVER} Evidence: {'SUFFICIENT' if is_sufficient else 'INSUFFICIENT'} "
            f"(evaluated {len(chunks)} chunks)"
        )

        return is_sufficient

    def _build_context(self, chunks: list["Chunk"]) -> str:
        """Build context string from chunks, respecting max length."""
        context_parts: list[str] = []
        total_chars = 0

        for chunk in chunks:
            content = chunk.content[:500]  # Truncate individual chunks
            if total_chars + len(content) > self.max_context_chars:
                break
            context_parts.append(content)
            total_chars += len(content) + 2  # +2 for newlines

        return "\n\n".join(context_parts)
