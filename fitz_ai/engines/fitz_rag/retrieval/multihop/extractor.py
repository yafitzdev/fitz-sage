# fitz_ai/engines/fitz_rag/retrieval/multihop/extractor.py
"""
Bridge question extractor for multi-hop retrieval.

Uses LLM to generate follow-up questions that would fill gaps in evidence.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from fitz_ai.engines.fitz_rag.retrieval.multihop.utils import build_context_from_chunks
from fitz_ai.llm.factory import ChatFactory, ModelTier
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

if TYPE_CHECKING:
    from fitz_ai.core.chunk import Chunk

logger = get_logger(__name__)


class BridgeExtractor:
    """
    Extracts bridge questions to fill evidence gaps.

    Given a query and current evidence, generates follow-up search queries
    that would find the missing information needed to answer the original question.

    Examples:
        Query: "What does Sarah Chen's company manufacture?"
        Evidence: "Sarah Chen is CEO of TechCorp..."
        Bridge: "What products does TechCorp manufacture?"
    """

    # Tier for bridge extraction (developer decision - fast for bulk)
    TIER_EXTRACT: ModelTier = "fast"

    def __init__(
        self,
        chat_factory: ChatFactory,
        max_questions: int = 2,
        max_context_chars: int = 5000,
    ):
        """
        Initialize the extractor.

        Args:
            chat_factory: Chat factory for per-task tier selection
            max_questions: Maximum bridge questions to generate
            max_context_chars: Maximum characters to include in prompt
        """
        self.chat_factory = chat_factory
        self.max_questions = max_questions
        self.max_context_chars = max_context_chars

    def extract(self, query: str, chunks: list["Chunk"]) -> list[str]:
        """
        Generate follow-up questions to find missing evidence.

        Args:
            query: Original user query
            chunks: All retrieved chunks so far

        Returns:
            List of bridge questions (empty if no clear gaps)
        """
        context = build_context_from_chunks(chunks, max_chars=self.max_context_chars)

        prompt = f"""You're helping answer a question. The current evidence is missing information.

Original question: {query}

Current evidence:
{context}

What specific follow-up question would help find the missing information?
Focus on concrete entities, relationships, or facts that are referenced but not explained.

Return 1-2 short, focused search queries that would find the missing information.
Return ONLY a JSON array: ["query1", "query2"]
If no clear gaps, return: []
"""

        chat = self.chat_factory(self.TIER_EXTRACT)
        response = chat.chat([{"role": "user", "content": prompt}])
        questions = self._parse_response(response)

        if questions:
            logger.debug(f"{RETRIEVER} Bridge questions: {questions}")
        else:
            logger.debug(f"{RETRIEVER} No bridge questions generated")

        return questions

    def _parse_response(self, response: str) -> list[str]:
        """Parse JSON array from response."""
        try:
            text = response.strip()

            # Handle markdown code blocks
            if text.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1].strip()
                    if text.startswith("json"):
                        text = text[4:].strip()

            result = json.loads(text)
            if isinstance(result, list):
                return [str(q) for q in result[: self.max_questions]]
        except json.JSONDecodeError:
            pass

        return []
