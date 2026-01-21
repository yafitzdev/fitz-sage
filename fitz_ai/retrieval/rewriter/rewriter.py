# fitz_ai/retrieval/rewriter/rewriter.py
"""
Query rewriter for improved retrieval.

Rewrites queries for:
- Conversational context resolution (pronouns, references)
- Clarity/simplification (typos, noise, complex phrasing)
- Retrieval optimization (question -> statement form)
- Ambiguity detection and disambiguation

Uses a single LLM call for efficiency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

from .types import ConversationContext, RewriteResult, RewriteType

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_rag.protocols import ChatClient

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a prompt template by name."""
    prompt_path = _PROMPTS_DIR / f"{name}.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


@dataclass
class QueryRewriter:
    """
    Rewrites queries for improved retrieval using LLM.

    Performs all rewrite types in a single LLM call:
    1. Conversational context resolution
    2. Clarity/simplification
    3. Retrieval optimization
    4. Ambiguity detection

    Uses fast-tier chat model for efficiency.
    """

    chat: "ChatClient"
    prompt_template: str | None = field(default=None, repr=False)
    min_query_length: int = 3  # Skip rewriting for very short queries

    def __post_init__(self):
        """Load default prompt template if not provided."""
        if self.prompt_template is None:
            self.prompt_template = _load_prompt("rewrite")

    def rewrite(
        self,
        query: str,
        context: ConversationContext | None = None,
    ) -> RewriteResult:
        """
        Rewrite a query for improved retrieval.

        Args:
            query: Original user query
            context: Optional conversation context for pronoun resolution

        Returns:
            RewriteResult with rewritten query and metadata
        """
        # Skip rewriting for very short queries
        if len(query.strip()) < self.min_query_length:
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                rewrite_type=RewriteType.NONE,
                confidence=1.0,
            )

        # Build prompt
        prompt = self._build_prompt(query, context)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.chat.chat(messages)
            result = self._parse_response(response, query)

            if result.was_rewritten:
                logger.debug(
                    f"{RETRIEVER} Query rewrite: '{query[:50]}...' -> "
                    f"'{result.rewritten_query[:50]}...' (type={result.rewrite_type.value})"
                )
            else:
                logger.debug(f"{RETRIEVER} Query rewrite: no changes needed")

            return result

        except Exception as e:
            logger.warning(f"{RETRIEVER} Query rewriting failed: {e}")
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                rewrite_type=RewriteType.NONE,
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def _build_prompt(
        self,
        query: str,
        context: ConversationContext | None,
    ) -> str:
        """Build the rewrite prompt."""
        history_section = ""
        if context and not context.is_empty():
            history_section = f"""
## Conversation History
{context.format_for_prompt()}
"""

        return self.prompt_template.format(
            query=query,
            history_section=history_section,
        )

    def _parse_response(self, response: str, original_query: str) -> RewriteResult:
        """Parse LLM response into RewriteResult."""
        try:
            # Handle markdown code blocks
            text = response.strip()
            if text.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1].strip()
                    if text.startswith("json"):
                        text = text[4:].strip()

            # Try to find JSON object in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

            result = json.loads(text)

            # Extract fields with defaults
            rewritten = result.get("rewritten_query", original_query)
            rewrite_type_str = result.get("rewrite_type", "none")
            confidence = float(result.get("confidence", 0.5))
            is_ambiguous = result.get("is_ambiguous", False)
            disambiguated = result.get("disambiguated_queries", [])

            # Map rewrite type
            type_mapping = {
                "none": RewriteType.NONE,
                "conversational": RewriteType.CONVERSATIONAL,
                "clarity": RewriteType.CLARITY,
                "retrieval": RewriteType.RETRIEVAL,
                "combined": RewriteType.COMBINED,
            }
            rewrite_type = type_mapping.get(rewrite_type_str.lower(), RewriteType.NONE)

            # Validate rewritten query
            if not rewritten or not rewritten.strip():
                rewritten = original_query
                rewrite_type = RewriteType.NONE

            return RewriteResult(
                original_query=original_query,
                rewritten_query=rewritten.strip(),
                rewrite_type=rewrite_type,
                confidence=confidence,
                is_ambiguous=is_ambiguous,
                disambiguated_queries=disambiguated[:3],  # Limit to 3
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"{RETRIEVER} Failed to parse rewrite response: {e}")

            return RewriteResult(
                original_query=original_query,
                rewritten_query=original_query,
                rewrite_type=RewriteType.NONE,
                confidence=0.0,
            )
