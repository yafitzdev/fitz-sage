# fitz_ai/retrieval/hyde/generator.py
"""
HyDE (Hypothetical Document Embeddings) generator.

Generates hypothetical documents from queries to improve retrieval recall.
The hypothetical documents are then embedded and used for retrieval alongside
the original query, with results merged via RRF.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_rag.protocols import ChatClient

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a prompt template by name."""
    prompt_path = _PROMPTS_DIR / f"{name}.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


@dataclass
class HydeGenerator:
    """
    Generates hypothetical documents from queries using LLM.

    Uses a single LLM call to generate multiple hypothetical documents
    that could contain the answer to the query. These are then embedded
    and searched alongside the original query for improved recall.
    """

    chat: "ChatClient"
    num_hypotheses: int = 3
    prompt_template: str | None = field(default=None, repr=False)

    def __post_init__(self):
        """Load default prompt template if not provided."""
        if self.prompt_template is None:
            self.prompt_template = _load_prompt("hypothesis")

    def generate(self, query: str) -> list[str]:
        """
        Generate hypothetical documents for a query.

        Args:
            query: User's query string

        Returns:
            List of hypothetical document passages (usually 3)
        """
        prompt = self._build_prompt(query)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.chat.chat(messages)
            hypotheses = self._parse_response(response)

            logger.debug(f"{RETRIEVER} HyDE: generated {len(hypotheses)} hypotheses for query")

            return hypotheses

        except Exception as e:
            logger.warning(f"{RETRIEVER} HyDE generation failed: {e}")
            return []

    def _build_prompt(self, query: str) -> str:
        """Build the prompt with query and hypothesis count."""
        return self.prompt_template.format(
            query=query,
            num_hypotheses=self.num_hypotheses,
        )

    def _parse_response(self, response: str) -> list[str]:
        """Parse LLM response into list of hypothesis strings."""
        # Try to parse as JSON array
        try:
            # Find JSON array in response
            start = response.find("[")
            end = response.rfind("]") + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                parsed = json.loads(json_str)

                if isinstance(parsed, list):
                    # Filter to strings only, limit to num_hypotheses
                    hypotheses = [
                        str(h).strip() for h in parsed if isinstance(h, str) and h.strip()
                    ]
                    return hypotheses[: self.num_hypotheses]

        except json.JSONDecodeError:
            pass

        # Fallback: split by newlines if JSON parsing fails
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        # Filter out lines that look like JSON syntax brackets only
        # Strip quotes from lines that may be JSON string elements
        hypotheses = []
        for line in lines:
            if line.startswith("[") or line.startswith("]"):
                continue
            # Strip leading/trailing quotes and commas (from malformed JSON arrays)
            cleaned = line.strip('",')
            if len(cleaned) > 20:
                hypotheses.append(cleaned)

        return hypotheses[: self.num_hypotheses]
