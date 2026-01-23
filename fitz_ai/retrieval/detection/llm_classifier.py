# fitz_ai/retrieval/detection/llm_classifier.py
"""
LLM-based query classification for detection.

Replaces regex-based detection with a single LLM call that classifies
all detection categories at once. More robust and handles edge cases
that regex patterns miss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

CLASSIFICATION_PROMPT = '''Classify this search query. Return JSON only.

Query: "{query}"

Return this exact structure:
{{
  "temporal": {{
    "detected": true/false,
    "intent": "COMPARISON" | "TREND" | "POINT_IN_TIME" | "RANGE" | "SEQUENCE" | null,
    "references": [],
    "time_focused_queries": []
  }},
  "aggregation": {{
    "detected": true/false,
    "type": "LIST" | "COUNT" | "UNIQUE" | null,
    "target": null,
    "fetch_multiplier": 1
  }},
  "comparison": {{
    "detected": true/false,
    "entities": [],
    "comparison_queries": []
  }},
  "freshness": {{
    "boost_recency": true/false,
    "boost_authority": true/false
  }},
  "rewriter": {{
    "needs_context": true/false,
    "is_compound": true/false,
    "decomposed_queries": []
  }}
}}

Rules:
- temporal: time periods, versions, dates, quarters, "between X and Y", "since", "before", "last week/month/year"
- temporal.intent: COMPARISON="between X and Y", TREND="over time/history", POINT_IN_TIME="as of/in Q1", RANGE="from X to Y", SEQUENCE="first/then/after"
- aggregation: "list all", "how many", "count", "enumerate", "what are all the", "show me all"
- aggregation.type: LIST="list/show/enumerate", COUNT="how many/count", UNIQUE="unique/distinct"
- comparison: "vs", "versus", "compare", "difference between", "differ", "which is better", "how does X compare to Y", "X or Y"
- freshness.boost_recency: "latest", "recent", "new", "current", "updated", "newest"
- freshness.boost_authority: "official", "recommended", "best practice", "standard", "proper way"
- rewriter.needs_context: contains "it", "this", "that", "they", "the same" without clear referent in query
- rewriter.is_compound: multiple distinct questions, "and also", semicolons separating topics

Be generous with detection - it's better to trigger a strategy that helps than to miss it.'''


class ChatProtocol(Protocol):
    """Protocol for chat clients."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send messages and get response."""
        ...


@dataclass
class LLMClassifier:
    """
    Single LLM call to classify query into all detection categories.

    Replaces 7 separate regex-based detectors with one fast LLM call.
    """

    chat_client: ChatProtocol

    def classify(self, query: str) -> dict[str, Any]:
        """
        Classify query into all detection categories.

        Args:
            query: User's query string

        Returns:
            Dict with classification results for all categories
        """
        prompt = CLASSIFICATION_PROMPT.format(query=query)

        try:
            response = self.chat_client.chat([{"role": "user", "content": prompt}])
            return self._parse_response(response)
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return self._empty_classification()

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = response.strip()

        # Try to find JSON in code block
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        continue

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        start = text.find("{")
        if start >= 0:
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break

        logger.warning(f"Failed to parse LLM response as JSON: {text[:100]}...")
        return self._empty_classification()

    def _empty_classification(self) -> dict[str, Any]:
        """Return empty classification (no detection)."""
        return {
            "temporal": {"detected": False},
            "aggregation": {"detected": False},
            "comparison": {"detected": False},
            "freshness": {"boost_recency": False, "boost_authority": False},
            "rewriter": {"needs_context": False, "is_compound": False},
        }
