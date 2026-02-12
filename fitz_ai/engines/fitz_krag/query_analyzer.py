# fitz_ai/engines/fitz_krag/query_analyzer.py
"""
Query analyzer for fitz_krag — classifies queries by knowledge type.

Uses a single LLM call to determine whether a query targets code, documentation,
or both, so the retrieval router can prioritize the right strategies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fitz_ai.llm.providers.base import ChatProvider

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Knowledge type a query targets."""

    CODE = "code"
    DOCUMENTATION = "documentation"
    GENERAL = "general"
    CROSS = "cross"
    DATA = "data"


@dataclass(frozen=True)
class QueryAnalysis:
    """Result of analyzing a query's knowledge type intent."""

    primary_type: QueryType
    secondary_type: QueryType | None = None
    confidence: float = 0.5
    entities: tuple[str, ...] = ()
    refined_query: str = ""

    @property
    def strategy_weights(self) -> dict[str, float]:
        """Compute per-strategy weights from query analysis."""
        weights = _TYPE_WEIGHTS.get(self.primary_type, _TYPE_WEIGHTS[QueryType.GENERAL])
        return dict(weights)


# Default strategy weights per query type
_TYPE_WEIGHTS: dict[QueryType, dict[str, float]] = {
    QueryType.CODE: {"code": 0.75, "section": 0.1, "table": 0.05, "chunk": 0.1},
    QueryType.DOCUMENTATION: {"code": 0.1, "section": 0.75, "table": 0.05, "chunk": 0.1},
    QueryType.GENERAL: {"code": 0.25, "section": 0.25, "table": 0.15, "chunk": 0.35},
    QueryType.CROSS: {"code": 0.35, "section": 0.35, "table": 0.1, "chunk": 0.2},
    QueryType.DATA: {"code": 0.05, "section": 0.15, "table": 0.70, "chunk": 0.10},
}

ANALYSIS_PROMPT = """Classify this search query by knowledge type. Return JSON only.

Query: "{query}"

Return this exact structure:
{{
  "primary_type": "code" | "documentation" | "general" | "cross" | "data",
  "secondary_type": null or "code" | "documentation" | "data",
  "confidence": 0.0-1.0,
  "entities": ["entity1", "entity2"],
  "refined_query": "cleaned query text"
}}

Categories:
- "code": References functions, classes, methods, implementations, code behavior
- "documentation": References document sections, specs, procedures, policies
- "data": Explicitly asks about CSV files, spreadsheet data, database tables, or SQL-like operations (filter rows, count records, aggregate columns). Queries referencing alphanumeric record identifiers (e.g. E016, ID-123) typically target data records. NOT for questions about facts, specifications, or information that happen to involve numbers
- "general": Overview questions, summaries, "what does this project do"
- "cross": Explicitly asks about both code and documentation together

"entities": Extract specific symbol names (functions, classes) or section titles mentioned.
"refined_query": Rewrite the query to be more specific for search. Keep original if already clear.
"confidence": How confident you are in the classification (0.0-1.0).

Return JSON only, no markdown."""


class QueryAnalyzer:
    """Classifies queries by knowledge type using a single LLM call."""

    def __init__(self, chat: "ChatProvider"):
        self._chat = chat

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Classify a query into knowledge type intent.

        Falls back to GENERAL with low confidence on any failure.
        """
        prompt = ANALYSIS_PROMPT.format(query=query)

        try:
            response = self._chat.chat([{"role": "user", "content": prompt}])
            return self._parse_response(response, query)
        except Exception as e:
            logger.warning(f"Query analysis failed, defaulting to GENERAL: {e}")
            return QueryAnalysis(
                primary_type=QueryType.GENERAL,
                confidence=0.3,
                refined_query=query,
            )

    def _parse_response(self, response: str, original_query: str) -> QueryAnalysis:
        """Parse LLM JSON response into QueryAnalysis."""
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.debug(f"Failed to parse query analysis JSON: {text[:200]}")
            return QueryAnalysis(
                primary_type=QueryType.GENERAL,
                confidence=0.3,
                refined_query=original_query,
            )

        primary = _parse_query_type(data.get("primary_type", "general"))
        secondary = None
        if data.get("secondary_type"):
            secondary = _parse_query_type(data["secondary_type"])

        entities = data.get("entities", [])
        if not isinstance(entities, list):
            entities = []

        return QueryAnalysis(
            primary_type=primary,
            secondary_type=secondary,
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
            entities=tuple(str(e) for e in entities),
            refined_query=str(data.get("refined_query", original_query)),
        )


def _parse_query_type(value: str) -> QueryType:
    """Parse a string into QueryType, defaulting to GENERAL."""
    try:
        return QueryType(value.lower())
    except ValueError:
        return QueryType.GENERAL
