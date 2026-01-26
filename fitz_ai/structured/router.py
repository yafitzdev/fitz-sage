# fitz_ai/structured/router.py
"""
Query routing for structured vs semantic queries.

Uses LLM-based semantic classification to detect whether a query
should be routed to structured data (SQL) or semantic search (RAG).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from fitz_ai.llm.factory import ChatFactory, ModelTier
from fitz_ai.logging.logger import get_logger
from fitz_ai.structured.schema import SchemaStore, TableSchema

logger = get_logger(__name__)


@dataclass
class SemanticRoute:
    """Route decision for semantic (RAG) queries."""

    reason: str = "No structured data match"


@dataclass
class StructuredRoute:
    """Route decision for structured (SQL) queries."""

    tables: list[TableSchema]
    scores: list[float]
    query_type: str = ""  # e.g., "aggregation", "listing", "filtering"
    confidence: float = 0.0

    @property
    def primary_table(self) -> TableSchema:
        """Get the highest-scoring matched table."""
        return self.tables[0]


RouteDecision = SemanticRoute | StructuredRoute


# Prompt for semantic classification
CLASSIFICATION_PROMPT = """You are a query classifier. Determine if a user query should be answered using:
1. STRUCTURED: Query requires aggregation, counting, filtering, or listing from a database table
2. SEMANTIC: Query is asking for explanations, concepts, or free-form knowledge

Given the user query and available table schemas, respond with JSON only:
{{
  "route": "structured" or "semantic",
  "confidence": 0.0-1.0,
  "query_type": "aggregation|listing|filtering|comparison|lookup" (only if structured),
  "reason": "brief explanation"
}}

Available tables:
{schemas}

User query: {query}

Respond with JSON only, no explanation outside the JSON."""


def _format_schemas_for_prompt(schemas: list[TableSchema]) -> str:
    """Format table schemas for the classification prompt."""
    if not schemas:
        return "No tables available."

    lines = []
    for schema in schemas:
        cols = ", ".join(f"{c.name} ({c.type})" for c in schema.columns)
        lines.append(f"- {schema.table_name}: {cols} ({schema.row_count} rows)")

    return "\n".join(lines)


def _parse_classification_response(response: str) -> dict[str, Any]:
    """Parse LLM classification response."""
    # Clean up response - remove markdown code blocks if present
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        # Remove first and last lines (code block markers)
        lines = [line for line in lines if not line.startswith("```")]
        response = "\n".join(lines)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse classification response: {response[:100]}")
        return {"route": "semantic", "confidence": 0.0, "reason": "Parse error"}


class QueryRouter:
    """
    Routes queries to structured or semantic paths.

    Uses LLM-based semantic classification to determine if a query
    should be handled via SQL or RAG.
    """

    # Tier for classification (developer decision - fast for routing)
    TIER_CLASSIFY: ModelTier = "fast"

    def __init__(
        self,
        schema_store: SchemaStore,
        chat_factory: ChatFactory,
        schema_match_threshold: float = 0.3,
        structured_confidence_threshold: float = 0.6,
    ):
        """
        Initialize router.

        Args:
            schema_store: Schema store for table discovery
            chat_factory: Chat factory for per-task tier selection
            schema_match_threshold: Min score for schema match
            structured_confidence_threshold: Min confidence to route to structured
        """
        self._schema_store = schema_store
        self._chat_factory = chat_factory
        self._schema_match_threshold = schema_match_threshold
        self._confidence_threshold = structured_confidence_threshold

    def route(self, query: str) -> RouteDecision:
        """
        Route a query to structured or semantic path.

        Uses semantic LLM classification to determine intent.

        Args:
            query: User query text

        Returns:
            RouteDecision (SemanticRoute or StructuredRoute)
        """
        # Step 1: Search for matching schemas
        schema_results = self._schema_store.search_tables(
            query=query,
            limit=5,
            min_score=self._schema_match_threshold,
        )

        if not schema_results:
            logger.debug(f"No schema match for query: {query[:50]}...")
            return SemanticRoute(reason="No matching table schemas found")

        # Step 2: Use LLM to classify the query semantically
        schemas = [r.schema for r in schema_results]
        classification = self._classify_query(query, schemas)

        route_type = classification.get("route", "semantic").lower()
        confidence = float(classification.get("confidence", 0.0))
        query_type = classification.get("query_type", "")
        reason = classification.get("reason", "")

        # Step 3: Route based on classification
        if route_type == "structured" and confidence >= self._confidence_threshold:
            logger.info(
                f"Routing to structured: {schemas[0].table_name}, "
                f"type={query_type}, confidence={confidence:.2f}"
            )
            return StructuredRoute(
                tables=schemas,
                scores=[r.score for r in schema_results],
                query_type=query_type,
                confidence=confidence,
            )

        logger.debug(f"Routing to semantic: {reason} (confidence={confidence:.2f})")
        return SemanticRoute(reason=reason or "LLM classified as semantic query")

    def _classify_query(self, query: str, schemas: list[TableSchema]) -> dict[str, Any]:
        """
        Use LLM to semantically classify the query.

        Args:
            query: User query
            schemas: Matched table schemas

        Returns:
            Classification dict with route, confidence, query_type, reason
        """
        schemas_text = _format_schemas_for_prompt(schemas)
        prompt = CLASSIFICATION_PROMPT.format(
            schemas=schemas_text,
            query=query,
        )

        try:
            chat = self._chat_factory(self.TIER_CLASSIFY)
            response = chat.chat([{"role": "user", "content": prompt}])
            return _parse_classification_response(response)

        except Exception as e:
            logger.warning(f"Classification failed: {e}, defaulting to semantic")
            return {
                "route": "semantic",
                "confidence": 0.0,
                "reason": f"Classification error: {e}",
            }

    def should_use_structured(self, query: str) -> bool:
        """
        Quick check if query should use structured path.

        Args:
            query: User query text

        Returns:
            True if structured path should be used
        """
        return isinstance(self.route(query), StructuredRoute)


__all__ = [
    "QueryRouter",
    "RouteDecision",
    "SemanticRoute",
    "StructuredRoute",
]
