# fitz_ai/structured/sql_generator.py
"""
SQL generation from natural language queries.

Uses LLM to generate SQL queries from user questions and table schemas.
Generated SQL is a simplified subset executable via metadata filtering.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from fitz_ai.llm.factory import ChatFactory, ModelTier
from fitz_ai.logging.logger import get_logger
from fitz_ai.structured.constants import MAX_QUERIES_PER_REQUEST, MAX_RESULT_NAMES
from fitz_ai.structured.schema import TableSchema

logger = get_logger(__name__)


@dataclass
class SQLQuery:
    """
    Parsed SQL query for execution.

    Represents a simplified SQL query that can be translated
    to vector DB metadata filters.
    """

    table: str
    select: list[str]  # Column names or aggregations like "COUNT(*)", "SUM(salary)"
    where: list[dict[str, Any]]  # List of conditions
    group_by: list[str] | None = None
    order_by: str | None = None
    order_desc: bool = True
    limit: int | None = None
    raw_sql: str = ""  # Original SQL for logging

    @property
    def is_aggregation(self) -> bool:
        """Check if this is an aggregation query."""
        agg_keywords = ["COUNT", "SUM", "AVG", "MIN", "MAX", "GROUP_CONCAT"]
        return any(any(kw in col.upper() for kw in agg_keywords) for col in self.select)

    @property
    def aggregation_type(self) -> str | None:
        """Get the primary aggregation type."""
        for col in self.select:
            col_upper = col.upper()
            if "COUNT" in col_upper:
                return "COUNT"
            if "SUM" in col_upper:
                return "SUM"
            if "AVG" in col_upper:
                return "AVG"
            if "MIN" in col_upper:
                return "MIN"
            if "MAX" in col_upper:
                return "MAX"
            if "GROUP_CONCAT" in col_upper:
                return "GROUP_CONCAT"
        return None


@dataclass
class GenerationResult:
    """Result from SQL generation."""

    queries: list[SQLQuery]
    error: str | None = None


# SQL generation prompt
SQL_GENERATION_PROMPT = """You are a SQL generator. Generate SQL queries to answer the user's question using the available table schemas.

Rules:
1. Always use aggregation functions (COUNT, SUM, AVG, MIN, MAX) - never SELECT *
2. Use GROUP_CONCAT(column, ', ') to list names (limit to {max_names} results)
3. Generate 1-{max_queries} queries as needed
4. Only use columns that exist in the schema
5. Use simple WHERE conditions (=, >, <, >=, <=, LIKE, IN, BETWEEN)
6. For "top N" queries, use ORDER BY with LIMIT

Available tables:
{schemas}

User question: {query}

Respond with JSON array of SQL queries:
{{
  "queries": [
    {{
      "sql": "SELECT COUNT(*) FROM employees WHERE department = 'engineering'",
      "table": "employees",
      "description": "Count of engineering employees"
    }}
  ]
}}

Respond with JSON only."""


def _format_schemas_for_sql(schemas: list[TableSchema]) -> str:
    """Format table schemas for SQL generation prompt."""
    lines = []
    for schema in schemas:
        cols = []
        for c in schema.columns:
            indexed = " [indexed]" if c.indexed else ""
            cols.append(f"{c.name} ({c.type}){indexed}")
        lines.append(f"Table: {schema.table_name}")
        lines.append(f"  Columns: {', '.join(cols)}")
        lines.append(f"  Primary key: {schema.primary_key}")
        lines.append(f"  Rows: {schema.row_count}")
        lines.append("")
    return "\n".join(lines)


def _parse_sql_response(response: str) -> list[dict[str, Any]]:
    """Parse LLM response containing SQL queries."""
    response = response.strip()

    # Remove markdown code blocks
    if response.startswith("```"):
        lines = response.split("\n")
        lines = [line for line in lines if not line.startswith("```")]
        response = "\n".join(lines)

    try:
        data = json.loads(response)
        if isinstance(data, dict) and "queries" in data:
            return data["queries"]
        if isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse SQL response: {response[:100]}")
        return []


def _parse_where_clause(sql: str) -> list[dict[str, Any]]:
    """
    Parse WHERE clause into structured conditions.

    Returns list of condition dicts like:
    {"column": "salary", "op": ">", "value": 100000}
    """
    conditions = []

    # Extract WHERE clause
    where_match = re.search(
        r"\bWHERE\b\s+(.+?)(?:\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|$)", sql, re.IGNORECASE
    )
    if not where_match:
        return conditions

    where_clause = where_match.group(1).strip()

    # Handle BETWEEN first (before splitting by AND, since BETWEEN uses AND internally)
    between_pattern = re.compile(
        r"(\w+)\s+BETWEEN\s+['\"]?([^'\"]+?)['\"]?\s+AND\s+['\"]?([^'\"]+?)['\"]?(?=\s+AND\s+|\s*$)",
        re.IGNORECASE,
    )
    for match in between_pattern.finditer(where_clause):
        conditions.append(
            {
                "column": match.group(1),
                "op": "BETWEEN",
                "value": [
                    _parse_value(match.group(2).strip()),
                    _parse_value(match.group(3).strip()),
                ],
            }
        )

    # Remove BETWEEN clauses from where_clause for further processing
    remaining = between_pattern.sub("", where_clause).strip()

    # Split remaining by AND (simplified - doesn't handle nested conditions)
    if remaining:
        parts = re.split(r"\s+AND\s+", remaining, flags=re.IGNORECASE)
    else:
        parts = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Handle IN
        in_match = re.match(r"(\w+)\s+IN\s*\(([^)]+)\)", part, re.IGNORECASE)
        if in_match:
            values = [_parse_value(v.strip().strip("'\"")) for v in in_match.group(2).split(",")]
            conditions.append(
                {
                    "column": in_match.group(1),
                    "op": "IN",
                    "value": values,
                }
            )
            continue

        # Handle LIKE
        like_match = re.match(r"(\w+)\s+LIKE\s+['\"]([^'\"]+)['\"]", part, re.IGNORECASE)
        if like_match:
            conditions.append(
                {
                    "column": like_match.group(1),
                    "op": "LIKE",
                    "value": like_match.group(2),
                }
            )
            continue

        # Handle comparison operators
        comp_match = re.match(r"(\w+)\s*(>=|<=|!=|<>|=|>|<)\s*['\"]?([^'\"]+)['\"]?", part)
        if comp_match:
            conditions.append(
                {
                    "column": comp_match.group(1),
                    "op": comp_match.group(2),
                    "value": _parse_value(comp_match.group(3).strip()),
                }
            )

    return conditions


def _parse_value(value: str) -> Any:
    """Parse a string value to appropriate type."""
    value = value.strip()

    # Boolean
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    # Number
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    return value


def _parse_select_clause(sql: str) -> list[str]:
    """Parse SELECT clause to get columns/aggregations."""
    select_match = re.search(r"\bSELECT\b\s+(.+?)\s+\bFROM\b", sql, re.IGNORECASE)
    if not select_match:
        return []

    select_clause = select_match.group(1).strip()

    # Handle nested parentheses in aggregations
    columns = []
    current = ""
    paren_depth = 0

    for char in select_clause + ",":
        if char == "(":
            paren_depth += 1
            current += char
        elif char == ")":
            paren_depth -= 1
            current += char
        elif char == "," and paren_depth == 0:
            if current.strip():
                columns.append(current.strip())
            current = ""
        else:
            current += char

    return columns


def _parse_group_by(sql: str) -> list[str] | None:
    """Parse GROUP BY clause."""
    match = re.search(r"\bGROUP BY\b\s+(.+?)(?:\bORDER BY\b|\bLIMIT\b|$)", sql, re.IGNORECASE)
    if not match:
        return None

    cols = [c.strip() for c in match.group(1).split(",")]
    return cols if cols else None


def _parse_order_by(sql: str) -> tuple[str | None, bool]:
    """Parse ORDER BY clause. Returns (column, is_desc)."""
    match = re.search(r"\bORDER BY\b\s+(\w+)(?:\s+(ASC|DESC))?", sql, re.IGNORECASE)
    if not match:
        return None, True

    column = match.group(1)
    direction = match.group(2) or "DESC"
    return column, direction.upper() == "DESC"


def _parse_limit(sql: str) -> int | None:
    """Parse LIMIT clause."""
    match = re.search(r"\bLIMIT\b\s+(\d+)", sql, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _parse_table(sql: str) -> str:
    """Parse table name from FROM clause."""
    match = re.search(r"\bFROM\b\s+(\w+)", sql, re.IGNORECASE)
    return match.group(1) if match else ""


def _sql_to_query(sql_dict: dict[str, Any]) -> SQLQuery | None:
    """Convert SQL dict from LLM to SQLQuery object."""
    sql = sql_dict.get("sql", "")
    table = sql_dict.get("table", "") or _parse_table(sql)

    if not sql or not table:
        return None

    try:
        return SQLQuery(
            table=table,
            select=_parse_select_clause(sql),
            where=_parse_where_clause(sql),
            group_by=_parse_group_by(sql),
            order_by=_parse_order_by(sql)[0],
            order_desc=_parse_order_by(sql)[1],
            limit=_parse_limit(sql),
            raw_sql=sql,
        )
    except Exception as e:
        logger.warning(f"Failed to parse SQL '{sql}': {e}")
        return None


class SQLGenerator:
    """
    Generates SQL queries from natural language using LLM.

    Produces simplified SQL that can be executed via metadata filtering.
    """

    # Tier for SQL generation (developer decision - smart for accuracy-critical SQL)
    TIER_SQL_GENERATE: ModelTier = "smart"

    def __init__(self, chat_factory: ChatFactory):
        """
        Initialize generator.

        Args:
            chat_factory: Chat factory for per-task tier selection
        """
        self._chat_factory = chat_factory

    def generate(
        self,
        query: str,
        schemas: list[TableSchema],
    ) -> GenerationResult:
        """
        Generate SQL queries for a natural language question.

        Args:
            query: User's natural language question
            schemas: Available table schemas

        Returns:
            GenerationResult with list of SQLQuery objects
        """
        if not schemas:
            return GenerationResult(queries=[], error="No schemas provided")

        schemas_text = _format_schemas_for_sql(schemas)
        prompt = SQL_GENERATION_PROMPT.format(
            schemas=schemas_text,
            query=query,
            max_queries=MAX_QUERIES_PER_REQUEST,
            max_names=MAX_RESULT_NAMES,
        )

        try:
            chat = self._chat_factory(self.TIER_SQL_GENERATE)
            response = chat.chat([{"role": "user", "content": prompt}])
            sql_dicts = _parse_sql_response(response)

            if not sql_dicts:
                return GenerationResult(
                    queries=[],
                    error="LLM did not generate valid SQL",
                )

            # Parse each SQL into SQLQuery
            queries = []
            for sql_dict in sql_dicts[:MAX_QUERIES_PER_REQUEST]:
                parsed = _sql_to_query(sql_dict)
                if parsed:
                    queries.append(parsed)

            if not queries:
                return GenerationResult(
                    queries=[],
                    error="Failed to parse generated SQL",
                )

            logger.info(f"Generated {len(queries)} SQL queries for: {query[:50]}...")
            return GenerationResult(queries=queries)

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return GenerationResult(queries=[], error=str(e))


__all__ = [
    "SQLGenerator",
    "SQLQuery",
    "GenerationResult",
]
