# fitz_ai/structured/formatter.py
"""
Result formatting for structured queries.

Converts SQL execution results to natural language sentences using LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from fitz_ai.logging.logger import get_logger
from fitz_ai.structured.executor import ExecutionResult

logger = get_logger(__name__)


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for chat completion."""

    def chat(self, messages: list[dict[str, Any]]) -> str:
        """Send messages and get response."""
        ...


@dataclass
class FormattedResult:
    """Result of formatting SQL execution results."""

    sentence: str  # Natural language sentence
    query: str  # Original SQL
    table: str  # Source table
    data: dict[str, Any]  # Raw execution data


# Formatting prompt
FORMATTING_PROMPT = """You are a data analyst. Convert the following SQL query result into a clear, concise natural language sentence.

SQL Query: {sql}
Table: {table}
Result: {result}

Rules:
1. Be precise - include exact numbers from the result
2. Be concise - one sentence, no introduction
3. Format numbers appropriately (use commas for thousands, $ for money if obvious)
4. If the result is empty or zero, state that clearly
5. Do not add opinions or analysis, just state the fact

Examples:
- Query: SELECT COUNT(*) FROM employees WHERE department = 'engineering'
  Result: {{"COUNT(*)": 42}}
  Output: There are 42 employees in the engineering department.

- Query: SELECT SUM(salary), AVG(salary) FROM employees
  Result: {{"SUM(salary)": 5000000, "AVG(salary)": 100000}}
  Output: The total salary is $5,000,000 with an average salary of $100,000.

- Query: SELECT department, COUNT(*) FROM employees GROUP BY department
  Result: {{"groups": {{"engineering": {{"COUNT(*)": 20}}, "sales": {{"COUNT(*)": 15}}}}, "group_by": ["department"]}}
  Output: There are 20 employees in engineering and 15 in sales.

Respond with the sentence only, no quotes or explanation."""


def _format_result_for_prompt(result: dict[str, Any]) -> str:
    """Format execution result for prompt."""
    return json.dumps(result, default=str)


class ResultFormatter:
    """
    Formats SQL execution results into natural language.

    Uses LLM to generate clear, concise sentences from query results.
    """

    def __init__(self, chat_client: ChatClient):
        """
        Initialize formatter.

        Args:
            chat_client: Chat client for LLM calls (use fast tier)
        """
        self._chat = chat_client

    def format(self, execution_result: ExecutionResult) -> FormattedResult:
        """
        Format execution result as natural language sentence.

        Args:
            execution_result: Result from StructuredExecutor

        Returns:
            FormattedResult with natural language sentence
        """
        query = execution_result.query
        data = execution_result.data

        prompt = FORMATTING_PROMPT.format(
            sql=query.raw_sql or self._reconstruct_sql(query),
            table=query.table,
            result=_format_result_for_prompt(data),
        )

        try:
            response = self._chat.chat([{"role": "user", "content": prompt}])
            sentence = response.strip().strip('"\'')

            logger.debug(f"Formatted result for {query.table}: {sentence[:50]}...")

            return FormattedResult(
                sentence=sentence,
                query=query.raw_sql or self._reconstruct_sql(query),
                table=query.table,
                data=data,
            )

        except Exception as e:
            logger.error(f"Formatting failed: {e}")
            # Fallback to basic formatting
            sentence = self._fallback_format(query, data)
            return FormattedResult(
                sentence=sentence,
                query=query.raw_sql or self._reconstruct_sql(query),
                table=query.table,
                data=data,
            )

    def _reconstruct_sql(self, query) -> str:
        """Reconstruct SQL string from SQLQuery object."""
        parts = ["SELECT", ", ".join(query.select), "FROM", query.table]

        if query.where:
            conditions = []
            for cond in query.where:
                col = cond.get("column", "")
                op = cond.get("op", "=")
                val = cond.get("value", "")
                if isinstance(val, str):
                    conditions.append(f"{col} {op} '{val}'")
                else:
                    conditions.append(f"{col} {op} {val}")
            parts.extend(["WHERE", " AND ".join(conditions)])

        if query.group_by:
            parts.extend(["GROUP BY", ", ".join(query.group_by)])

        if query.order_by:
            direction = "DESC" if query.order_desc else "ASC"
            parts.extend(["ORDER BY", query.order_by, direction])

        if query.limit:
            parts.extend(["LIMIT", str(query.limit)])

        return " ".join(parts)

    def _fallback_format(self, query, data: dict[str, Any]) -> str:
        """Basic formatting when LLM fails."""
        parts = []

        for key, value in data.items():
            if key in ("groups", "group_by"):
                continue

            if "COUNT" in key.upper():
                parts.append(f"count is {value}")
            elif "SUM" in key.upper():
                parts.append(f"total is {value:,}" if isinstance(value, (int, float)) else f"total is {value}")
            elif "AVG" in key.upper():
                parts.append(f"average is {value:,.2f}" if isinstance(value, float) else f"average is {value}")
            elif "MIN" in key.upper():
                parts.append(f"minimum is {value}")
            elif "MAX" in key.upper():
                parts.append(f"maximum is {value}")
            else:
                parts.append(f"{key} is {value}")

        if not parts:
            return f"Query on {query.table} returned no results."

        return f"For {query.table}: " + ", ".join(parts) + "."


def format_multiple_results(
    formatter: ResultFormatter, results: list[ExecutionResult]
) -> list[FormattedResult]:
    """Format multiple execution results."""
    return [formatter.format(result) for result in results]


__all__ = [
    "ResultFormatter",
    "FormattedResult",
    "format_multiple_results",
]
