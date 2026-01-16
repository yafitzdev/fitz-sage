# fitz_ai/structured/executor.py
"""
SQL execution via vector DB metadata filtering.

Translates SQL queries to metadata filters, fetches matching rows,
and performs client-side aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from fitz_ai.logging.logger import get_logger
from fitz_ai.structured.constants import (
    FIELD_ROW_DATA,
    FIELD_TABLE,
    MAX_RESULT_NAMES,
    MAX_SCAN_ROWS,
    get_tables_collection,
)
from fitz_ai.structured.sql_generator import SQLQuery

logger = get_logger(__name__)


class QueryLimitExceededError(Exception):
    """Raised when query would scan too many rows."""

    def __init__(self, scanned: int, limit: int = MAX_SCAN_ROWS):
        self.scanned = scanned
        self.limit = limit
        super().__init__(
            f"Query would scan {scanned} rows, exceeding limit of {limit}"
        )


@runtime_checkable
class VectorDBClient(Protocol):
    """Protocol for vector DB operations."""

    def scroll(
        self,
        collection_name: str,
        limit: int,
        offset: int = 0,
        scroll_filter: dict[str, Any] | None = None,
        with_payload: bool = True,
    ) -> tuple[list[Any], int | None]:
        """Scroll through collection with optional filter."""
        ...


@dataclass
class ExecutionResult:
    """Result from SQL execution."""

    data: dict[str, Any]  # Aggregation results or row data
    row_count: int  # Number of rows processed
    query: SQLQuery
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None


def _condition_to_filter(condition: dict[str, Any]) -> dict[str, Any] | None:
    """
    Convert a SQL condition to vector DB filter format.

    Supports Qdrant-style filter format:
    - match: {"value": x} for equality
    - range: {"gt": x, "lt": y} for comparisons
    """
    column = condition.get("column")
    op = condition.get("op", "=")
    value = condition.get("value")

    if not column:
        return None

    # Equality
    if op in ("=", "=="):
        return {"key": column, "match": {"value": value}}

    # Not equal (not directly supported, skip)
    if op in ("!=", "<>"):
        logger.warning(f"!= operator not supported for column {column}, skipping")
        return None

    # Comparisons
    if op == ">":
        return {"key": column, "range": {"gt": value}}
    if op == ">=":
        return {"key": column, "range": {"gte": value}}
    if op == "<":
        return {"key": column, "range": {"lt": value}}
    if op == "<=":
        return {"key": column, "range": {"lte": value}}

    # BETWEEN
    if op == "BETWEEN" and isinstance(value, list) and len(value) == 2:
        return {"key": column, "range": {"gte": value[0], "lte": value[1]}}

    # IN (multiple match conditions with should)
    if op == "IN" and isinstance(value, list):
        return {
            "should": [{"key": column, "match": {"value": v}} for v in value]
        }

    # LIKE (partial match - limited support)
    if op == "LIKE":
        # Remove SQL wildcards for simple contains check
        clean_value = value.replace("%", "").replace("_", "")
        logger.warning(f"LIKE converted to exact match for {column}={clean_value}")
        return {"key": column, "match": {"value": clean_value}}

    return None


def _build_scroll_filter(query: SQLQuery) -> dict[str, Any] | None:
    """Build vector DB scroll filter from SQL query."""
    conditions = []

    # Always filter by table
    conditions.append({"key": FIELD_TABLE, "match": {"value": query.table}})

    # Add WHERE conditions
    for cond in query.where:
        filter_cond = _condition_to_filter(cond)
        if filter_cond:
            conditions.append(filter_cond)

    if len(conditions) == 1:
        return conditions[0]

    return {"must": conditions}


def _extract_row_value(row: dict[str, Any], column: str) -> Any:
    """Extract column value from row payload."""
    # Try direct access (indexed columns)
    if column in row:
        return row[column]

    # Try from __row data
    row_data = row.get(FIELD_ROW_DATA, {})
    if column in row_data:
        return row_data[column]

    return None


def _aggregate_count(rows: list[dict[str, Any]], column: str = "*") -> int:
    """Count rows or non-null values."""
    if column == "*":
        return len(rows)

    return sum(1 for r in rows if _extract_row_value(r, column) is not None)


def _aggregate_sum(rows: list[dict[str, Any]], column: str) -> float:
    """Sum numeric column values."""
    total = 0.0
    for row in rows:
        value = _extract_row_value(row, column)
        if value is not None:
            try:
                total += float(value)
            except (TypeError, ValueError):
                pass
    return total


def _aggregate_avg(rows: list[dict[str, Any]], column: str) -> float | None:
    """Average numeric column values."""
    values = []
    for row in rows:
        value = _extract_row_value(row, column)
        if value is not None:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass

    if not values:
        return None
    return sum(values) / len(values)


def _aggregate_min(rows: list[dict[str, Any]], column: str) -> Any:
    """Find minimum value."""
    values = []
    for row in rows:
        value = _extract_row_value(row, column)
        if value is not None:
            values.append(value)

    return min(values) if values else None


def _aggregate_max(rows: list[dict[str, Any]], column: str) -> Any:
    """Find maximum value."""
    values = []
    for row in rows:
        value = _extract_row_value(row, column)
        if value is not None:
            values.append(value)

    return max(values) if values else None


def _aggregate_group_concat(
    rows: list[dict[str, Any]], column: str, limit: int = MAX_RESULT_NAMES
) -> str:
    """Concatenate column values as comma-separated string."""
    values = []
    for row in rows[:limit]:
        value = _extract_row_value(row, column)
        if value is not None:
            values.append(str(value))

    result = ", ".join(values)
    if len(rows) > limit:
        result += f" (and {len(rows) - limit} more)"
    return result


def _parse_aggregation(select_expr: str) -> tuple[str, str]:
    """
    Parse aggregation expression like 'COUNT(*)' or 'SUM(salary)'.

    Returns: (function_name, column_name)
    """
    import re

    match = re.match(r"(\w+)\s*\(\s*(\*|\w+)\s*\)", select_expr.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper(), match.group(2)

    # Not an aggregation, treat as column name
    return "", select_expr.strip()


def _execute_aggregations(
    rows: list[dict[str, Any]], query: SQLQuery
) -> dict[str, Any]:
    """Execute aggregations on fetched rows."""
    results = {}

    for select_expr in query.select:
        func_name, column = _parse_aggregation(select_expr)

        if func_name == "COUNT":
            results[select_expr] = _aggregate_count(rows, column)
        elif func_name == "SUM":
            results[select_expr] = _aggregate_sum(rows, column)
        elif func_name == "AVG":
            results[select_expr] = _aggregate_avg(rows, column)
        elif func_name == "MIN":
            results[select_expr] = _aggregate_min(rows, column)
        elif func_name == "MAX":
            results[select_expr] = _aggregate_max(rows, column)
        elif func_name == "GROUP_CONCAT":
            results[select_expr] = _aggregate_group_concat(rows, column)
        else:
            # Not an aggregation - collect values
            values = [_extract_row_value(r, column) for r in rows]
            results[column] = values

    return results


def _execute_group_by(
    rows: list[dict[str, Any]], query: SQLQuery
) -> dict[str, Any]:
    """Execute GROUP BY aggregations."""
    if not query.group_by:
        return _execute_aggregations(rows, query)

    # Group rows by the GROUP BY columns
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(_extract_row_value(row, col) for col in query.group_by)
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    # Aggregate each group
    results = {}
    for group_key, group_rows in groups.items():
        group_name = ", ".join(str(k) for k in group_key)
        group_results = _execute_aggregations(group_rows, query)
        results[group_name] = group_results

    return {"groups": results, "group_by": query.group_by}


class StructuredExecutor:
    """
    Executes SQL queries via vector DB metadata filtering.

    Translates SQL to metadata filters, fetches rows, and performs
    client-side aggregation.
    """

    def __init__(self, vector_db: VectorDBClient, base_collection: str):
        """
        Initialize executor.

        Args:
            vector_db: Vector DB client with scroll support
            base_collection: Base collection name
        """
        self._vector_db = vector_db
        self._tables_collection = get_tables_collection(base_collection)

    def execute(self, query: SQLQuery) -> ExecutionResult:
        """
        Execute a SQL query.

        Args:
            query: Parsed SQL query

        Returns:
            ExecutionResult with aggregated data
        """
        # Build filter
        scroll_filter = _build_scroll_filter(query)

        # Fetch rows with pagination
        rows = []
        offset = 0
        batch_size = 100

        try:
            while True:
                # Safety check
                if len(rows) >= MAX_SCAN_ROWS:
                    raise QueryLimitExceededError(len(rows))

                batch, next_offset = self._vector_db.scroll(
                    collection_name=self._tables_collection,
                    limit=batch_size,
                    offset=offset,
                    scroll_filter=scroll_filter,
                    with_payload=True,
                )

                if not batch:
                    break

                # Extract payloads
                for record in batch:
                    payload = getattr(record, "payload", None) or record.get("payload", {})
                    rows.append(payload)

                if next_offset is None or len(batch) < batch_size:
                    break

                offset = next_offset

        except QueryLimitExceededError:
            raise
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return ExecutionResult(
                data={},
                row_count=0,
                query=query,
                error=f"Fetch failed: {e}",
            )

        logger.debug(f"Fetched {len(rows)} rows for query on {query.table}")

        # Apply LIMIT if specified (for non-aggregation queries)
        if query.limit and not query.is_aggregation:
            # Sort first if ORDER BY specified
            if query.order_by:
                rows.sort(
                    key=lambda r: _extract_row_value(r, query.order_by) or 0,
                    reverse=query.order_desc,
                )
            rows = rows[: query.limit]

        # Execute aggregations
        if query.group_by:
            data = _execute_group_by(rows, query)
        else:
            data = _execute_aggregations(rows, query)

        return ExecutionResult(
            data=data,
            row_count=len(rows),
            query=query,
        )


__all__ = [
    "StructuredExecutor",
    "ExecutionResult",
    "QueryLimitExceededError",
]
