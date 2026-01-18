# fitz_ai/structured/types.py
"""
Type inference and coercion for structured data.

Infers column types from sample values and provides utilities
for type conversion and validation.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

# Type constants
TYPE_STRING = "string"
TYPE_NUMBER = "number"
TYPE_DATE = "date"
TYPE_BOOLEAN = "boolean"

# Date patterns (ordered by specificity)
DATE_PATTERNS = [
    # ISO format
    r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?",
    # US format
    r"^\d{1,2}/\d{1,2}/\d{4}$",
    # EU format
    r"^\d{1,2}-\d{1,2}-\d{4}$",
    # Month name formats
    r"^\w+ \d{1,2},? \d{4}$",
    r"^\d{1,2} \w+ \d{4}$",
]

# Boolean values (case-insensitive)
BOOLEAN_TRUE = {"true", "yes", "1", "t", "y"}
BOOLEAN_FALSE = {"false", "no", "0", "f", "n"}


def infer_type(value: Any) -> str:
    """
    Infer the type of a single value.

    Args:
        value: Value to analyze

    Returns:
        One of: "string", "number", "date", "boolean"
    """
    if value is None:
        return TYPE_STRING  # Default for nulls

    # Already typed
    if isinstance(value, bool):
        return TYPE_BOOLEAN
    if isinstance(value, (int, float)):
        return TYPE_NUMBER
    if isinstance(value, datetime):
        return TYPE_DATE

    # String analysis
    if isinstance(value, str):
        value = value.strip()

        if not value:
            return TYPE_STRING

        # Boolean check
        if value.lower() in BOOLEAN_TRUE | BOOLEAN_FALSE:
            return TYPE_BOOLEAN

        # Number check
        try:
            float(value.replace(",", ""))  # Handle comma separators
            return TYPE_NUMBER
        except ValueError:
            pass

        # Date check
        for pattern in DATE_PATTERNS:
            if re.match(pattern, value):
                return TYPE_DATE

    return TYPE_STRING


def infer_column_type(values: list[Any], sample_size: int = 100) -> str:
    """
    Infer the type of a column from sample values.

    Uses majority voting with null handling. If mixed types detected,
    falls back to string.

    Args:
        values: List of column values
        sample_size: Max values to sample for inference

    Returns:
        One of: "string", "number", "date", "boolean"
    """
    if not values:
        return TYPE_STRING

    # Sample values (skip nulls for type inference)
    non_null = [v for v in values[:sample_size] if v is not None and v != ""]

    if not non_null:
        return TYPE_STRING

    # Count types
    type_counts: dict[str, int] = {}
    for value in non_null:
        inferred = infer_type(value)
        type_counts[inferred] = type_counts.get(inferred, 0) + 1

    # Majority wins (but require high confidence for non-string)
    total = len(non_null)
    for type_name in [TYPE_NUMBER, TYPE_DATE, TYPE_BOOLEAN]:
        count = type_counts.get(type_name, 0)
        if count / total >= 0.9:  # 90% threshold
            return type_name

    return TYPE_STRING


def coerce_value(value: Any, target_type: str) -> Any:
    """
    Coerce a value to the target type.

    Args:
        value: Value to coerce
        target_type: Target type ("string", "number", "date", "boolean")

    Returns:
        Coerced value (or original if coercion fails)
    """
    if value is None:
        return None

    if target_type == TYPE_STRING:
        return str(value) if value is not None else None

    if target_type == TYPE_NUMBER:
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                cleaned = value.strip().replace(",", "")
                if "." in cleaned:
                    return float(cleaned)
                return int(cleaned)
            except ValueError:
                return None
        return None

    if target_type == TYPE_BOOLEAN:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() in BOOLEAN_TRUE:
                return True
            if value.lower() in BOOLEAN_FALSE:
                return False
        return None

    if target_type == TYPE_DATE:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str):
            # Return as-is for storage (parsing happens at query time)
            return value

    return value


def is_indexable_column(
    column_type: str,
    unique_ratio: float,
    is_primary_key: bool = False,
) -> bool:
    """
    Determine if a column should be indexed for filtering.

    Heuristics:
    - Primary keys are always indexed
    - Booleans are always indexed (2 values = fast filter)
    - Dates are always indexed (range queries)
    - Numbers with high cardinality are indexed
    - Strings with low cardinality (enums) are indexed

    Args:
        column_type: Type of the column
        unique_ratio: Ratio of unique values to total rows (0.0-1.0)
        is_primary_key: Whether this is the primary key

    Returns:
        True if column should be indexed
    """
    if is_primary_key:
        return True

    if column_type == TYPE_BOOLEAN:
        return True  # Always useful for filtering

    if column_type == TYPE_DATE:
        return True  # Range queries are common

    if column_type == TYPE_NUMBER:
        # Index numbers that aren't too unique (continuous values)
        # and aren't all the same (constant)
        return 0.01 < unique_ratio < 0.9

    if column_type == TYPE_STRING:
        # Index strings that look like enums (low cardinality)
        # E.g., department, status, category
        return unique_ratio < 0.3  # Less than 30% unique = likely enum

    return False


def select_indexed_columns(
    column_names: list[str],
    column_types: list[str],
    sample_values: list[list[Any]],
    primary_key: str,
    max_indexed: int = 5,
) -> list[str]:
    """
    Auto-select columns to index for filtering.

    Args:
        column_names: List of column names
        column_types: List of column types (parallel to names)
        sample_values: List of value lists per column
        primary_key: Name of the primary key column
        max_indexed: Maximum columns to index (excluding PK)

    Returns:
        List of column names to index
    """
    indexed = []

    for i, (name, col_type) in enumerate(zip(column_names, column_types)):
        if name == primary_key:
            indexed.append(name)
            continue

        # Calculate unique ratio
        values = sample_values[i] if i < len(sample_values) else []
        non_null = [v for v in values if v is not None and v != ""]
        unique_ratio = len(set(non_null)) / len(non_null) if non_null else 0.0

        if is_indexable_column(col_type, unique_ratio, is_primary_key=False):
            indexed.append(name)

        if len(indexed) >= max_indexed + 1:  # +1 for PK
            break

    return indexed


# =============================================================================
# Typed Models for SQL Execution
# =============================================================================


from dataclasses import dataclass, field
from typing import TypedDict


class RowRecord(TypedDict, total=False):
    """
    Row record from SQL query results.

    Contains column values plus optional nested row data.
    """

    __row_data: dict[str, Any]  # Nested row structure for complex queries


@dataclass
class TableRow:
    """A row from a table with typed access."""

    data: dict[str, Any]
    row_id: str | None = None

    def get(self, column: str, default: Any = None) -> Any:
        """Get a column value with optional default."""
        return self.data.get(column, default)

    def __getitem__(self, column: str) -> Any:
        """Get a column value by key."""
        return self.data[column]


@dataclass
class AggregationResult:
    """Result from an aggregation query (COUNT, SUM, AVG, etc.)."""

    value: Any
    function: str  # COUNT, SUM, AVG, MIN, MAX
    column: str | None = None


@dataclass
class GroupByResult:
    """Result from a GROUP BY query."""

    groups: dict[str, list[dict[str, Any]]]
    aggregations: dict[str, Any] = field(default_factory=dict)
    group_by_columns: list[str] = field(default_factory=list)


class SQLFilterCondition(TypedDict, total=False):
    """
    SQL filter condition converted to Qdrant format.

    Examples:
        {"key": "name", "match": {"value": "Alice"}}
        {"key": "age", "range": {"gte": 18}}
    """

    key: str
    match: dict[str, Any]
    range: dict[str, Any]


@dataclass
class SQLFilter:
    """
    Complete SQL filter specification.

    Represents a WHERE clause converted to Qdrant filter format.
    """

    conditions: list[SQLFilterCondition] = field(default_factory=list)
    operator: str = "AND"  # AND, OR

    def to_qdrant_filter(self) -> dict[str, Any]:
        """Convert to Qdrant filter format."""
        if not self.conditions:
            return {}

        if self.operator == "AND":
            return {"must": self.conditions}
        else:
            return {"should": self.conditions}


__all__ = [
    # Type constants
    "TYPE_STRING",
    "TYPE_NUMBER",
    "TYPE_DATE",
    "TYPE_BOOLEAN",
    # Type inference functions
    "infer_type",
    "infer_column_type",
    "coerce_value",
    "is_indexable_column",
    "select_indexed_columns",
    # Typed models
    "RowRecord",
    "TableRow",
    "AggregationResult",
    "GroupByResult",
    "SQLFilterCondition",
    "SQLFilter",
]
