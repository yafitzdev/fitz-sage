# fitz_ai/tabular/models.py
"""
Data models for tabular data routing.

ParsedTable represents a table extracted from a document.
Schema chunks are created with embedded table data for vector DB storage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from fitz_ai.core.chunk import Chunk


@dataclass
class ParsedTable:
    """
    Table extracted from a document.

    Stores headers and rows in a structured format that can be serialized
    to JSON for storage in chunk payloads.
    """

    table_id: str
    source_doc: str
    headers: list[str]
    rows: list[list[str]]
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize table data for storage in chunk payload."""
        return json.dumps(
            {
                "headers": self.headers,
                "rows": self.rows,
            }
        )

    @classmethod
    def from_json(cls, data: str, table_id: str, source_doc: str) -> ParsedTable:
        """Deserialize table data from chunk payload."""
        parsed = json.loads(data)
        return cls(
            table_id=table_id,
            source_doc=source_doc,
            headers=parsed["headers"],
            rows=parsed["rows"],
        )

    @property
    def row_count(self) -> int:
        """Number of data rows (excluding header)."""
        return len(self.rows)

    @property
    def column_count(self) -> int:
        """Number of columns."""
        return len(self.headers)


def _format_sample_rows(table: ParsedTable, max_rows: int = 3) -> str:
    """Format sample rows for human-readable content."""
    if not table.rows:
        return "(empty table)"

    lines = []
    for row in table.rows[:max_rows]:
        # Truncate long cell values
        truncated = [cell[:30] + "..." if len(cell) > 30 else cell for cell in row]
        lines.append(" | ".join(truncated))

    if len(table.rows) > max_rows:
        lines.append(f"... and {len(table.rows) - max_rows} more rows")

    return "\n  ".join(lines)


def create_schema_chunk(table: ParsedTable) -> Chunk:
    """
    Create a schema chunk with embedded table data.

    The chunk content is human-readable (for embedding/retrieval).
    The full table data is stored in metadata as JSON (for SQL queries).

    Args:
        table: Parsed table to create chunk for.

    Returns:
        Chunk with table schema in content and full data in metadata.
    """
    # Human-readable content for embedding/retrieval
    content = f"""Table from {table.source_doc}
Columns: {', '.join(table.headers)}
Row count: {table.row_count} rows
Sample data:
  {_format_sample_rows(table, max_rows=3)}"""

    return Chunk(
        id=f"table_{table.table_id}",
        doc_id=table.source_doc,
        content=content,
        chunk_index=0,
        metadata={
            "is_table_schema": True,
            "table_id": table.table_id,
            "table_data": table.to_json(),  # Full data in payload
            "columns": table.headers,
            "row_count": table.row_count,
            "source_page": table.page,
        },
    )


__all__ = [
    "ParsedTable",
    "create_schema_chunk",
]
