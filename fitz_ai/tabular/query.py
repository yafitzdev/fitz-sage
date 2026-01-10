# fitz_ai/tabular/query.py
"""
Table Query Step - Handles schema chunks at query time.

When a schema chunk is retrieved:
1. LLM selects relevant columns (column pruning)
2. Pruned data is extracted from JSON payload
3. In-memory SQLite is built with pruned columns
4. LLM generates SQL query
5. SQL is executed
6. Results are formatted and chunk is augmented
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from fitz_ai.core.chunk import Chunk
from fitz_ai.engines.fitz_rag.retrieval.steps.base import ChatClient, RetrievalStep

from .models import ParsedTable

logger = logging.getLogger(__name__)


@dataclass
class TableQueryStep(RetrievalStep):
    """
    Retrieval step that handles table schema chunks.

    For each schema chunk retrieved:
    1. Prunes columns based on query (via LLM)
    2. Builds in-memory SQLite with pruned data
    3. Generates and executes SQL query (via LLM)
    4. Augments chunk with query results

    Regular (non-table) chunks pass through unchanged.

    Args:
        chat: Chat client for LLM calls (column selection + SQL generation).
        max_results: Maximum number of SQL result rows to include (default: 100).
    """

    chat: ChatClient
    max_results: int = 100

    COLUMN_SELECT_PROMPT = """Given this table schema and question, which columns are needed to answer?

Table columns: {columns}
Question: {question}

Return ONLY a JSON array of column names needed. Example: ["col1", "col2"]
Include all columns that might be relevant for filtering or returning results."""

    SQL_PROMPT = """Generate a SQLite query to answer this question.

Table name: data
Columns: {columns}
Sample values: {samples}

Question: {question}

Rules:
1. Use only the columns listed above
2. Table name is 'data'
3. Use LIMIT {max_results} unless aggregating
4. For text search use LIKE with % wildcards
5. Column names with spaces need double quotes

Return ONLY the SQL query, no explanation."""

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """
        Process chunks, handling table schema chunks specially.

        Args:
            query: User query.
            chunks: Retrieved chunks (may include schema chunks).

        Returns:
            Chunks with table schema chunks augmented with query results.
        """
        result_chunks: list[Chunk] = []

        for chunk in chunks:
            if chunk.metadata.get("is_table_schema"):
                augmented = self._process_table_chunk(query, chunk)
                result_chunks.append(augmented)
            else:
                result_chunks.append(chunk)

        return result_chunks

    def _process_table_chunk(self, query: str, chunk: Chunk) -> Chunk:
        """
        Process a table schema chunk.

        1. Load table data from payload
        2. Select relevant columns via LLM
        3. Build pruned SQLite
        4. Generate and execute SQL
        5. Return augmented chunk

        Args:
            query: User query.
            chunk: Table schema chunk.

        Returns:
            Augmented chunk with query results, or original on failure.
        """
        try:
            # 1. Load table data from payload
            table_data = chunk.metadata.get("table_data")
            if not table_data:
                logger.warning(f"Table chunk {chunk.id} missing table_data")
                return chunk

            table = ParsedTable.from_json(
                table_data,
                chunk.metadata.get("table_id", "unknown"),
                chunk.doc_id,
            )

            logger.debug(
                f"Processing table {table.table_id}: "
                f"{table.column_count} cols, {table.row_count} rows"
            )

            # 2. LLM selects relevant columns
            needed_cols = self._select_columns(query, table.headers)
            logger.debug(f"Selected columns: {needed_cols}")

            # 3. Prune to needed columns only
            col_indices = [
                table.headers.index(c) for c in needed_cols if c in table.headers
            ]
            if not col_indices:
                # Fallback: use all columns
                col_indices = list(range(len(table.headers)))
                needed_cols = table.headers

            pruned_headers = [table.headers[i] for i in col_indices]
            pruned_rows = [[row[i] for i in col_indices] for row in table.rows]

            logger.debug(
                f"Pruned from {table.column_count} to {len(pruned_headers)} columns"
            )

            # 4. Build in-memory SQLite with pruned data
            conn = sqlite3.connect(":memory:")
            self._create_and_populate(conn, pruned_headers, pruned_rows)

            # 5. LLM generates SQL
            sample_rows = pruned_rows[:3] if pruned_rows else []
            sql = self._generate_sql(query, pruned_headers, sample_rows)
            logger.debug(f"Generated SQL: {sql}")

            # 6. Execute SQL
            try:
                cursor = conn.execute(sql)
                results = cursor.fetchall()
                col_names = [d[0] for d in cursor.description]
            except sqlite3.Error as e:
                logger.warning(f"SQL execution failed: {e}")
                conn.close()
                return self._augment_chunk_error(chunk, sql, str(e))

            conn.close()

            # 7. Format results and augment chunk
            return self._augment_chunk(chunk, sql, col_names, results)

        except Exception as e:
            logger.warning(f"Table query failed for {chunk.id}: {e}")
            return chunk  # Return original on failure

    def _select_columns(self, query: str, columns: list[str]) -> list[str]:
        """Use LLM to select relevant columns."""
        prompt = self.COLUMN_SELECT_PROMPT.format(
            columns=columns,
            question=query,
        )
        response = self.chat.chat([{"role": "user", "content": prompt}])
        return self._parse_json_list(response, fallback=columns)

    def _generate_sql(
        self, query: str, columns: list[str], sample_rows: list[list[str]]
    ) -> str:
        """Use LLM to generate SQL query."""
        # Format samples as dict for prompt
        samples: dict[str, list[str]] = {}
        for i, col in enumerate(columns):
            samples[col] = [row[i] for row in sample_rows if i < len(row)]

        prompt = self.SQL_PROMPT.format(
            columns=columns,
            samples=samples,
            question=query,
            max_results=self.max_results,
        )
        response = self.chat.chat([{"role": "user", "content": prompt}])
        return self._extract_sql(response)

    def _create_and_populate(
        self, conn: sqlite3.Connection, headers: list[str], rows: list[list[str]]
    ) -> None:
        """Create table and insert rows."""
        # Use TEXT for all columns (safest for mixed data)
        cols_def = ", ".join(f'"{h}" TEXT' for h in headers)
        conn.execute(f"CREATE TABLE data ({cols_def})")

        # Insert rows
        placeholders = ", ".join("?" * len(headers))
        conn.executemany(f"INSERT INTO data VALUES ({placeholders})", rows)
        conn.commit()

    def _augment_chunk(
        self, chunk: Chunk, sql: str, cols: list[str], rows: list[tuple]
    ) -> Chunk:
        """Create augmented chunk with query results."""
        results_md = self._format_as_markdown(cols, rows)

        content = f"""{chunk.content}

--- Query Results ---
SQL: {sql}
Results ({len(rows)} rows):
{results_md}"""

        return Chunk(
            id=chunk.id,
            doc_id=chunk.doc_id,
            content=content,
            chunk_index=chunk.chunk_index,
            metadata={
                **chunk.metadata,
                "sql_executed": sql,
                "result_count": len(rows),
            },
        )

    def _augment_chunk_error(self, chunk: Chunk, sql: str, error: str) -> Chunk:
        """Create augmented chunk with error message."""
        content = f"""{chunk.content}

--- Query Results ---
SQL: {sql}
Error: {error}

The table data is available but the generated SQL query failed.
Please reformulate your question or ask about specific column values."""

        return Chunk(
            id=chunk.id,
            doc_id=chunk.doc_id,
            content=content,
            chunk_index=chunk.chunk_index,
            metadata={
                **chunk.metadata,
                "sql_executed": sql,
                "sql_error": error,
            },
        )

    def _format_as_markdown(self, cols: list[str], rows: list[tuple]) -> str:
        """Format query results as markdown table."""
        if not rows:
            return "(no results)"

        # Limit rows for display
        display_rows = rows[: self.max_results]

        # Build markdown table
        lines = []

        # Header
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")

        # Data rows
        for row in display_rows:
            # Truncate long values
            cells = []
            for val in row:
                s = str(val) if val is not None else ""
                if len(s) > 50:
                    s = s[:47] + "..."
                cells.append(s)
            lines.append("| " + " | ".join(cells) + " |")

        if len(rows) > self.max_results:
            lines.append(f"\n... and {len(rows) - self.max_results} more rows")

        return "\n".join(lines)

    def _parse_json_list(self, response: str, fallback: list[str]) -> list[str]:
        """Parse JSON list from LLM response."""
        try:
            text = response.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1].strip()
                    if text.startswith("json"):
                        text = text[4:].strip()

            result = json.loads(text)
            if isinstance(result, list):
                return [str(item) for item in result]
        except (json.JSONDecodeError, ValueError):
            pass

        logger.warning("Failed to parse column selection, using all columns")
        return fallback

    def _extract_sql(self, response: str) -> str:
        """Extract SQL from LLM response."""
        text = response.strip()

        # Handle markdown code blocks
        if "```" in text:
            # Find SQL in code block
            match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
            else:
                # Just strip the markers
                text = text.replace("```sql", "").replace("```", "").strip()

        # Basic validation
        if not text.upper().startswith("SELECT"):
            # Try to find SELECT statement
            match = re.search(r"(SELECT\s+.+)", text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1)

        return text


__all__ = ["TableQueryStep"]
