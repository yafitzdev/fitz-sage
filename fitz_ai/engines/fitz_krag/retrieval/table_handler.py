# fitz_ai/engines/fitz_krag/retrieval/table_handler.py
"""
Table query handler — LLM SQL generation and execution for TABLE read results.

Runs after expansion, before assembly. Takes ReadResults that contain table
schemas and replaces their content with actual SQL query results.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from fitz_ai.engines.fitz_krag.types import AddressKind, ReadResult

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_ai.llm.providers.base import ChatProvider
    from fitz_ai.tabular.store.postgres import PostgresTableStore

logger = logging.getLogger(__name__)

COLUMN_SELECT_PROMPT = """Given this table schema and question, which columns are needed to answer?

Table columns: {columns}
Sample data: {samples}
Question: {question}

Return ONLY a JSON array of column names needed. Example: ["col1", "col2"]
Rules:
- Include columns needed for filtering AND for displaying results
- Look at sample data to find which columns contain relevant values for the question
- For "who/which/what" questions, ALWAYS include identifying columns (name, id, title, etc.)
- For numeric comparisons (highest, lowest, average), include the numeric column
- When in doubt, include more columns rather than fewer"""

SQL_PROMPT = """Generate a PostgreSQL query to answer this question.

Table name: {table_name}
Columns: {columns}
Sample values: {samples}

Question: {question}

Rules:
1. Use only the columns listed above
2. Use the exact table name: {table_name}
3. Use LIMIT {max_results} unless aggregating
4. For text search use ILIKE with % wildcards (PostgreSQL is case-insensitive with ILIKE)
5. Column names need double quotes if they contain special characters
6. For "highest/maximum" use ORDER BY column DESC LIMIT 1
7. For "lowest/minimum" use ORDER BY column ASC LIMIT 1
8. For "who/which" questions, include identifying columns (name, id) in SELECT
9. All columns are TEXT type. For numeric operations (MAX, MIN, AVG, SUM, ORDER BY numbers), \
use CAST(column AS NUMERIC) or column::NUMERIC

Return ONLY the SQL query, no explanation."""


class TableQueryHandler:
    """Generates SQL and executes queries for TABLE read results."""

    def __init__(
        self,
        chat: "ChatProvider",
        pg_table_store: "PostgresTableStore",
        config: "FitzKragConfig",
    ):
        self._chat = chat
        self._pg_table_store = pg_table_store
        self._config = config

    def process(self, query: str, read_results: list[ReadResult]) -> list[ReadResult]:
        """Identify TABLE results, generate SQL, execute, augment content."""
        table_results = [r for r in read_results if r.address.kind == AddressKind.TABLE]
        non_table_results = [r for r in read_results if r.address.kind != AddressKind.TABLE]

        if not table_results:
            return read_results

        augmented: list[ReadResult] = []
        for result in table_results:
            try:
                aug = self._process_table_result(query, result)
                augmented.append(aug)
            except Exception as e:
                logger.warning(f"Table query failed for {result.address.location}: {e}")
                augmented.append(result)

        return non_table_results + augmented

    def _process_table_result(self, query: str, result: ReadResult) -> ReadResult:
        """Process a single TABLE ReadResult: column select → SQL gen → execute."""
        table_id = result.metadata.get("table_id") or result.address.metadata.get("table_id")
        if not table_id:
            return result

        # Get table info from PostgresTableStore
        pg_table_name = self._pg_table_store.get_table_name(table_id)
        if not pg_table_name:
            return result

        col_info = self._pg_table_store.get_columns(table_id)
        if not col_info:
            return result

        sanitized_cols, original_cols = col_info
        row_count = self._pg_table_store.get_row_count(table_id)

        # Get sample data
        sample_rows = self._get_sample_data(pg_table_name, sanitized_cols)

        # LLM column selection (using original names)
        needed_original = self._select_columns(query, original_cols, sample_rows)
        col_mapping = dict(zip(original_cols, sanitized_cols))
        needed_sanitized = [col_mapping.get(c, c) for c in needed_original if c in col_mapping]
        if not needed_sanitized:
            needed_sanitized = sanitized_cols

        # Get samples for selected columns
        sample_for_sql = self._get_sample_data(pg_table_name, needed_sanitized)

        # LLM SQL generation with retry
        sql = self._generate_sql(query, pg_table_name, needed_sanitized, sample_for_sql)

        # Execute SQL
        exec_result = self._pg_table_store.execute_query(table_id, sql)
        if exec_result is None:
            return result

        col_names, rows = exec_result

        # If SQL returned 0 rows, don't augment — the table has no relevant data
        if not rows:
            logger.debug(f"SQL returned 0 rows for query '{query[:50]}', dropping table result")
            return result

        name = result.address.metadata.get("name", result.address.location)
        columns = result.address.metadata.get("columns", original_cols)

        # Format augmented content
        results_md = self._format_as_markdown(col_names, rows)
        content = (
            f"Table: {name}\n"
            f"Columns: {', '.join(columns)}\n"
            f"Total rows: {row_count}\n\n"
            f"--- SQL Query Results ---\n"
            f"Query: {sql}\n"
            f"Results ({len(rows)} rows):\n"
            f"{results_md}\n\n"
            f"Note: Results computed from all {row_count} rows."
        )

        return ReadResult(
            address=result.address,
            content=content,
            file_path=result.file_path,
            metadata={**result.metadata, "sql_executed": sql, "result_count": len(rows)},
        )

    def _get_sample_data(
        self, pg_table_name: str, columns: list[str], limit: int = 3
    ) -> list[list[str]]:
        """Fetch sample data from PostgreSQL table."""
        cols_str = ", ".join(f'"{c}"' for c in columns)
        sql = f'SELECT {cols_str} FROM "{pg_table_name}" LIMIT {limit}'
        result = self._pg_table_store.execute_query("", sql)
        if result:
            _, rows = result
            return [[str(v) if v is not None else "" for v in row] for row in rows]
        return []

    def _select_columns(
        self, query: str, columns: list[str], sample_rows: list[list[str]]
    ) -> list[str]:
        """Use LLM to select relevant columns."""
        samples_str = "(no sample data)"
        if sample_rows:
            sample_lines = []
            for row in sample_rows[:3]:
                row_str = " | ".join(f"{col}={val}" for col, val in zip(columns, row))
                sample_lines.append(row_str)
            samples_str = "\n".join(sample_lines)

        prompt = COLUMN_SELECT_PROMPT.format(
            columns=columns,
            samples=samples_str,
            question=query,
        )
        response = self._chat.chat([{"role": "user", "content": prompt}])
        return self._parse_json_list(response, fallback=columns)

    def _generate_sql(
        self,
        query: str,
        table_name: str,
        columns: list[str],
        sample_rows: list[list[str]],
        max_retries: int = 2,
    ) -> str:
        """Generate PostgreSQL query with validation and retry."""
        previous_error = None

        for attempt in range(max_retries + 1):
            sql = self._generate_sql_attempt(
                query, table_name, columns, sample_rows, previous_error
            )

            # Validate by test execution
            result = self._pg_table_store.execute_query("", sql)
            if result is not None:
                return sql

            previous_error = "Query execution failed"
            logger.warning(f"SQL validation failed (attempt {attempt + 1}/{max_retries + 1})")

        return sql

    def _generate_sql_attempt(
        self,
        query: str,
        table_name: str,
        columns: list[str],
        sample_rows: list[list[str]],
        previous_error: str | None,
    ) -> str:
        """Generate SQL query (single attempt)."""
        samples: dict[str, list[str]] = {}
        for i, col in enumerate(columns):
            samples[col] = [row[i] for row in sample_rows if i < len(row)]

        error_context = ""
        if previous_error:
            error_context = (
                f"\nIMPORTANT: Your previous SQL query failed with this error:\n"
                f"{previous_error}\n\n"
                f"Please generate corrected SQL that avoids this error."
            )

        prompt = SQL_PROMPT.format(
            table_name=table_name,
            columns=columns,
            samples=samples,
            question=query,
            max_results=self._config.max_table_results,
        )
        prompt = prompt + error_context

        response = self._chat.chat([{"role": "user", "content": prompt}])
        return self._extract_sql(response)

    def _format_as_markdown(self, cols: list[str], rows: list[list[Any]]) -> str:
        """Format query results as markdown table."""
        if not rows:
            return "(no results)"

        max_results = self._config.max_table_results
        display_rows = rows[:max_results]

        lines = []
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")

        for row in display_rows:
            cells = []
            for val in row:
                s = str(val) if val is not None else ""
                if len(s) > 50:
                    s = s[:47] + "..."
                cells.append(s)
            lines.append("| " + " | ".join(cells) + " |")

        if len(rows) > max_results:
            lines.append(f"\n... and {len(rows) - max_results} more rows")

        return "\n".join(lines)

    def _parse_json_list(self, response: str, fallback: list[str]) -> list[str]:
        """Parse JSON list from LLM response."""
        try:
            text = response.strip()
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

        if "```" in text:
            match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
            else:
                text = text.replace("```sql", "").replace("```", "").strip()

        if not text.upper().startswith("SELECT"):
            match = re.search(r"(SELECT\s+.+)", text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1)

        return text
