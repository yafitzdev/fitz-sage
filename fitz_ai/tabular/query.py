# fitz_ai/tabular/query.py
"""
Table Query Step - Handles schema chunks at query time.

When a schema chunk is retrieved:
1. LLM selects relevant columns (column pruning)
2. Data is loaded from payload (embedded tables) or TableStore (CSV files)
3. In-memory SQLite is built with pruned columns
4. LLM generates SQL query
5. SQL is executed
6. Results are formatted and chunk is augmented

Multi-table join support:
- When multiple schema chunks are retrieved, LLM detects if query needs joins
- All tables are loaded into a single SQLite database with distinct names
- LLM generates SQL with JOINs, inferring join keys from column names

Table storage modes:
- Embedded tables (from documents): Data in chunk metadata as JSON
- Stored tables (CSV files): Data in TableStore, fetched by table_id
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_ai.core.chunk import Chunk
from fitz_ai.engines.fitz_rag.retrieval.steps.base import RetrievalStep

from .models import ParsedTable

if TYPE_CHECKING:
    from fitz_ai.tabular.store.base import TableStore

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

    Supports two table storage modes:
    - Embedded tables: Data in chunk metadata (small tables from documents)
    - Stored tables: Data in TableStore (CSV files, fetched by table_id)

    Args:
        chat: Chat client for LLM calls (column selection + SQL generation).
        max_results: Maximum number of SQL result rows to include (default: 100).
        table_store: Optional TableStore for fetching stored tables (CSV files).
    """

    chat: Any  # Chat client for LLM calls (duck-typed)
    max_results: int = 100
    table_store: "TableStore | None" = field(default=None)

    COLUMN_SELECT_PROMPT = """Given this table schema and question, which columns are needed to answer?

Table columns: {columns}
Question: {question}

Return ONLY a JSON array of column names needed. Example: ["col1", "col2"]
Rules:
- Include columns needed for filtering AND for displaying results
- For "who/which/what" questions, ALWAYS include identifying columns (name, id, title, etc.)
- For numeric comparisons (highest, lowest, average), include the numeric column
- When in doubt, include more columns rather than fewer"""

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
6. For "highest/maximum" use ORDER BY column DESC LIMIT 1
7. For "lowest/minimum" use ORDER BY column ASC LIMIT 1
8. For "who/which" questions, include identifying columns (name, id) in SELECT
9. IMPORTANT: All columns are TEXT type. For numeric operations (MAX, MIN, AVG, SUM, ORDER BY numbers), use CAST(column AS REAL) or CAST(column AS INTEGER)

Return ONLY the SQL query, no explanation."""

    MULTI_TABLE_DETECT_PROMPT = """Does answering this question require combining data from multiple tables?

Available tables and their columns:
{table_schemas}

Question: {question}

Answer ONLY "yes" or "no"."""

    MULTI_TABLE_SQL_PROMPT = """Generate a SQLite query to answer this question using these tables.

Tables:
{table_schemas}

Question: {question}

Rules:
1. Use the exact table names shown above
2. If tables need to be joined, infer join columns from matching column names
3. Use LIMIT {max_results} unless aggregating
4. Column names with spaces need double quotes

Return ONLY the SQL query, no explanation."""

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """
        Process chunks, with multi-table join support.

        When multiple schema chunks are retrieved:
        1. LLM decides if query needs multiple tables
        2. If yes, all tables loaded into one SQLite with JOIN SQL
        3. If no, each table processed independently (existing behavior)

        Args:
            query: User query.
            chunks: Retrieved chunks (may include schema chunks).

        Returns:
            Chunks with table schema chunks augmented with query results.
        """
        table_chunks = [c for c in chunks if c.metadata.get("is_table_schema")]
        regular_chunks = [c for c in chunks if not c.metadata.get("is_table_schema")]

        if not table_chunks:
            return chunks

        # Check if multi-table query needed
        if self._needs_multi_table(query, table_chunks):
            logger.debug(f"Multi-table query detected for {len(table_chunks)} tables")
            result_chunk = self._process_multi_table(query, table_chunks)
            # Put regular chunks first, table results at the end
            return regular_chunks + [result_chunk]

        # Single-table processing (existing logic)
        result_chunks: list[Chunk] = []
        for chunk in table_chunks:
            augmented = self._process_table_chunk(query, chunk)
            result_chunks.append(augmented)

        # Put regular chunks first, table results at the end
        # This prevents table SQL results from dominating the context
        return regular_chunks + result_chunks

    def _load_table_data(self, chunk: Chunk) -> ParsedTable | None:
        """
        Load table data from chunk metadata or TableStore.

        Handles two storage modes:
        - Embedded tables: Data in chunk metadata as JSON (table_data field)
        - Stored tables: Data in TableStore, fetched by table_id

        Args:
            chunk: Table schema chunk.

        Returns:
            ParsedTable if data loaded successfully, None otherwise.
        """
        table_id = chunk.metadata.get("table_id", "unknown")

        # Check if this is a stored table (CSV file in TableStore)
        if chunk.metadata.get("is_stored_table"):
            if self.table_store is None:
                logger.warning(f"Table {table_id} requires TableStore but none provided")
                return None

            stored = self.table_store.retrieve(table_id)
            if stored is None:
                logger.warning(f"Table {table_id} not found in TableStore")
                return None

            return ParsedTable(
                table_id=stored.table_id,
                source_doc=stored.source_file,
                headers=stored.columns,
                rows=stored.rows,
            )

        # Embedded table (data in chunk metadata)
        table_data = chunk.metadata.get("table_data")
        if not table_data:
            logger.warning(f"Table chunk {chunk.id} missing table_data")
            return None

        return ParsedTable.from_json(
            table_data,
            table_id,
            chunk.doc_id,
        )

    def _process_table_chunk(self, query: str, chunk: Chunk) -> Chunk:
        """
        Process a table schema chunk.

        1. Load table data from payload or TableStore
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
            # 1. Load table data
            table = self._load_table_data(chunk)
            if table is None:
                return chunk

            logger.debug(
                f"Processing table {table.table_id}: "
                f"{table.column_count} cols, {table.row_count} rows"
            )

            # 2. LLM selects relevant columns
            needed_cols = self._select_columns(query, table.headers)
            logger.debug(f"Selected columns: {needed_cols}")

            # 3. Prune to needed columns only
            col_indices = [table.headers.index(c) for c in needed_cols if c in table.headers]
            if not col_indices:
                # Fallback: use all columns
                col_indices = list(range(len(table.headers)))
                needed_cols = table.headers

            pruned_headers = [table.headers[i] for i in col_indices]
            pruned_rows = [[row[i] for i in col_indices] for row in table.rows]

            logger.debug(f"Pruned from {table.column_count} to {len(pruned_headers)} columns")

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
                # Return original chunk instead of error-augmented one
                # This prevents SQL errors from confusing the LLM during answer generation
                return chunk

            conn.close()

            # 7. Format results and augment chunk
            return self._augment_chunk(chunk, sql, col_names, results)

        except Exception as e:
            logger.warning(f"Table query failed for {chunk.id}: {e}")
            return chunk  # Return original on failure

    def _needs_multi_table(self, query: str, table_chunks: list[Chunk]) -> bool:
        """
        Detect if query needs data from multiple tables.

        Uses LLM to determine if the query requires joining tables.

        Args:
            query: User query.
            table_chunks: Retrieved table schema chunks.

        Returns:
            True if multi-table query detected.
        """
        if len(table_chunks) < 2:
            return False

        schemas = []
        for chunk in table_chunks:
            name = self._derive_table_name(chunk)
            cols = chunk.metadata.get("columns", [])
            schemas.append(f"- {name}: {', '.join(cols)}")

        prompt = self.MULTI_TABLE_DETECT_PROMPT.format(
            table_schemas="\n".join(schemas),
            question=query,
        )
        response = self.chat.chat([{"role": "user", "content": prompt}])
        return response.strip().lower() == "yes"

    def _derive_table_name(self, chunk: Chunk) -> str:
        """
        Derive SQL-safe table name from chunk metadata.

        Uses doc_id stem, sanitized for SQL identifiers.

        Args:
            chunk: Table schema chunk.

        Returns:
            SQL-safe table name.
        """
        table_id = chunk.metadata.get("table_id", "")

        # Try to get meaningful name from doc_id
        doc_id = chunk.doc_id or ""
        base_name = Path(doc_id).stem  # "report.csv" → "report"

        # Sanitize for SQL
        name = base_name or f"table_{table_id[:8]}"
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

        # Handle leading digit
        if name and name[0].isdigit():
            name = f"t_{name}"

        return name or "data"

    def _process_multi_table(self, query: str, table_chunks: list[Chunk]) -> Chunk:
        """
        Process multiple tables with JOIN support.

        Loads all tables into one SQLite database with distinct names,
        then generates and executes JOIN SQL.

        Args:
            query: User query.
            table_chunks: Retrieved table schema chunks.

        Returns:
            Merged result chunk.
        """
        conn = sqlite3.connect(":memory:")
        tables_info: list[tuple[str, list[str], list[list[str]]]] = []

        try:
            # Load each table with its own name
            for chunk in table_chunks:
                table = self._load_table_data(chunk)
                if table is None:
                    continue

                table_name = self._derive_table_name(chunk)
                self._create_and_populate_named(conn, table_name, table.headers, table.rows)
                tables_info.append((table_name, table.headers, table.rows[:3]))

                logger.debug(
                    f"Loaded table {table_name}: {len(table.headers)} cols, {len(table.rows)} rows"
                )

            if not tables_info:
                return self._process_table_chunk(query, table_chunks[0])

            # Generate multi-table SQL
            sql = self._generate_multi_table_sql(query, tables_info)
            logger.debug(f"Generated multi-table SQL: {sql}")

            # Execute
            try:
                cursor = conn.execute(sql)
                results = cursor.fetchall()
                col_names = [d[0] for d in cursor.description]
            except sqlite3.Error as e:
                logger.warning(f"Multi-table SQL failed: {e}, falling back to single-table")
                conn.close()
                return self._process_table_chunk(query, table_chunks[0])

            conn.close()

            # Create merged result chunk
            table_names = [t[0] for t in tables_info]
            return self._create_multi_table_result_chunk(
                table_chunks, table_names, sql, col_names, results
            )

        except Exception as e:
            logger.warning(f"Multi-table query failed: {e}, falling back to single-table")
            conn.close()
            return self._process_table_chunk(query, table_chunks[0])

    def _generate_multi_table_sql(
        self,
        query: str,
        tables: list[tuple[str, list[str], list[list[str]]]],
        max_retries: int = 2,
    ) -> str:
        """
        Generate SQL that may JOIN multiple tables, with validation and retry.

        Args:
            query: User query.
            tables: List of (table_name, headers, sample_rows).
            max_retries: Maximum retry attempts (default: 2).

        Returns:
            Validated SQL query string.
        """
        previous_error = None

        for attempt in range(max_retries + 1):
            # Generate SQL (with error feedback on retries)
            sql = self._generate_multi_table_sql_attempt(query, tables, previous_error)

            # Validate by test execution on sample data
            try:
                test_conn = sqlite3.connect(":memory:")
                for table_name, headers, sample_rows in tables:
                    self._create_and_populate_named(test_conn, table_name, headers, sample_rows[:3])
                test_conn.execute(sql)  # Test query
                test_conn.close()

                logger.debug(
                    f"Multi-table SQL validated successfully{' (retry '+str(attempt)+')' if attempt > 0 else ''}"
                )
                return sql  # Success!

            except sqlite3.Error as e:
                previous_error = str(e)
                logger.warning(
                    f"Multi-table SQL validation failed (attempt {attempt+1}/{max_retries+1}): {previous_error}"
                )

                if attempt == max_retries:
                    logger.error(
                        f"Multi-table SQL generation failed after {max_retries+1} attempts"
                    )
                    return sql

        return sql  # Fallback (should not reach here)

    def _generate_multi_table_sql_attempt(
        self,
        query: str,
        tables: list[tuple[str, list[str], list[list[str]]]],
        previous_error: str | None,
    ) -> str:
        """Generate multi-table SQL query (single attempt)."""
        schemas = []
        for name, headers, samples in tables:
            sample_vals = {}
            for i, h in enumerate(headers):
                sample_vals[h] = [row[i] for row in samples if i < len(row)]
            schemas.append(f"Table: {name}\nColumns: {headers}\nSamples: {sample_vals}")

        # Add error feedback to prompt if retrying
        error_context = ""
        if previous_error:
            error_context = f"""
IMPORTANT: Your previous SQL query failed with this error:
{previous_error}

Please generate corrected SQL that avoids this error."""

        prompt = self.MULTI_TABLE_SQL_PROMPT.format(
            table_schemas="\n\n".join(schemas),
            question=query,
            max_results=self.max_results,
        )
        prompt = prompt + error_context

        response = self.chat.chat([{"role": "user", "content": prompt}])
        return self._extract_sql(response)

    def _create_and_populate_named(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        headers: list[str],
        rows: list[list[str]],
    ) -> None:
        """
        Create named table and insert rows.

        Args:
            conn: SQLite connection.
            table_name: Table name to use.
            headers: Column names.
            rows: Row data.
        """
        cols_def = ", ".join(f'"{h}" TEXT' for h in headers)
        conn.execute(f'CREATE TABLE "{table_name}" ({cols_def})')

        placeholders = ", ".join("?" * len(headers))
        conn.executemany(f'INSERT INTO "{table_name}" VALUES ({placeholders})', rows)
        conn.commit()

    def _create_multi_table_result_chunk(
        self,
        source_chunks: list[Chunk],
        table_names: list[str],
        sql: str,
        col_names: list[str],
        results: list[tuple],
    ) -> Chunk:
        """
        Create merged result chunk from multi-table query.

        Args:
            source_chunks: Original table schema chunks.
            table_names: Names of tables in the query.
            sql: Executed SQL query.
            col_names: Result column names.
            results: Query results.

        Returns:
            New chunk with combined results.
        """
        results_md = self._format_as_markdown(col_names, results)

        content = f"""Multi-table query result
Source tables: {", ".join(table_names)}

SQL executed: {sql}
Results ({len(results)} rows):
{results_md}"""

        # Combine table IDs for unique chunk ID
        combined_id = "_".join(c.metadata.get("table_id", "unknown")[:4] for c in source_chunks)

        return Chunk(
            id=f"multi_table_{combined_id}",
            doc_id=source_chunks[0].doc_id,
            content=content,
            chunk_index=0,
            metadata={
                "is_multi_table_result": True,
                "source_tables": table_names,
                "sql_executed": sql,
                "result_count": len(results),
            },
        )

    def _select_columns(self, query: str, columns: list[str]) -> list[str]:
        """Use LLM to select relevant columns."""
        prompt = self.COLUMN_SELECT_PROMPT.format(
            columns=columns,
            question=query,
        )
        response = self.chat.chat([{"role": "user", "content": prompt}])
        return self._parse_json_list(response, fallback=columns)

    def _generate_sql(
        self, query: str, columns: list[str], sample_rows: list[list[str]], max_retries: int = 2
    ) -> str:
        """
        Use LLM to generate SQL query with validation and retry.

        Generates SQL, tests it on sample data, and retries with error feedback
        if execution fails. This handles LLM hallucinations like invalid functions.

        Args:
            query: User query.
            columns: Table columns.
            sample_rows: Sample data rows.
            max_retries: Maximum retry attempts (default: 2).

        Returns:
            Validated SQL query string.
        """
        previous_error = None

        for attempt in range(max_retries + 1):
            # Generate SQL (with error feedback on retries)
            sql = self._generate_sql_attempt(query, columns, sample_rows, previous_error)

            # Validate by test execution on sample data
            try:
                test_conn = sqlite3.connect(":memory:")
                self._create_and_populate(test_conn, columns, sample_rows[:3])
                test_conn.execute(sql)  # Test query
                test_conn.close()

                logger.debug(
                    f"SQL validated successfully{' (retry '+str(attempt)+')' if attempt > 0 else ''}"
                )
                return sql  # Success!

            except sqlite3.Error as e:
                previous_error = str(e)
                logger.warning(
                    f"SQL validation failed (attempt {attempt+1}/{max_retries+1}): {previous_error}"
                )

                if attempt == max_retries:
                    # Give up, return the SQL anyway (will fail with proper error later)
                    logger.error(f"SQL generation failed after {max_retries+1} attempts")
                    return sql

        return sql  # Fallback (should not reach here)

    def _generate_sql_attempt(
        self,
        query: str,
        columns: list[str],
        sample_rows: list[list[str]],
        previous_error: str | None,
    ) -> str:
        """Generate SQL query (single attempt)."""
        # Format samples as dict for prompt
        samples: dict[str, list[str]] = {}
        for i, col in enumerate(columns):
            samples[col] = [row[i] for row in sample_rows if i < len(row)]

        # Add error feedback to prompt if retrying
        error_context = ""
        if previous_error:
            error_context = f"""
IMPORTANT: Your previous SQL query failed with this error:
{previous_error}

Please generate corrected SQL that avoids this error."""

        prompt = self.SQL_PROMPT.format(
            columns=columns,
            samples=samples,
            question=query,
            max_results=self.max_results,
        )
        prompt = prompt + error_context

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

    def _augment_chunk(self, chunk: Chunk, sql: str, cols: list[str], rows: list[tuple]) -> Chunk:
        """Create augmented chunk with query results."""
        results_md = self._format_as_markdown(cols, rows)

        # Get table metadata for context
        row_count = chunk.metadata.get("row_count", "unknown")
        columns = chunk.metadata.get("columns", [])

        content = f"""Table: {chunk.doc_id}
Columns: {", ".join(columns) if columns else "see below"}
Total rows: {row_count}

--- SQL Query Results (from FULL dataset, not just sample) ---
Query: {sql}
Results ({len(rows)} rows):
{results_md}

Note: These results are computed from all {row_count} rows in the table. Use these results to answer the question."""

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
