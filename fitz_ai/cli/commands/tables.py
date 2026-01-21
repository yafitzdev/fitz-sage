# fitz_ai/cli/commands/tables.py
"""
Structured table management commands.

Commands:
    fitz tables                         - List all tables in collection
    fitz tables info <table>            - Show table schema details
    fitz tables delete <table>          - Delete a table
    fitz ingest-table <file>            - Ingest CSV/Excel as structured table
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

import typer

from fitz_ai.cli.ui import RICH, Table, console, ui
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

app = typer.Typer(
    name="tables",
    help="Manage structured tables for SQL-like queries.",
    no_args_is_help=True,
)


def _get_collection(collection: Optional[str]) -> str:
    """Get collection name, using default from context if not specified."""
    if collection is None:
        from fitz_ai.cli.context import CLIContext

        ctx = CLIContext.load()
        return ctx.retrieval_collection
    return collection


def _get_schema_store(collection: str):
    """Get schema store for collection."""
    from fitz_ai.cli.context import CLIContext
    from fitz_ai.structured.schema import SchemaStore

    ctx = CLIContext.load()
    vector_client = ctx.require_vector_db_client()
    embedding_client = ctx.require_embedding_client()

    return SchemaStore(
        vector_db=vector_client,
        embedding=embedding_client,
        base_collection=collection,
    )


def _display_tables_list(tables: list[dict[str, Any]]) -> None:
    """Display list of tables."""
    if RICH:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Table", style="cyan")
        table.add_column("Columns", justify="right")
        table.add_column("Rows", justify="right")
        table.add_column("Primary Key", style="dim")

        for t in tables:
            table.add_row(
                t["name"],
                str(t["column_count"]),
                str(t["row_count"]),
                t["primary_key"],
            )

        console.print(table)
    else:
        print()
        for t in tables:
            print(f"  {t['name']}")
            print(f"    Columns: {t['column_count']}, Rows: {t['row_count']}")
            print(f"    Primary Key: {t['primary_key']}")
            print()


def _display_table_schema(schema: Any) -> None:
    """Display detailed table schema."""
    if RICH:
        from rich.panel import Panel

        # Table info header
        info_table = Table(show_header=False, box=None, padding=(0, 2))
        info_table.add_column("Key", style="bold")
        info_table.add_column("Value", style="cyan")
        info_table.add_row("Table", schema.table_name)
        info_table.add_row("Rows", str(schema.row_count))
        info_table.add_row("Primary Key", schema.primary_key)
        info_table.add_row("Version", schema.version[:8] if schema.version else "-")

        console.print(Panel(info_table, title="[bold]Table Info[/bold]", border_style="blue"))
        print()

        # Columns table
        col_table = Table(show_header=True, header_style="bold", title="Columns")
        col_table.add_column("#", style="dim", width=3)
        col_table.add_column("Name", style="cyan")
        col_table.add_column("Type")
        col_table.add_column("Indexed", justify="center")
        col_table.add_column("Nullable", justify="center")

        for i, col in enumerate(schema.columns, 1):
            indexed = "[green]Yes[/green]" if col.indexed else "[dim]No[/dim]"
            nullable = "[yellow]Yes[/yellow]" if col.nullable else "[dim]No[/dim]"
            col_table.add_row(str(i), col.name, col.type, indexed, nullable)

        console.print(col_table)
    else:
        print(f"\nTable: {schema.table_name}")
        print(f"Rows: {schema.row_count}")
        print(f"Primary Key: {schema.primary_key}")
        print(f"Version: {schema.version[:8] if schema.version else '-'}")
        print("\nColumns:")
        print("-" * 60)
        for i, col in enumerate(schema.columns, 1):
            indexed = "Yes" if col.indexed else "No"
            nullable = "Yes" if col.nullable else "No"
            print(f"  {i}. {col.name} ({col.type}) - Indexed: {indexed}, Nullable: {nullable}")


@app.command("list")
def list_tables(
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection name. Uses default if not specified.",
    ),
) -> None:
    """List all structured tables in the collection."""
    collection = _get_collection(collection)

    ui.info(f"Collection: {collection}")
    print()

    try:
        store = _get_schema_store(collection)
        schemas = store.get_all_schemas()
    except Exception as e:
        ui.error(f"Failed to load schemas: {e}")
        raise typer.Exit(1)

    if not schemas:
        ui.info("No tables found in this collection.")
        ui.info("Use 'fitz ingest-table' to add structured data.")
        return

    tables = []
    for schema in schemas:
        tables.append(
            {
                "name": schema.table_name,
                "column_count": len(schema.columns),
                "row_count": schema.row_count,
                "primary_key": schema.primary_key,
            }
        )

    ui.success(f"Found {len(tables)} table(s)")
    print()
    _display_tables_list(tables)


@app.command("info")
def table_info(
    table_name: str = typer.Argument(..., help="Table name to show info for"),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection name. Uses default if not specified.",
    ),
) -> None:
    """Show detailed schema for a specific table."""
    collection = _get_collection(collection)

    try:
        store = _get_schema_store(collection)
        schema = store.get_schema(table_name)
    except Exception as e:
        ui.error(f"Failed to load schema: {e}")
        raise typer.Exit(1)

    if schema is None:
        ui.error(f"Table '{table_name}' not found in collection '{collection}'.")
        raise typer.Exit(1)

    _display_table_schema(schema)


@app.command("delete")
def delete_table(
    table_name: str = typer.Argument(..., help="Table name to delete"),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection name. Uses default if not specified.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt.",
    ),
) -> None:
    """Delete a structured table and its data."""
    from fitz_ai.cli.context import CLIContext
    from fitz_ai.structured.constants import get_tables_collection

    collection = _get_collection(collection)

    try:
        store = _get_schema_store(collection)
        schema = store.get_schema(table_name)
    except Exception as e:
        ui.error(f"Failed to load schema: {e}")
        raise typer.Exit(1)

    if schema is None:
        ui.error(f"Table '{table_name}' not found.")
        raise typer.Exit(1)

    if not force:
        ui.warning(f"This will delete table '{table_name}' with {schema.row_count} rows.")
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            ui.info("Cancelled.")
            return

    ctx = CLIContext.load()
    vector_client = ctx.require_vector_db_client()

    # Delete rows from __tables collection
    tables_collection = get_tables_collection(collection)
    try:
        # Build filter for this table's rows
        filter_condition = {"must": [{"key": "__table", "match": {"value": table_name}}]}

        # Scroll to find all row IDs
        row_ids = []
        offset = 0
        while True:
            records, next_offset = vector_client.scroll(
                collection_name=tables_collection,
                limit=100,
                offset=offset,
                scroll_filter=filter_condition,
                with_payload=False,
            )
            if not records:
                break
            for record in records:
                record_id = getattr(record, "id", None) or record.get("id")
                if record_id:
                    row_ids.append(record_id)
            if next_offset is None or len(records) < 100:
                break
            offset = next_offset

        if row_ids:
            vector_client.delete(tables_collection, {"points": row_ids})
            ui.info(f"Deleted {len(row_ids)} rows from tables collection")
    except Exception as e:
        logger.warning(f"Failed to delete rows: {e}")

    # Delete derived sentences for this table
    try:
        from fitz_ai.structured.derived import DerivedStore

        embedding_client = ctx.require_embedding_client()
        derived_store = DerivedStore(vector_client, embedding_client, collection)
        deleted = derived_store.invalidate(table_name)
        if deleted:
            ui.info(f"Deleted {deleted} derived sentences")
    except Exception as e:
        logger.warning(f"Failed to delete derived sentences: {e}")

    # Delete schema
    try:
        store.delete_schema(table_name)
        ui.success(f"Deleted table '{table_name}' from collection '{collection}'")
    except Exception as e:
        ui.error(f"Failed to delete schema: {e}")
        raise typer.Exit(1)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read CSV file and return headers and rows."""
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    return headers, rows


def _read_excel(path: Path, sheet: Optional[str] = None) -> tuple[list[str], list[dict[str, Any]]]:
    """Read Excel file and return headers and rows."""
    try:
        import pandas as pd
    except ImportError:
        raise typer.Exit("Excel support requires pandas. Install with: pip install pandas openpyxl")

    if sheet:
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_excel(path)

    headers = list(df.columns)
    rows = df.to_dict("records")
    return headers, rows


def ingest_table_command(
    source: Path = typer.Argument(..., help="Path to CSV or Excel file"),
    table_name: Optional[str] = typer.Option(
        None,
        "--table",
        "-t",
        help="Table name. Defaults to filename without extension.",
    ),
    primary_key: Optional[str] = typer.Option(
        None,
        "--pk",
        help="Primary key column. Auto-detected if not specified.",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection name. Uses default if not specified.",
    ),
    sheet: Optional[str] = typer.Option(
        None,
        "--sheet",
        "-s",
        help="Excel sheet name (for .xlsx files).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing table.",
    ),
) -> None:
    """
    Ingest a CSV or Excel file as a structured table.

    The table will be stored alongside semantic chunks and can be queried
    using natural language that gets translated to SQL.

    Examples:
        fitz ingest-table employees.csv --pk employee_id
        fitz ingest-table sales.xlsx --table quarterly_sales --sheet Q1
    """
    from fitz_ai.cli.context import CLIContext
    from fitz_ai.structured.ingestion import (
        MissingPrimaryKeyError,
        StructuredIngester,
        TableTooLargeError,
    )

    # Validate source file
    if not source.exists():
        ui.error(f"File not found: {source}")
        raise typer.Exit(1)

    suffix = source.suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls"):
        ui.error(f"Unsupported file type: {suffix}. Use .csv, .xlsx, or .xls")
        raise typer.Exit(1)

    # Default table name from filename
    if table_name is None:
        table_name = source.stem.lower().replace(" ", "_").replace("-", "_")

    collection = _get_collection(collection)

    ui.header("Ingest Table", f"Loading {source.name}")

    # Read file
    ui.info(f"Reading {source.name}...")
    try:
        if suffix == ".csv":
            headers, rows = _read_csv(source)
        else:
            headers, rows = _read_excel(source, sheet)
    except Exception as e:
        ui.error(f"Failed to read file: {e}")
        raise typer.Exit(1)

    if not rows:
        ui.error("File contains no data rows.")
        raise typer.Exit(1)

    ui.success(f"Read {len(rows)} rows, {len(headers)} columns")

    # Show preview
    print()
    ui.info("Column preview:")
    if RICH:
        preview_table = Table(show_header=True, header_style="bold")
        preview_table.add_column("Column", style="cyan")
        preview_table.add_column("Sample Value", style="dim")

        for col in headers[:10]:  # Limit to 10 columns
            sample = str(rows[0].get(col, ""))[:50] if rows else ""
            preview_table.add_row(col, sample)

        if len(headers) > 10:
            preview_table.add_row(f"... +{len(headers) - 10} more", "")

        console.print(preview_table)
    else:
        for col in headers[:10]:
            sample = str(rows[0].get(col, ""))[:30] if rows else ""
            print(f"  {col}: {sample}")
        if len(headers) > 10:
            print(f"  ... +{len(headers) - 10} more columns")

    # Auto-detect primary key if not specified
    if primary_key is None:
        # Look for common PK patterns
        pk_patterns = ["id", "_id", "key", "_key", "pk", "primary"]
        for col in headers:
            col_lower = col.lower()
            if any(p in col_lower for p in pk_patterns):
                # Check if values are unique
                values = [row.get(col) for row in rows]
                if len(set(values)) == len(values):
                    primary_key = col
                    ui.info(f"Auto-detected primary key: {primary_key}")
                    break

        if primary_key is None:
            # Use first column with unique values
            for col in headers:
                values = [row.get(col) for row in rows]
                if len(set(values)) == len(values) and all(
                    v is not None and v != "" for v in values
                ):
                    primary_key = col
                    ui.info(f"Using first unique column as primary key: {primary_key}")
                    break

        if primary_key is None:
            ui.error("Could not auto-detect primary key. Please specify with --pk")
            raise typer.Exit(1)

    # Validate primary key column exists
    if primary_key not in headers:
        ui.error(f"Primary key column '{primary_key}' not found in file.")
        ui.info(f"Available columns: {', '.join(headers[:5])}{'...' if len(headers) > 5 else ''}")
        raise typer.Exit(1)

    print()

    # Check if table exists
    try:
        store = _get_schema_store(collection)
        existing = store.get_schema(table_name)
        if existing and not force:
            ui.warning(f"Table '{table_name}' already exists with {existing.row_count} rows.")
            overwrite = typer.confirm("Overwrite?")
            if not overwrite:
                ui.info("Cancelled.")
                return
    except Exception:
        pass  # No existing table

    # Create ingester and ingest
    ctx = CLIContext.load()
    vector_client = ctx.require_vector_db_client()
    embedding_client = ctx.require_embedding_client()

    ingester = StructuredIngester(
        vector_db=vector_client,
        embedding=embedding_client,
        base_collection=collection,
    )

    ui.info(f"Ingesting table '{table_name}' to collection '{collection}'...")

    try:
        schema = ingester.ingest_table(
            table_name=table_name,
            rows=rows,
            primary_key=primary_key,
        )
    except TableTooLargeError as e:
        ui.error(f"Table too large: {e}")
        ui.info(
            "Maximum supported rows: 10,000. Consider using a proper database for larger datasets."
        )
        raise typer.Exit(1)
    except MissingPrimaryKeyError as e:
        ui.error(f"Primary key error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        ui.error(f"Ingestion failed: {e}")
        logger.exception("Ingestion error")
        raise typer.Exit(1)

    print()
    ui.success(f"Ingested table '{table_name}' with {len(rows)} rows")

    # Show indexed columns
    indexed = [col.name for col in schema.columns if col.indexed]
    if indexed:
        ui.info(f"Indexed columns: {', '.join(indexed)}")

    print()
    ui.info("You can now query this table with natural language:")
    ui.info(f'  fitz query "How many rows are in {table_name}?" -c {collection}')


# Export for main CLI registration
def command() -> None:
    """Run the tables command group."""
    app()
