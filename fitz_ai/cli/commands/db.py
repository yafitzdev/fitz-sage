# fitz_ai/cli/commands/db.py
"""
Database inspection command.

Usage:
    fitz db                           # Interactive - list and select collection
    fitz db my_collection             # Inspect specific collection
    fitz db my_collection --search "query"  # Search in collection
    fitz db my_collection --export out.jsonl  # Export collection
    fitz db my_collection --delete    # Delete a collection
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Any

import typer

from fitz_ai.core.config import load_config_dict, ConfigNotFoundError
from fitz_ai.core.paths import FitzPaths
from fitz_ai.vector_db.registry import get_vector_db_plugin
from fitz_ai.logging.logger import get_logger
from fitz_ai.cli.ui import ui, console, RICH, Panel, Table

logger = get_logger(__name__)


# =============================================================================
# Config Loading
# =============================================================================


def _load_config_safe() -> dict:
    """Load config or return empty dict."""
    try:
        return load_config_dict(FitzPaths.config())
    except (ConfigNotFoundError, FileNotFoundError):
        return {}
    except Exception:
        return {}


# =============================================================================
# Display Functions
# =============================================================================


def _list_collections(vdb, vector_db_name: str) -> list[str]:
    """List all collections with stats."""
    collections = vdb.list_collections()

    if not collections:
        ui.info("No collections found.")
        ui.info("Run 'fitz ingest' to create one.")
        return []

    print()
    if RICH:
        table = Table(title="Collections")
        table.add_column("Name", style="cyan")
        table.add_column("Chunks", justify="right")
        table.add_column("Status", style="dim")

        for name in sorted(collections):
            try:
                stats = vdb.get_collection_stats(name)
                points = stats.get("points_count", "?")
                if isinstance(points, int):
                    points = f"{points:,}"
                status = stats.get("status", "ready")
            except Exception:
                points = "?"
                status = "?"
            table.add_row(name, str(points), str(status))

        console.print(table)
    else:
        print("Collections:")
        print("-" * 40)
        for name in sorted(collections):
            try:
                stats = vdb.get_collection_stats(name)
                points = stats.get("points_count", "?")
            except Exception:
                points = "?"
            print(f"  {name}: {points} chunks")

    print()
    return collections


def _show_collection_details(vdb, collection: str, num_samples: int = 3) -> None:
    """Show detailed stats and sample chunks for a collection."""
    try:
        stats = vdb.get_collection_stats(collection)
    except Exception as e:
        ui.error(f"Failed to get stats: {e}")
        return

    # Display stats
    print()
    if RICH:
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="dim")
        table.add_column("Value", style="bold")

        points = stats.get("points_count", "N/A")
        if isinstance(points, int):
            points = f"{points:,}"
        table.add_row("Chunks", str(points))

        vectors = stats.get("vectors_count", stats.get("points_count", "N/A"))
        if isinstance(vectors, int):
            vectors = f"{vectors:,}"
        table.add_row("Vectors", str(vectors))

        table.add_row("Status", str(stats.get("status", "ready")))

        if "vector_size" in stats:
            table.add_row("Dimension", str(stats["vector_size"]))

        console.print(table)
    else:
        print(f"  Chunks:  {stats.get('points_count', 'N/A')}")
        print(f"  Vectors: {stats.get('vectors_count', 'N/A')}")
        print(f"  Status:  {stats.get('status', 'ready')}")

    # Show sample chunks
    if num_samples > 0:
        _show_sample_chunks(vdb, collection, num_samples)


def _show_sample_chunks(vdb, collection: str, num_samples: int) -> None:
    """Show sample chunks from collection."""
    print()

    try:
        # Try to access underlying client for scroll/sample
        client = getattr(vdb, "_client", None) or getattr(vdb, "client", None)
        if not client:
            ui.info("Cannot fetch samples (no direct client access)")
            return

        # Try Qdrant-style scroll
        try:
            records, _ = client.scroll(
                collection_name=collection,
                limit=num_samples,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            ui.info("Cannot fetch samples (scroll not supported)")
            return

        if not records:
            ui.info("No chunks found in collection.")
            return

        if RICH:
            console.print("[bold]Sample Chunks:[/bold]")
            for i, record in enumerate(records, 1):
                payload = getattr(record, "payload", {}) or {}
                content = payload.get("content", payload.get("text", ""))
                doc_id = payload.get("doc_id", payload.get("source_file", "?"))
                chunk_idx = payload.get("chunk_index", "?")

                # Truncate content
                if len(content) > 150:
                    content = content[:150] + "..."
                content = content.replace("\n", " ")

                console.print()
                console.print(f"  [dim]#{i}[/dim] [cyan]{doc_id}[/cyan] [dim](chunk {chunk_idx})[/dim]")
                console.print(f"      [dim]{content}[/dim]")
        else:
            print("Sample Chunks:")
            print("-" * 40)
            for i, record in enumerate(records, 1):
                payload = getattr(record, "payload", {}) or {}
                content = payload.get("content", payload.get("text", ""))
                doc_id = payload.get("doc_id", "?")

                if len(content) > 150:
                    content = content[:150] + "..."
                content = content.replace("\n", " ")

                print(f"  #{i} {doc_id}")
                print(f"      {content}")
                print()

    except Exception as e:
        logger.debug(f"Failed to fetch samples: {e}")
        ui.info(f"Could not fetch samples: {e}")


def _delete_collection(vdb, collection: str) -> bool:
    """Delete a collection after confirmation."""
    try:
        stats = vdb.get_collection_stats(collection)
        points = stats.get("points_count", 0)
    except Exception:
        points = "?"

    ui.warning(f"Collection '{collection}' has {points} chunks.")

    if not ui.prompt_confirm(f"Delete collection '{collection}'? This cannot be undone", default=False):
        ui.info("Cancelled.")
        return False

    try:
        vdb.delete_collection(collection)
        ui.success(f"Deleted collection '{collection}'")
        return True
    except Exception as e:
        ui.error(f"Failed to delete: {e}")
        return False


# =============================================================================
# Search Functions
# =============================================================================


def _search_collection(vdb, collection: str, query: str, config: dict, limit: int = 10) -> None:
    """Search collection using vector similarity."""
    from fitz_ai.llm.registry import get_llm_plugin

    # Get embedder from config
    embedding_plugin = config.get("embedding", {}).get("plugin_name")
    if not embedding_plugin:
        ui.error("No embedding plugin configured. Run 'fitz init' first.")
        return

    try:
        embedder = get_llm_plugin(plugin_type="embedding", plugin_name=embedding_plugin)
    except Exception as e:
        ui.error(f"Failed to load embedder: {e}")
        return

    # Generate query embedding
    ui.info(f"Searching for: \"{query}\"")

    try:
        query_vector = embedder.embed(query)
    except Exception as e:
        ui.error(f"Failed to embed query: {e}")
        return

    # Search
    try:
        results = vdb.search(
            collection=collection,
            query_vector=query_vector,
            top_k=limit,
        )
    except Exception as e:
        ui.error(f"Search failed: {e}")
        return

    if not results:
        ui.info("No results found.")
        return

    # Display results
    print()
    if RICH:
        console.print(f"[bold]Found {len(results)} results:[/bold]")
        for i, result in enumerate(results, 1):
            score = getattr(result, 'score', None)
            payload = getattr(result, 'payload', {}) or {}
            content = payload.get("content", payload.get("text", ""))
            doc_id = payload.get("doc_id", payload.get("source_file", "?"))

            # Truncate content
            if len(content) > 200:
                content = content[:200] + "..."
            content = content.replace("\n", " ")

            score_str = f" [dim](score: {score:.3f})[/dim]" if score is not None else ""
            console.print()
            console.print(f"  [green]#{i}[/green] [cyan]{doc_id}[/cyan]{score_str}")
            console.print(f"      [dim]{content}[/dim]")
    else:
        print(f"Found {len(results)} results:")
        print("-" * 40)
        for i, result in enumerate(results, 1):
            score = getattr(result, 'score', None)
            payload = getattr(result, 'payload', {}) or {}
            content = payload.get("content", payload.get("text", ""))
            doc_id = payload.get("doc_id", "?")

            if len(content) > 200:
                content = content[:200] + "..."
            content = content.replace("\n", " ")

            score_str = f" (score: {score:.3f})" if score is not None else ""
            print(f"  #{i} {doc_id}{score_str}")
            print(f"      {content}")
            print()


# =============================================================================
# Export Functions
# =============================================================================


def _get_all_records(vdb, collection: str) -> List[Any]:
    """Scroll through all records in a collection."""
    client = getattr(vdb, "_client", None) or getattr(vdb, "client", None)
    if not client:
        raise RuntimeError("Cannot access vector DB client for export")

    all_records = []
    offset = None
    batch_size = 100

    while True:
        try:
            records, next_offset = client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to scroll: {e}")

        if not records:
            break

        all_records.extend(records)

        if next_offset is None:
            break
        offset = next_offset

    return all_records


def _export_collection(vdb, collection: str, output_path: Path, format: str = "jsonl") -> None:
    """Export collection to file."""

    # Get stats first
    try:
        stats = vdb.get_collection_stats(collection)
        total = stats.get("points_count", "?")
    except Exception:
        total = "?"

    ui.info(f"Exporting {total} chunks from '{collection}'...")

    # Get all records
    try:
        records = _get_all_records(vdb, collection)
    except Exception as e:
        ui.error(f"Failed to read collection: {e}")
        return

    if not records:
        ui.warning("No records to export.")
        return

    # Export based on format
    try:
        if format == "jsonl":
            _export_jsonl(records, output_path)
        elif format == "csv":
            _export_csv(records, output_path)
        elif format == "json":
            _export_json(records, output_path)
        else:
            ui.error(f"Unknown format: {format}")
            return
    except Exception as e:
        ui.error(f"Failed to write file: {e}")
        return

    ui.success(f"Exported {len(records)} chunks to {output_path}")


def _export_jsonl(records: List[Any], output_path: Path) -> None:
    """Export as JSON Lines."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            payload = getattr(record, "payload", {}) or {}
            record_id = getattr(record, "id", None)

            row = {
                "id": str(record_id) if record_id else None,
                "doc_id": payload.get("doc_id", payload.get("source_file")),
                "chunk_index": payload.get("chunk_index"),
                "content": payload.get("content", payload.get("text", "")),
                "metadata": {k: v for k, v in payload.items()
                             if k not in ("doc_id", "source_file", "chunk_index", "content", "text")},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _export_csv(records: List[Any], output_path: Path) -> None:
    """Export as CSV."""
    import csv

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "doc_id", "chunk_index", "content"])

        for record in records:
            payload = getattr(record, "payload", {}) or {}
            record_id = getattr(record, "id", None)

            writer.writerow([
                str(record_id) if record_id else "",
                payload.get("doc_id", payload.get("source_file", "")),
                payload.get("chunk_index", ""),
                payload.get("content", payload.get("text", "")),
            ])


def _export_json(records: List[Any], output_path: Path) -> None:
    """Export as JSON array."""
    rows = []
    for record in records:
        payload = getattr(record, "payload", {}) or {}
        record_id = getattr(record, "id", None)

        rows.append({
            "id": str(record_id) if record_id else None,
            "doc_id": payload.get("doc_id", payload.get("source_file")),
            "chunk_index": payload.get("chunk_index"),
            "content": payload.get("content", payload.get("text", "")),
            "metadata": {k: v for k, v in payload.items()
                         if k not in ("doc_id", "source_file", "chunk_index", "content", "text")},
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


# =============================================================================
# Main Command
# =============================================================================


def command(
        collection: Optional[str] = typer.Argument(
            None,
            help="Collection to inspect (will prompt if not provided).",
        ),
        samples: int = typer.Option(
            3,
            "--samples",
            "-n",
            help="Number of sample chunks to show.",
        ),
        search: Optional[str] = typer.Option(
            None,
            "--search",
            "-s",
            help="Search query to find similar chunks.",
        ),
        export: Optional[Path] = typer.Option(
            None,
            "--export",
            "-e",
            help="Export collection to file (jsonl, csv, or json).",
        ),
        export_format: str = typer.Option(
            "jsonl",
            "--format",
            "-f",
            help="Export format: jsonl, csv, or json.",
        ),
        delete: bool = typer.Option(
            False,
            "--delete",
            "-d",
            help="Delete the specified collection.",
        ),
        list_only: bool = typer.Option(
            False,
            "--list",
            "-l",
            help="Only list collections, don't inspect.",
        ),
) -> None:
    """
    Inspect vector database collections.

    Run without arguments for interactive mode:
        fitz db

    Or specify a collection:
        fitz db my_collection

    Search in a collection:
        fitz db my_collection --search "machine learning"

    Export a collection:
        fitz db my_collection --export backup.jsonl
        fitz db my_collection --export data.csv --format csv

    Delete a collection:
        fitz db my_collection --delete
    """
    # =========================================================================
    # Load config to get vector DB plugin
    # =========================================================================

    config = _load_config_safe()
    vector_db_plugin = config.get("vector_db", {}).get("plugin_name", "qdrant")

    # =========================================================================
    # Header
    # =========================================================================

    ui.header("Fitz Database")
    ui.info(f"Vector DB: {vector_db_plugin}")

    # =========================================================================
    # Get vector DB client
    # =========================================================================

    try:
        vdb = get_vector_db_plugin(vector_db_plugin)
    except Exception as e:
        ui.error(f"Failed to connect to vector DB: {e}")
        raise typer.Exit(1)

    # =========================================================================
    # List only mode
    # =========================================================================

    if list_only:
        _list_collections(vdb, vector_db_plugin)
        return

    # =========================================================================
    # Require collection for search/export/delete
    # =========================================================================

    collections = vdb.list_collections()

    if search or export or delete:
        if not collection:
            ui.error("Collection required for this operation.")
            ui.info("Usage: fitz db <collection> --search/--export/--delete")
            raise typer.Exit(1)

        if collection not in collections:
            ui.error(f"Collection '{collection}' not found.")
            raise typer.Exit(1)

    # =========================================================================
    # Delete mode
    # =========================================================================

    if delete:
        _delete_collection(vdb, collection)
        return

    # =========================================================================
    # Search mode
    # =========================================================================

    if search:
        _search_collection(vdb, collection, search, config)
        return

    # =========================================================================
    # Export mode
    # =========================================================================

    if export:
        # Auto-detect format from extension if not specified
        if export_format == "jsonl" and export.suffix:
            suffix = export.suffix.lower()
            if suffix == ".csv":
                export_format = "csv"
            elif suffix == ".json":
                export_format = "json"

        _export_collection(vdb, collection, export, export_format)
        return

    # =========================================================================
    # Interactive mode - select collection if not provided
    # =========================================================================

    _list_collections(vdb, vector_db_plugin)

    if not collections:
        raise typer.Exit(0)

    if collection is None:
        # Interactive selection
        if len(collections) == 1:
            collection = collections[0]
            ui.info(f"Auto-selected: {collection}")
        else:
            collection = ui.prompt_choice("Select collection", collections, collections[0])

    # =========================================================================
    # Validate collection exists
    # =========================================================================

    if collection not in collections:
        ui.error(f"Collection '{collection}' not found.")
        print()
        ui.info("Available collections:")
        for c in collections:
            ui.info(f"  • {c}")
        raise typer.Exit(1)

    # =========================================================================
    # Show collection details
    # =========================================================================

    if RICH:
        console.print(Panel.fit(f"[bold]{collection}[/bold]", title="Collection", border_style="green"))
    else:
        print(f"\nCollection: {collection}")
        print("=" * 40)

    _show_collection_details(vdb, collection, num_samples=samples)
    print()