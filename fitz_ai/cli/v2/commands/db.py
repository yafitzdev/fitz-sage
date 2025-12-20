# fitz_ai/cli/v2/commands/db.py
"""
Database inspection command.

Usage:
    fitz db                    # Interactive - list and select collection
    fitz db my_collection      # Inspect specific collection
    fitz db -d my_collection   # Delete a collection
"""

from __future__ import annotations

from typing import Optional

import typer

from fitz_ai.core.config import load_config_dict, ConfigNotFoundError
from fitz_ai.core.paths import FitzPaths
from fitz_ai.vector_db.registry import get_vector_db_plugin
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

# Rich for UI (optional)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.table import Table

    console = Console()
    RICH = True
except ImportError:
    console = None
    RICH = False


# =============================================================================
# UI Helpers
# =============================================================================


def _print(msg: str, style: str = "") -> None:
    if RICH and style:
        console.print(f"[{style}]{msg}[/{style}]")
    else:
        print(msg)


def _header(title: str) -> None:
    if RICH:
        console.print(Panel.fit(f"[bold]{title}[/bold]", border_style="blue"))
    else:
        print(f"\n{'=' * 50}")
        print(title)
        print('=' * 50)


def _error(msg: str) -> None:
    if RICH:
        console.print(f"[red]✗[/red] {msg}")
    else:
        print(f"✗ {msg}")


def _success(msg: str) -> None:
    if RICH:
        console.print(f"[green]✓[/green] {msg}")
    else:
        print(f"✓ {msg}")


def _prompt_choice(prompt: str, choices: list[str], default: str = None) -> str:
    if RICH:
        return Prompt.ask(prompt, choices=choices, default=default)
    else:
        choices_str = "/".join(choices)
        while True:
            if default:
                response = input(f"{prompt} [{choices_str}] ({default}): ").strip()
                if not response:
                    return default
            else:
                response = input(f"{prompt} [{choices_str}]: ").strip()
            if response in choices:
                return response
            print(f"Choose from: {', '.join(choices)}")


def _prompt_confirm(prompt: str, default: bool = False) -> bool:
    if RICH:
        return Confirm.ask(prompt, default=default)
    else:
        yn = "Y/n" if default else "y/N"
        response = input(f"{prompt} [{yn}]: ").strip().lower()
        if not response:
            return default
        return response in ("y", "yes")


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
        _print("No collections found.", "dim")
        _print("Run 'fitz ingest' to create one.", "dim")
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
        _error(f"Failed to get stats: {e}")
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

        # Show dimension if available
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
            _print("Cannot fetch samples (no direct client access)", "dim")
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
            _print("Cannot fetch samples (scroll not supported)", "dim")
            return

        if not records:
            _print("No chunks found in collection.", "dim")
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
        _print(f"Could not fetch samples: {e}", "dim")


def _delete_collection(vdb, collection: str) -> bool:
    """Delete a collection after confirmation."""
    try:
        stats = vdb.get_collection_stats(collection)
        points = stats.get("points_count", 0)
    except Exception:
        points = "?"

    _print(f"Collection '{collection}' has {points} chunks.", "yellow" if RICH else "")

    if not _prompt_confirm(f"Delete collection '{collection}'? This cannot be undone", default=False):
        _print("Cancelled.", "dim")
        return False

    try:
        vdb.delete_collection(collection)
        _success(f"Deleted collection '{collection}'")
        return True
    except Exception as e:
        _error(f"Failed to delete: {e}")
        return False


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

    _header("Fitz Database")
    _print(f"Vector DB: {vector_db_plugin}", "dim")

    # =========================================================================
    # Get vector DB client
    # =========================================================================

    try:
        vdb = get_vector_db_plugin(vector_db_plugin)
    except Exception as e:
        _error(f"Failed to connect to vector DB: {e}")
        raise typer.Exit(1)

    # =========================================================================
    # List only mode
    # =========================================================================

    if list_only:
        _list_collections(vdb, vector_db_plugin)
        return

    # =========================================================================
    # Delete mode
    # =========================================================================

    if delete:
        if not collection:
            _error("Specify collection to delete: fitz db <collection> --delete")
            raise typer.Exit(1)

        collections = vdb.list_collections()
        if collection not in collections:
            _error(f"Collection '{collection}' not found.")
            raise typer.Exit(1)

        _delete_collection(vdb, collection)
        return

    # =========================================================================
    # Interactive mode - select collection if not provided
    # =========================================================================

    collections = _list_collections(vdb, vector_db_plugin)

    if not collections:
        raise typer.Exit(0)

    if collection is None:
        # Interactive selection
        if len(collections) == 1:
            collection = collections[0]
            _print(f"Auto-selected: {collection}", "dim")
        else:
            if RICH:
                collection = Prompt.ask(
                    "Select collection",
                    choices=collections,
                    default=collections[0],
                )
            else:
                print(f"Available: {', '.join(collections)}")
                collection = input(f"Select collection [{collections[0]}]: ").strip()
                if not collection:
                    collection = collections[0]
                if collection not in collections:
                    _error(f"Unknown collection: {collection}")
                    raise typer.Exit(1)

    # =========================================================================
    # Validate collection exists
    # =========================================================================

    if collection not in collections:
        _error(f"Collection '{collection}' not found.")
        print()
        _print("Available collections:", "dim")
        for c in collections:
            _print(f"  • {c}", "dim")
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