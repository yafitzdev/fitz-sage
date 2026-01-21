# fitz_ai/cli/commands/ingest_engines.py
"""
Engine-specific ingestion for ingest command.

Handles ingestion for non-fitz_rag engines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from fitz_ai.cli.ui import RICH, console, ui
from fitz_ai.logging.logger import get_logger

from .ingest_helpers import suggest_collection_name

logger = get_logger(__name__)


def run_engine_specific_ingest(
    source: Optional[str],
    collection: Optional[str],
    engine_name: str,
    non_interactive: bool,
) -> None:
    """Run ingest for engines with supports_persistent_ingest capability."""
    from fitz_ai.runtime import create_engine, get_engine_registry

    # Validate engine
    registry = get_engine_registry()
    available = registry.list()
    if engine_name not in available:
        ui.error(f"Unknown engine: '{engine_name}'. Available: {', '.join(available)}")
        raise typer.Exit(1)

    # Check if engine supports persistent ingest
    caps = registry.get_capabilities(engine_name)
    if not caps.supports_persistent_ingest:
        ui.error(f"Engine '{engine_name}' does not support persistent ingestion.")
        ui.info("Use 'fitz ingest' without --engine for fitz_rag ingestion.")
        raise typer.Exit(1)

    # Get source path
    if source is None:
        if non_interactive:
            ui.error("Source path required in non-interactive mode.")
            raise typer.Exit(1)
        source = ui.prompt_path("Source path", ".")

    source_path = Path(source).resolve()
    if not source_path.exists():
        ui.error(f"Source path not found: {source_path}")
        raise typer.Exit(1)

    # Get collection name
    if collection is None:
        collection = suggest_collection_name(source)
        if not non_interactive:
            collection = ui.prompt_text("Collection name", collection)

    ui.info(f"Source: {source_path}")
    ui.info(f"Collection: {collection}")
    ui.info(f"Engine: {engine_name}")
    print()

    # Create engine and run ingest
    ui.step(1, 2, f"Initializing {engine_name} engine...")

    try:
        engine = create_engine(engine_name)
        ui.success("Engine initialized")
    except Exception as e:
        ui.error(f"Failed to initialize engine: {e}")
        logger.debug("Engine init error", exc_info=True)
        raise typer.Exit(1)

    ui.step(2, 2, "Ingesting documents...")

    try:
        result = engine.ingest(source_path, collection)
        ui.success("Ingestion complete")
        print()

        # Display results
        if RICH:
            from rich.table import Table

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Metric", style="bold")
            table.add_column("Value", style="cyan")

            for key, value in result.items():
                if key != "storage_path":
                    table.add_row(key.replace("_", " ").title(), str(value))

            console.print(table)
        else:
            for key, value in result.items():
                if key != "storage_path":
                    print(f"  {key.replace('_', ' ').title()}: {value}")

        print()
        ui.success(
            f"Collection '{collection}' saved to {result.get('storage_path', 'persistent storage')}"
        )

    except Exception as e:
        ui.error(f"Ingestion failed: {e}")
        logger.debug("Ingestion error", exc_info=True)
        raise typer.Exit(1)
