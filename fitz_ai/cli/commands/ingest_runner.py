# fitz_ai/cli/commands/ingest_runner.py
"""
Main ingestion orchestration for ingest command.

Contains the fitz_krag ingestion flow with progress tracking and summary display.
"""

from __future__ import annotations

from typing import Optional

import typer

from fitz_ai.cli.context import CLIContext
from fitz_ai.cli.ui import ui
from fitz_ai.logging.logger import get_logger
from fitz_ai.runtime import get_default_engine

from .ingest_direct import ingest_direct_text
from .ingest_engines import run_engine_specific_ingest
from .ingest_helpers import (
    is_direct_text,
)

logger = get_logger(__name__)


def command(
    source: Optional[str] = typer.Argument(
        None,
        help="Source path (file/directory) or direct text. Prompts if not provided.",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection name. Uses config default if not provided.",
    ),
    engine: Optional[str] = typer.Option(
        None,
        "--engine",
        "-e",
        help="Engine to use. Uses default from 'fitz engine' if not specified.",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Non-interactive mode, use defaults.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-ingest all files, ignoring state.",
    ),
    artifacts: Optional[str] = typer.Option(
        None,
        "--artifacts",
        "-a",
        help="Artifacts to generate: 'all', 'none', or comma-separated list (e.g. 'navigation_index,interface_catalog'). Interactive selection if not provided.",
    ),
) -> None:
    """
    Ingest documents into the vector database.

    Uses the default engine (set via 'fitz engine'). Override with --engine.
    Incremental by default - only processes new/changed files.
    Hierarchical summaries (L1/L2) are generated automatically.

    Examples:
        fitz ingest ./src            # Uses default engine
        fitz ingest ./src -a all     # All applicable artifacts
        fitz ingest ./src -a none    # No artifacts
        fitz ingest ./src -a navigation_index,interface_catalog  # Specific artifacts
        fitz ingest ./src -f         # Force re-ingest
        fitz ingest ./src -a all -y  # Non-interactive with all artifacts
        fitz ingest ./docs -e custom    # Use a custom engine
        fitz ingest "my boss likes red cars"  # Direct text ingestion
    """
    # =========================================================================
    # Direct Text Detection (before header to provide correct title)
    # =========================================================================

    if source is not None and is_direct_text(source):
        # Load context for collection
        ctx = CLIContext.load()

        # Get collection name
        if collection is None:
            collection = ctx.retrieval_collection

        # Ingest direct text and return
        ingest_direct_text(source, collection, ctx)
        return

    # =========================================================================
    # Header (for file/directory ingestion)
    # =========================================================================

    ui.header("Fitz Ingest", "Upload documents to vector database")

    # =========================================================================
    # Engine Selection (use default if not specified)
    # =========================================================================

    if engine is None:
        engine = get_default_engine()

    ui.info(f"Engine: {engine}")

    # Route to engine-specific ingest
    from fitz_ai.runtime import get_engine_registry

    registry = get_engine_registry()
    caps = registry.get_capabilities(engine)
    if caps.supports_persistent_ingest:
        run_engine_specific_ingest(source, collection, engine, non_interactive, force=force)
        return

    ui.error(f"Engine '{engine}' does not support persistent ingestion.")
    ui.info(f"Use 'fitz quickstart <folder> \"question\" --engine {engine}' instead.")
    raise typer.Exit(1)


