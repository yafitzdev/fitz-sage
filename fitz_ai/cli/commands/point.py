# fitz_ai/cli/commands/point.py
"""
CLI command: fitz point — point at a folder to start querying immediately.

Builds a manifest, starts background indexing, and returns instantly.
Queries work immediately via agentic LLM-driven search; they get
progressively faster as the background worker indexes files.
"""

from __future__ import annotations

from pathlib import Path

from fitz_ai.cli.ui import ui
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


def command(source: Path, collection: str) -> None:
    """Point at a folder for immediate querying with background indexing."""
    source = Path(source).resolve()
    if not source.exists():
        ui.error(f"Source path does not exist: {source}")
        raise SystemExit(1)

    from fitz_ai.services import FitzService

    service = FitzService()
    manifest = service.point(source=source, collection=collection)
    file_count = len(manifest.entries())

    ui.success(f"Ready! {file_count} files registered.")
    ui.info("Ask questions now — queries get faster over time.")
