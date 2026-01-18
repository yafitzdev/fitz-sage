# fitz_ai/core/paths/ingestion.py
"""Ingestion state and table registry paths."""

from __future__ import annotations

from pathlib import Path

from .workspace import workspace


def ingest_state() -> Path:
    """
    Ingestion state file for incremental ingestion.

    Location: {workspace}/ingest.json

    This file tracks:
    - Which files have been ingested
    - Content hashes for change detection
    - Deletion tracking
    - Config snapshots for staleness detection
    """
    return workspace() / "ingest.json"


def table_registry(collection: str) -> Path:
    """
    Table chunk IDs registry for a collection.

    Location: {workspace}/tables/{collection}.json
    """
    return workspace() / "tables" / f"{collection}.json"


def ensure_table_registry_dir() -> Path:
    """Get tables directory and create it if it doesn't exist."""
    path = workspace() / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path
