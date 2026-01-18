# fitz_ai/cli/commands/ingest.py
"""
Document ingestion with incremental (diff) support.

Usage:
    fitz ingest              # Interactive mode
    fitz ingest ./src        # Ingest specific directory
    fitz ingest ./src -y     # Non-interactive with defaults
    fitz ingest "my text"    # Direct text ingestion

This module is the entry point for the ingest command.
Implementation is split across:
    - ingest_helpers.py: Content detection and utility functions
    - ingest_direct.py: Direct text ingestion
    - ingest_config.py: Config building helpers
    - ingest_adapters.py: Protocol adapters
    - ingest_engines.py: Engine-specific ingestion
    - ingest_runner.py: Main ingestion orchestration
"""

from __future__ import annotations

from .ingest_runner import command

__all__ = ["command"]
