# fitz_ai/cli/__init__.py
"""
Fitz CLI - Clean, minimal CLI.

Usage:
    fitz query "question" --source ./docs            # Register + query
    fitz query "question"                            # Query existing collection
    fitz collections                                 # List/manage collections
    fitz serve                                       # Start REST API

Config: .fitz/config.yaml (auto-created on first run)
"""

from fitz_ai.cli.cli import app  # noqa: E402

__all__ = ["app"]
