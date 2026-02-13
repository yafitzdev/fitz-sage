# fitz_ai/cli/__init__.py
"""
Fitz CLI - Clean, minimal CLI.

Usage:
    fitz init                                        # Setup wizard
    fitz query "question" --source ./docs            # Register + query
    fitz query "question"                            # Query existing collection
    fitz collections                                 # List/manage collections
    fitz config                                      # Show configuration
"""

from fitz_ai.cli.cli import app  # noqa: E402

__all__ = ["app"]
