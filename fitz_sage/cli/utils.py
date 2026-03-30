# fitz_sage/cli/utils.py
"""
Shared CLI utilities.

Common functions used across multiple CLI commands.
Prefer using CLIContext directly for new code.
"""

from __future__ import annotations

from typing import List

from fitz_sage.cli.context import CLIContext
from fitz_sage.logging.logger import get_logger

logger = get_logger(__name__)


def get_collections(ctx: CLIContext = None) -> List[str]:
    """
    Get list of collections from vector DB.

    Args:
        ctx: CLIContext (optional, will load if not provided)

    Returns:
        Sorted list of collection names, or empty list on error.

    Note:
        Prefer using ctx.get_collections() from CLIContext directly.
    """
    if ctx is None:
        ctx = CLIContext.load()
    return ctx.get_collections()


def get_vector_db_client(ctx: CLIContext = None):
    """
    Get vector DB client from ctx.

    Args:
        ctx: CLIContext (optional, will load if not provided)

    Returns:
        Vector DB client instance.

    Note:
        Prefer using ctx.get_vector_db_client() from CLIContext directly.
    """
    if ctx is None:
        ctx = CLIContext.load()
    return ctx.get_vector_db_client()


__all__ = [
    "CLIContext",
    "get_collections",
    "get_vector_db_client",
]
