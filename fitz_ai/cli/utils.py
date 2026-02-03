# fitz_ai/cli/utils.py
"""
Shared CLI utilities.

Common functions used across multiple CLI commands.
Prefer using CLIContext directly for new code.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from fitz_ai.cli.context import CLIContext
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


def load_fitz_rag_config() -> Tuple[dict, Any]:
    """
    Load fitz_rag config.

    Returns:
        Tuple of (raw_config_dict, typed_config).
        Always succeeds - package defaults always exist.

    Note:
        Prefer using CLIContext.load() directly.
    """
    ctx = CLIContext.load()
    return ctx.raw_config, ctx.typed_config


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
    "load_fitz_rag_config",
    "get_collections",
    "get_vector_db_client",
]
