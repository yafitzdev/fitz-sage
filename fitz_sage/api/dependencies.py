# fitz_sage/api/dependencies.py
"""Shared dependencies for API routes."""

from __future__ import annotations

from functools import lru_cache

from fitz_sage.services import FitzService


@lru_cache(maxsize=1)
def get_service() -> FitzService:
    """Get the singleton FitzService instance."""
    return FitzService()


def get_fitz_version() -> str:
    """Get the current fitz version."""
    try:
        from fitz_sage import __version__

        return __version__
    except ImportError:
        return "unknown"
