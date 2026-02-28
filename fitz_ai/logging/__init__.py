# fitz_ai/logging/__init__.py
"""
Logging module for fitz-ai.

Provides structured logging with automatic context enrichment.
"""

from .logger import (
    StructuredLogger,
    clear_query_context,
    configure_logging,
    get_logger,
    set_query_context,
)

__all__ = [
    "get_logger",
    "configure_logging",
    "set_query_context",
    "clear_query_context",
    "StructuredLogger",
]