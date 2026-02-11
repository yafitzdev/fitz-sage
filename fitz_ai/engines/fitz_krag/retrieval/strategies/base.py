# fitz_ai/engines/fitz_krag/retrieval/strategies/base.py
"""Base protocol for retrieval strategies."""

from __future__ import annotations

from typing import Protocol

from fitz_ai.engines.fitz_krag.types import Address


class RetrievalStrategy(Protocol):
    """Protocol for KRAG retrieval strategies."""

    def retrieve(self, query: str, limit: int) -> list[Address]:
        """Retrieve addresses matching the query."""
        ...
