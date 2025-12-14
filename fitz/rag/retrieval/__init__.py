# rag/retrieval/__init__.py
"""
Retrieval subsystem public API.

Keep this module import-light to avoid circular imports.
We provide lazy re-exports for convenience.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

__all__ = ["RetrievalPlugin", "RetrieverEngine"]


def __getattr__(name: str) -> Any:
    if name == "RetrievalPlugin":
        from rag.retrieval import RetrievalPlugin as _RetrievalPlugin

        return _RetrievalPlugin

    if name == "RetrieverEngine":
        from rag.retrieval import RetrieverEngine as _RetrieverEngine

        return _RetrieverEngine

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
