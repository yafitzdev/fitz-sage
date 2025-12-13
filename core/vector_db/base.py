# core/vector_db/base.py
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorDBPlugin(Protocol):
    plugin_name: str
    plugin_type: str  # must be "vector_db"

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool = True,
    ) -> list[Any]:
        ...
