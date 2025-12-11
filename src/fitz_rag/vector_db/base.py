from __future__ import annotations
from typing import Protocol, Iterable, List
from dataclasses import dataclass


@dataclass
class VectorRecord:
    id: str
    vector: List[float]
    metadata: dict


class VectorDBPlugin(Protocol):
    """
    Abstract interface for vector database backends.
    """

    def connect(self) -> None:
        """Initialize a connection if required."""
        ...

    def upsert(self, records: Iterable[VectorRecord]) -> None:
        """Insert or update vectors."""
        ...

    def query(self, vector: List[float], top_k: int) -> List[VectorRecord]:
        """Return nearest neighbors."""
        ...

    def delete(self, ids: Iterable[str]) -> None:
        """Delete vectors by ID."""
        ...
