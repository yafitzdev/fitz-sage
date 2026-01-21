# fitz_ai/vector_db/types.py
"""
Typed models for vector database operations.

Provides strong typing for vector points, filter specifications,
and collection statistics used across vector DB implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

# =============================================================================
# Vector Point Types
# =============================================================================


class VectorPointDict(TypedDict, total=False):
    """
    Vector point as dictionary for upsert operations.

    Required fields:
        id: Unique identifier for the point
        vector: Embedding vector

    Optional fields:
        payload: Metadata attached to the point
    """

    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class VectorPoint:
    """Vector point with typed fields."""

    id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> VectorPointDict:
        """Convert to dictionary format for upsert."""
        return {"id": self.id, "vector": self.vector, "payload": self.payload}


# =============================================================================
# Filter Types (Qdrant-style)
# =============================================================================


class MatchFilter(TypedDict, total=False):
    """Match filter for exact value matching."""

    value: Any
    any: list[Any]  # Match any of these values


class RangeFilter(TypedDict, total=False):
    """Range filter for numeric comparisons."""

    gte: float | int  # Greater than or equal
    gt: float | int  # Greater than
    lte: float | int  # Less than or equal
    lt: float | int  # Less than


class FilterCondition(TypedDict, total=False):
    """
    Single filter condition (Qdrant-style).

    Examples:
        {"key": "category", "match": {"value": "tech"}}
        {"key": "price", "range": {"gte": 10, "lte": 100}}
    """

    key: str
    match: MatchFilter
    range: RangeFilter


class FilterSpec(TypedDict, total=False):
    """
    Complete filter specification with boolean combinators.

    Examples:
        {"must": [{"key": "category", "match": {"value": "tech"}}]}
        {"should": [cond1, cond2], "must_not": [cond3]}
    """

    must: list[FilterCondition | "FilterSpec"]
    should: list[FilterCondition | "FilterSpec"]
    must_not: list[FilterCondition | "FilterSpec"]


# =============================================================================
# Collection Types
# =============================================================================


@dataclass
class CollectionStats:
    """Statistics for a vector collection."""

    points_count: int
    vectors_count: int
    status: str = "green"
    vector_size: int | None = None
    indexed_vectors_count: int | None = None
    segments_count: int | None = None


@dataclass
class CollectionInfo:
    """Information about a vector collection."""

    name: str
    vector_size: int
    distance: str = "cosine"  # cosine, euclidean, dot
    points_count: int = 0


# =============================================================================
# Scroll Types
# =============================================================================


class ScrollResult(TypedDict):
    """Result from a scroll operation."""

    id: str
    payload: dict[str, Any]
    vector: list[float] | None


@dataclass
class ScrollResponse:
    """Response from a scroll operation."""

    points: list[ScrollResult]
    next_offset: str | None = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Point types
    "VectorPoint",
    "VectorPointDict",
    # Filter types
    "MatchFilter",
    "RangeFilter",
    "FilterCondition",
    "FilterSpec",
    # Collection types
    "CollectionStats",
    "CollectionInfo",
    # Scroll types
    "ScrollResult",
    "ScrollResponse",
]
