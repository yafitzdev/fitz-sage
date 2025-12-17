# fitz/vector_db/plugins/qdrant.py
"""
Qdrant vector database plugin.

Supports qdrant-client >= 1.7 API.
Uses simple (unnamed) vectors for compatibility.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from fitz.vector_db.base import VectorDBPlugin, SearchResult


def _string_to_uuid(s: str) -> str:
    """Convert any string to a valid UUID (deterministic)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


@dataclass
class QdrantVectorDB(VectorDBPlugin):
    """
    Qdrant vector database plugin.

    Uses simple (unnamed) vectors. Create your collection like:
        curl -X PUT "http://host:6333/collections/myname" \\
             -H "Content-Type: application/json" \\
             -d '{"vectors": {"size": 1024, "distance": "Cosine"}}'
    """

    plugin_name: str = "qdrant"
    plugin_type: str = "vector_db"

    host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "192.168.178.2"))
    port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))

    _client: QdrantClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = QdrantClient(host=self.host, port=self.port)

    def search(
            self,
            collection_name: str,
            query_vector: list[float],
            limit: int,
            with_payload: bool = True,
    ) -> list[SearchResult]:
        """Search for similar vectors in a collection."""
        # Simple query - no named vectors
        result = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=with_payload,
        )

        return [
            SearchResult(
                id=str(point.id),
                score=point.score,
                payload=point.payload or {},
            )
            for point in result.points
        ]

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """Upsert points into a collection."""
        q_points = []
        for p in points:
            original_id = p["id"]

            # Convert string ID to UUID for Qdrant
            if isinstance(original_id, str):
                qdrant_id = _string_to_uuid(original_id)
            else:
                qdrant_id = original_id

            # Store original ID in payload
            payload = p.get("payload", {}) or {}
            payload["original_id"] = original_id

            # Simple vector (not named)
            q_points.append(
                PointStruct(
                    id=qdrant_id,
                    vector=p["vector"],
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=collection, points=q_points)