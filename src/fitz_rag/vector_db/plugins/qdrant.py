from __future__ import annotations

from typing import Iterable, List
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from fitz_rag.vector_db.base import VectorDBPlugin, VectorRecord
from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import VECTOR_DB

logger = get_logger(__name__)


@dataclass
class QdrantVectorDB(VectorDBPlugin):
    host: str = "localhost"
    port: int = 6333
    collection: str = "default"
    client: QdrantClient | None = None

    def connect(self) -> None:
        if self.client is None:
            logger.info(f"{VECTOR_DB} Connecting to Qdrant @ {self.host}:{self.port}")
            self.client = QdrantClient(host=self.host, port=self.port)

    def upsert(self, records: Iterable[VectorRecord]) -> None:
        self.connect()
        points = [
            PointStruct(
                id=rec.id,
                vector=rec.vector,
                payload=rec.metadata,
            )
            for rec in records
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def query(self, vector: List[float], top_k: int) -> List[VectorRecord]:
        self.connect()
        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
        )
        return [
            VectorRecord(
                id=str(hit.id),
                vector=hit.vector or [],
                metadata=hit.payload or {},
            )
            for hit in results
        ]

    def delete(self, ids: Iterable[str]) -> None:
        self.connect()
        self.client.delete(collection_name=self.collection, points=list(ids))
