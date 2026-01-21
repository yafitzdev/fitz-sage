# fitz_ai/structured/derived.py
"""
Derived sentence storage for structured query results.

Stores LLM-generated sentences from SQL results with provenance
for semantic retrieval alongside original chunks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from fitz_ai.logging.logger import get_logger
from fitz_ai.structured.constants import (
    get_derived_collection,
)

logger = get_logger(__name__)


# Field names for derived records
FIELD_DERIVED = "__derived"
FIELD_SOURCE_TABLE = "__source_table"
FIELD_SOURCE_QUERY = "__source_query"
FIELD_TABLE_VERSION = "__table_version"
FIELD_GENERATED_AT = "__generated_at"
FIELD_CONTENT = "content"


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding generation."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        ...


@runtime_checkable
class VectorDBClient(Protocol):
    """Protocol for vector DB operations."""

    def upsert(self, collection_name: str, points: list[dict[str, Any]]) -> None:
        """Upsert points to collection."""
        ...

    def scroll(
        self,
        collection_name: str,
        limit: int,
        offset: int = 0,
        scroll_filter: dict[str, Any] | None = None,
        with_payload: bool = True,
    ) -> tuple[list[Any], int | None]:
        """Scroll through collection with optional filter."""
        ...

    def delete(
        self,
        collection_name: str,
        points_selector: dict[str, Any],
    ) -> int:
        """Delete points matching selector. Returns count deleted."""
        ...


@dataclass
class DerivedRecord:
    """A derived sentence with provenance."""

    id: str
    content: str
    source_table: str
    source_query: str
    table_version: str
    generated_at: datetime

    def to_payload(self) -> dict[str, Any]:
        """Convert to vector DB payload format."""
        return {
            FIELD_DERIVED: True,
            FIELD_SOURCE_TABLE: self.source_table,
            FIELD_SOURCE_QUERY: self.source_query,
            FIELD_TABLE_VERSION: self.table_version,
            FIELD_GENERATED_AT: self.generated_at.isoformat(),
            FIELD_CONTENT: self.content,
        }

    @classmethod
    def from_payload(cls, id: str, payload: dict[str, Any]) -> DerivedRecord:
        """Create from vector DB payload."""
        generated_str = payload.get(FIELD_GENERATED_AT, "")
        try:
            generated_at = datetime.fromisoformat(generated_str)
        except (ValueError, TypeError):
            generated_at = datetime.now(timezone.utc)

        return cls(
            id=id,
            content=payload.get(FIELD_CONTENT, ""),
            source_table=payload.get(FIELD_SOURCE_TABLE, ""),
            source_query=payload.get(FIELD_SOURCE_QUERY, ""),
            table_version=payload.get(FIELD_TABLE_VERSION, ""),
            generated_at=generated_at,
        )


def _generate_derived_id(sentence: str, table: str) -> str:
    """Generate deterministic ID for derived sentence."""
    content = f"{table}:{sentence}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class DerivedStore:
    """
    Stores derived sentences from structured query results.

    Sentences are embedded and stored for semantic retrieval
    alongside original document chunks.
    """

    def __init__(
        self,
        vector_db: VectorDBClient,
        embedding: EmbeddingClient,
        base_collection: str,
    ):
        """
        Initialize derived store.

        Args:
            vector_db: Vector DB client
            embedding: Embedding client
            base_collection: Base collection name
        """
        self._vector_db = vector_db
        self._embedding = embedding
        self._collection = get_derived_collection(base_collection)

    @property
    def collection_name(self) -> str:
        """Get the derived collection name."""
        return self._collection

    def ingest(
        self,
        sentence: str,
        source_table: str,
        source_query: str,
        table_version: str,
    ) -> DerivedRecord:
        """
        Ingest a derived sentence.

        Args:
            sentence: Natural language sentence from SQL result
            source_table: Table the data came from
            source_query: SQL query that generated the result
            table_version: Version hash of the source table

        Returns:
            The created DerivedRecord
        """
        record = DerivedRecord(
            id=_generate_derived_id(sentence, source_table),
            content=sentence,
            source_table=source_table,
            source_query=source_query,
            table_version=table_version,
            generated_at=datetime.now(timezone.utc),
        )

        # Generate embedding for sentence
        vectors = self._embedding.embed([sentence])
        vector = vectors[0] if vectors else []

        # Store in vector DB
        point = {
            "id": record.id,
            "vector": vector,
            "payload": record.to_payload(),
        }

        self._vector_db.upsert(self._collection, [point])

        logger.info(f"Ingested derived sentence from {source_table}: {sentence[:50]}...")

        return record

    def ingest_batch(
        self,
        sentences: list[str],
        source_table: str,
        source_queries: list[str],
        table_version: str,
    ) -> list[DerivedRecord]:
        """
        Ingest multiple derived sentences.

        Args:
            sentences: List of natural language sentences
            source_table: Table the data came from
            source_queries: List of SQL queries (parallel with sentences)
            table_version: Version hash of the source table

        Returns:
            List of created DerivedRecords
        """
        if not sentences:
            return []

        records = []
        for sentence, query in zip(sentences, source_queries):
            record = DerivedRecord(
                id=_generate_derived_id(sentence, source_table),
                content=sentence,
                source_table=source_table,
                source_query=query,
                table_version=table_version,
                generated_at=datetime.now(timezone.utc),
            )
            records.append(record)

        # Generate embeddings for all sentences
        vectors = self._embedding.embed([r.content for r in records])

        # Build points
        points = []
        for record, vector in zip(records, vectors):
            points.append(
                {
                    "id": record.id,
                    "vector": vector,
                    "payload": record.to_payload(),
                }
            )

        # Store in vector DB
        self._vector_db.upsert(self._collection, points)

        logger.info(f"Ingested {len(records)} derived sentences from {source_table}")

        return records

    def invalidate(self, table_name: str) -> int:
        """
        Invalidate all derived sentences for a table.

        Called when a table is updated/re-ingested to remove stale sentences.

        Args:
            table_name: Table whose derived sentences should be deleted

        Returns:
            Number of sentences deleted
        """
        # Build filter for table's derived sentences
        filter_condition = {
            "must": [
                {"key": FIELD_DERIVED, "match": {"value": True}},
                {"key": FIELD_SOURCE_TABLE, "match": {"value": table_name}},
            ]
        }

        # Scroll to find all matching IDs
        ids_to_delete = []
        offset = 0
        batch_size = 100

        while True:
            records, next_offset = self._vector_db.scroll(
                collection_name=self._collection,
                limit=batch_size,
                offset=offset,
                scroll_filter=filter_condition,
                with_payload=False,
            )

            if not records:
                break

            for record in records:
                record_id = getattr(record, "id", None) or record.get("id")
                if record_id:
                    ids_to_delete.append(record_id)

            if next_offset is None or len(records) < batch_size:
                break

            offset = next_offset

        if not ids_to_delete:
            logger.debug(f"No derived sentences to invalidate for {table_name}")
            return 0

        # Delete all found IDs
        try:
            deleted = self._vector_db.delete(
                collection_name=self._collection,
                points_selector={"points": ids_to_delete},
            )
            logger.info(f"Invalidated {deleted} derived sentences for {table_name}")
            return deleted
        except Exception as e:
            logger.error(f"Failed to invalidate derived sentences: {e}")
            return 0

    def invalidate_stale(self, table_name: str, current_version: str) -> int:
        """
        Invalidate derived sentences with outdated table versions.

        Args:
            table_name: Table to check
            current_version: Current version hash of the table

        Returns:
            Number of sentences deleted
        """
        # Build filter for table's derived sentences with wrong version
        filter_condition = {
            "must": [
                {"key": FIELD_DERIVED, "match": {"value": True}},
                {"key": FIELD_SOURCE_TABLE, "match": {"value": table_name}},
            ],
            "must_not": [
                {"key": FIELD_TABLE_VERSION, "match": {"value": current_version}},
            ],
        }

        # Scroll to find stale IDs
        ids_to_delete = []
        offset = 0
        batch_size = 100

        while True:
            records, next_offset = self._vector_db.scroll(
                collection_name=self._collection,
                limit=batch_size,
                offset=offset,
                scroll_filter=filter_condition,
                with_payload=False,
            )

            if not records:
                break

            for record in records:
                record_id = getattr(record, "id", None) or record.get("id")
                if record_id:
                    ids_to_delete.append(record_id)

            if next_offset is None or len(records) < batch_size:
                break

            offset = next_offset

        if not ids_to_delete:
            return 0

        try:
            deleted = self._vector_db.delete(
                collection_name=self._collection,
                points_selector={"points": ids_to_delete},
            )
            logger.info(f"Invalidated {deleted} stale derived sentences for {table_name}")
            return deleted
        except Exception as e:
            logger.error(f"Failed to invalidate stale sentences: {e}")
            return 0

    def get_by_table(self, table_name: str) -> list[DerivedRecord]:
        """
        Get all derived sentences for a table.

        Args:
            table_name: Table to get sentences for

        Returns:
            List of DerivedRecords
        """
        filter_condition = {
            "must": [
                {"key": FIELD_DERIVED, "match": {"value": True}},
                {"key": FIELD_SOURCE_TABLE, "match": {"value": table_name}},
            ]
        }

        records = []
        offset = 0
        batch_size = 100

        while True:
            batch, next_offset = self._vector_db.scroll(
                collection_name=self._collection,
                limit=batch_size,
                offset=offset,
                scroll_filter=filter_condition,
                with_payload=True,
            )

            if not batch:
                break

            for item in batch:
                payload = getattr(item, "payload", None) or item.get("payload", {})
                record_id = getattr(item, "id", None) or item.get("id", "")
                records.append(DerivedRecord.from_payload(record_id, payload))

            if next_offset is None or len(batch) < batch_size:
                break

            offset = next_offset

        return records


__all__ = [
    "DerivedStore",
    "DerivedRecord",
    "FIELD_DERIVED",
    "FIELD_SOURCE_TABLE",
    "FIELD_SOURCE_QUERY",
    "FIELD_TABLE_VERSION",
    "FIELD_GENERATED_AT",
    "FIELD_CONTENT",
]
