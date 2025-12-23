# fitz_ai/vector_db/reader.py
"""
Vector DB reader interface for incremental ingestion.

Provides the read operations needed by the diff ingest system:
- Check if vectors exist for a content hash + config
- Mark vectors as deleted

This is separate from the writer to maintain clear responsibilities.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorDBClient(Protocol):
    """
    Protocol for vector database clients.

    Clients must implement scroll and update_payload for diff ingest support.
    """

    def scroll(
        self,
        collection: str,
        filter: Dict[str, Any],
        limit: int = 100,
        with_payload: bool = True,
    ) -> List[Any]:
        """Scroll through vectors matching a filter."""
        ...

    def update_payload(
        self,
        collection: str,
        ids: List[str],
        payload: Dict[str, Any],
    ) -> int:
        """Update payload for vectors by ID. Returns count updated."""
        ...


class VectorDBReader:
    """
    Reader for checking vector existence in incremental ingestion.

    This class provides the read operations required by the Differ:
    - has_content_hash: Check if vectors exist for a content hash + config
    - mark_deleted: Mark vectors as deleted (soft delete)

    Usage:
        reader = VectorDBReader(client)

        # Check if file already ingested
        if reader.has_content_hash(
            collection="docs",
            content_hash="sha256:abc...",
            parser_id="md.v1",
            chunker_id="tokens_800_120",
            embedding_id="openai:text-embedding-3-small",
        ):
            print("Already ingested, skip")

        # Mark deleted vectors
        count = reader.mark_deleted("docs", "/path/to/deleted.md")
        print(f"Marked {count} vectors as deleted")
    """

    def __init__(self, client: VectorDBClient) -> None:
        """
        Initialize the reader.

        Args:
            client: Vector database client with scroll/update support
        """
        self._client = client

    def has_content_hash(
        self,
        collection: str,
        content_hash: str,
        parser_id: str,
        chunker_id: str,
        embedding_id: str,
    ) -> bool:
        """
        Check if vectors exist for a given content hash + config.

        This is the authoritative check for incremental ingestion per spec §5:
        "Vector DB is authoritative - even if state is deleted, this check
        must correctly skip unchanged content."

        Args:
            collection: Vector DB collection name
            content_hash: SHA-256 hash of file content
            parser_id: Parser identifier (e.g., "md.v1")
            chunker_id: Chunker identifier (e.g., "tokens_800_120")
            embedding_id: Embedding identifier (e.g., "openai:text-embedding-3-small")

        Returns:
            True if vectors exist with is_deleted=false, False otherwise
        """
        try:
            # Build filter for exact match on all config fields
            filter_conditions = {
                "must": [
                    {"key": "content_hash", "match": {"value": content_hash}},
                    {"key": "parser_id", "match": {"value": parser_id}},
                    {"key": "chunker_id", "match": {"value": chunker_id}},
                    {"key": "embedding_id", "match": {"value": embedding_id}},
                    {"key": "is_deleted", "match": {"value": False}},
                ]
            }

            results = self._client.scroll(
                collection=collection,
                filter=filter_conditions,
                limit=1,  # We only need to know if any exist
                with_payload=False,
            )

            return len(results) > 0

        except Exception as e:
            logger.warning(f"Error checking content hash existence: {e}")
            # On error, return False to trigger re-ingestion
            # This is safe because upserts are idempotent
            return False

    def mark_deleted(
        self,
        collection: str,
        source_path: str,
    ) -> int:
        """
        Mark all vectors for a source path as deleted.

        Per spec §6.1:
        - Find all vectors where source_path == path AND is_deleted == false
        - Update metadata: is_deleted=true, deleted_at=timestamp
        - No hard delete

        Args:
            collection: Vector DB collection name
            source_path: File path to mark deleted

        Returns:
            Number of vectors marked as deleted
        """
        try:
            # Find vectors for this source path that aren't already deleted
            filter_conditions = {
                "must": [
                    {"key": "source_path", "match": {"value": source_path}},
                    {"key": "is_deleted", "match": {"value": False}},
                ]
            }

            results = self._client.scroll(
                collection=collection,
                filter=filter_conditions,
                limit=10000,  # Reasonable max for a single file
                with_payload=True,
            )

            if not results:
                return 0

            # Extract IDs
            ids = [r.id if hasattr(r, 'id') else r['id'] for r in results]

            # Update payload to mark as deleted
            deleted_payload = {
                "is_deleted": True,
                "deleted_at": datetime.utcnow().isoformat(),
            }

            count = self._client.update_payload(
                collection=collection,
                ids=ids,
                payload=deleted_payload,
            )

            logger.debug(f"Marked {count} vectors as deleted for {source_path}")
            return count

        except Exception as e:
            logger.error(f"Error marking vectors as deleted for {source_path}: {e}")
            raise

    def get_vector_count(
        self,
        collection: str,
        content_hash: Optional[str] = None,
        source_path: Optional[str] = None,
        include_deleted: bool = False,
    ) -> int:
        """
        Count vectors matching criteria.

        Useful for debugging and verification.

        Args:
            collection: Vector DB collection name
            content_hash: Optional filter by content hash
            source_path: Optional filter by source path
            include_deleted: If True, include deleted vectors in count

        Returns:
            Number of matching vectors
        """
        try:
            must_conditions = []

            if content_hash is not None:
                must_conditions.append(
                    {"key": "content_hash", "match": {"value": content_hash}}
                )

            if source_path is not None:
                must_conditions.append(
                    {"key": "source_path", "match": {"value": source_path}}
                )

            if not include_deleted:
                must_conditions.append(
                    {"key": "is_deleted", "match": {"value": False}}
                )

            filter_conditions = {"must": must_conditions} if must_conditions else {}

            results = self._client.scroll(
                collection=collection,
                filter=filter_conditions,
                limit=100000,
                with_payload=False,
            )

            return len(results)

        except Exception as e:
            logger.warning(f"Error counting vectors: {e}")
            return 0


__all__ = ["VectorDBClient", "VectorDBReader"]