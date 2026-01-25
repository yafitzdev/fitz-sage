# fitz_ai/ingestion/state/manager.py
"""
State manager for incremental ingestion using PostgreSQL.

Handles loading, saving, and updating ingestion state in PostgreSQL.

Key responsibilities:
- Track which files have been ingested
- Mark files as active (after successful ingestion)
- Mark files as deleted (when no longer on disk)
- Track config IDs (chunker_id, parser_id, embedding_id) per file
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fitz_ai.ingestion.state.schema import (
    EmbeddingConfig,
    FileEntry,
    FileStatus,
)
from fitz_ai.storage import get_connection_manager

logger = logging.getLogger(__name__)

# Use a dedicated database for global ingest state (not per-collection)
INGEST_STATE_COLLECTION = "_ingest_state"


class IngestStateManager:
    """
    Manages the ingestion state in PostgreSQL.

    The state tracks:
    - Which files have been ingested
    - Content hashes for change detection
    - Config IDs for re-chunking detection
    - Deletion tracking

    Usage:
        manager = IngestStateManager()
        manager.load()

        # After successful ingestion
        manager.mark_active(
            file_path="/path/to/file.md",
            root="/path/to",
            content_hash="sha256:abc123",
            ext=".md",
            size_bytes=1234,
            mtime_epoch=1234567890.0,
            chunker_id="simple:1000:0",
            parser_id="md.v1",
            embedding_id="cohere:embed-english-v3.0",
        )

        manager.save()  # Auto-commits, but kept for API compatibility
    """

    SCHEMA_SQL = """
        -- Project metadata
        CREATE TABLE IF NOT EXISTS ingest_project (
            id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            project_id TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            embedding_provider TEXT,
            embedding_model TEXT,
            embedding_dimension INTEGER,
            embedding_id TEXT
        );

        -- Root directories
        CREATE TABLE IF NOT EXISTS ingest_roots (
            root_path TEXT PRIMARY KEY,
            last_scan_at TIMESTAMPTZ NOT NULL
        );

        -- File entries
        CREATE TABLE IF NOT EXISTS ingest_files (
            file_path TEXT PRIMARY KEY,
            root_path TEXT NOT NULL REFERENCES ingest_roots(root_path) ON DELETE CASCADE,
            content_hash TEXT NOT NULL,
            ext TEXT NOT NULL,
            size_bytes BIGINT NOT NULL,
            mtime_epoch DOUBLE PRECISION NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            ingested_at TIMESTAMPTZ NOT NULL,
            chunker_id TEXT NOT NULL,
            parser_id TEXT NOT NULL,
            embedding_id TEXT NOT NULL,
            vector_db_id TEXT,
            enricher_id TEXT,
            collection TEXT NOT NULL DEFAULT 'default'
        );

        -- Index for root lookups
        CREATE INDEX IF NOT EXISTS idx_ingest_files_root
        ON ingest_files(root_path);

        -- Index for status queries
        CREATE INDEX IF NOT EXISTS idx_ingest_files_status
        ON ingest_files(status);
    """

    def __init__(self, path=None) -> None:
        """
        Initialize the state manager.

        Args:
            path: Ignored (kept for backwards compatibility). State is stored in PostgreSQL.
        """
        self._manager = get_connection_manager()
        self._manager.start()
        self._schema_initialized = False
        self._project_id: Optional[str] = None
        self._embedding: Optional[EmbeddingConfig] = None
        self._dirty: bool = False

    def _ensure_schema(self) -> None:
        """Create tables schema if not exists."""
        if self._schema_initialized:
            return

        with self._manager.connection(INGEST_STATE_COLLECTION) as conn:
            conn.execute(self.SCHEMA_SQL)
            conn.commit()

        self._schema_initialized = True
        logger.debug("Ingest state schema initialized in PostgreSQL")

    def load(self) -> "IngestStateManager":
        """
        Load state from PostgreSQL, or create new if not exists.

        Returns:
            Self for chaining.
        """
        self._ensure_schema()

        with self._manager.connection(INGEST_STATE_COLLECTION) as conn:
            # Load project metadata
            result = conn.execute(
                """
                SELECT project_id, embedding_provider, embedding_model,
                       embedding_dimension, embedding_id
                FROM ingest_project
                WHERE id = 1
                """
            ).fetchone()

            if result:
                self._project_id = result[0]
                if result[1] and result[2]:
                    self._embedding = EmbeddingConfig(
                        provider=result[1],
                        model=result[2],
                        dimension=result[3],
                        id=result[4] or f"{result[1]}:{result[2]}",
                    )
                logger.debug(f"Loaded ingest state for project {self._project_id}")
            else:
                # Create new project
                self._project_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO ingest_project (id, project_id, updated_at)
                    VALUES (1, %s, %s)
                    """,
                    (self._project_id, datetime.now(timezone.utc)),
                )
                conn.commit()
                logger.debug(f"Created new ingest state: {self._project_id}")

        return self

    def save(self) -> None:
        """
        Save state to PostgreSQL.

        In PostgreSQL mode, changes are committed immediately,
        but this method updates the project timestamp.
        """
        if not self._dirty:
            return

        self._ensure_schema()

        with self._manager.connection(INGEST_STATE_COLLECTION) as conn:
            conn.execute(
                """
                UPDATE ingest_project SET updated_at = %s WHERE id = 1
                """,
                (datetime.now(timezone.utc),),
            )
            conn.commit()

        self._dirty = False
        logger.debug("Saved ingest state to PostgreSQL")

    def _ensure_root(self, root: str) -> None:
        """Ensure root entry exists, creating if necessary."""
        self._ensure_schema()

        with self._manager.connection(INGEST_STATE_COLLECTION) as conn:
            conn.execute(
                """
                INSERT INTO ingest_roots (root_path, last_scan_at)
                VALUES (%s, %s)
                ON CONFLICT (root_path) DO NOTHING
                """,
                (root, datetime.now(timezone.utc)),
            )
            conn.commit()

    def mark_active(
        self,
        file_path: str,
        root: str,
        content_hash: str,
        ext: str,
        size_bytes: int,
        mtime_epoch: float,
        chunker_id: str,
        parser_id: str,
        embedding_id: str,
        enricher_id: Optional[str] = None,
        vector_db_id: Optional[str] = None,
        collection: str = "default",
    ) -> None:
        """
        Mark a file as active in state.

        Called after successful ingestion or when file is unchanged.

        Args:
            file_path: Absolute path to file.
            root: Absolute path to root directory.
            content_hash: SHA-256 content hash.
            ext: File extension (e.g., ".md").
            size_bytes: File size in bytes.
            mtime_epoch: File modification time as Unix epoch.
            chunker_id: Chunker ID used (e.g., "simple:1000:0").
            parser_id: Parser ID used (e.g., "md.v1").
            embedding_id: Embedding ID used (e.g., "cohere:embed-english-v3.0").
            enricher_id: Enricher ID used (e.g., "llm:gpt-4o-mini:v1"), None if not enriched.
            vector_db_id: Vector DB plugin used (e.g., "qdrant", "pgvector").
            collection: Collection the file was ingested into.
        """
        self._ensure_root(root)

        with self._manager.connection(INGEST_STATE_COLLECTION) as conn:
            now = datetime.now(timezone.utc)

            # Upsert file entry
            conn.execute(
                """
                INSERT INTO ingest_files (
                    file_path, root_path, content_hash, ext, size_bytes, mtime_epoch,
                    status, ingested_at, chunker_id, parser_id, embedding_id,
                    vector_db_id, enricher_id, collection
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (file_path) DO UPDATE SET
                    root_path = EXCLUDED.root_path,
                    content_hash = EXCLUDED.content_hash,
                    ext = EXCLUDED.ext,
                    size_bytes = EXCLUDED.size_bytes,
                    mtime_epoch = EXCLUDED.mtime_epoch,
                    status = EXCLUDED.status,
                    ingested_at = EXCLUDED.ingested_at,
                    chunker_id = EXCLUDED.chunker_id,
                    parser_id = EXCLUDED.parser_id,
                    embedding_id = EXCLUDED.embedding_id,
                    vector_db_id = EXCLUDED.vector_db_id,
                    enricher_id = EXCLUDED.enricher_id,
                    collection = EXCLUDED.collection
                """,
                (
                    file_path,
                    root,
                    content_hash,
                    ext,
                    size_bytes,
                    mtime_epoch,
                    FileStatus.ACTIVE.value,
                    now,
                    chunker_id,
                    parser_id,
                    embedding_id,
                    vector_db_id,
                    enricher_id,
                    collection,
                ),
            )

            # Update root scan time
            conn.execute(
                """
                UPDATE ingest_roots SET last_scan_at = %s WHERE root_path = %s
                """,
                (now, root),
            )

            conn.commit()

        self._dirty = True

    def mark_deleted(self, root: str, file_path: str) -> None:
        """
        Mark a file as deleted in state.

        Called when a file is no longer present on disk.

        Args:
            root: Absolute path to root directory.
            file_path: Absolute path to file.
        """
        self._ensure_schema()

        with self._manager.connection(INGEST_STATE_COLLECTION) as conn:
            result = conn.execute(
                """
                UPDATE ingest_files
                SET status = %s
                WHERE file_path = %s AND root_path = %s AND status != %s
                RETURNING file_path
                """,
                (FileStatus.DELETED.value, file_path, root, FileStatus.DELETED.value),
            ).fetchone()

            if result:
                conn.commit()
                self._dirty = True

    def get_active_paths(self, root: str) -> set[str]:
        """Get all active (non-deleted) file paths for a root."""
        self._ensure_schema()

        with self._manager.connection(INGEST_STATE_COLLECTION) as conn:
            cursor = conn.execute(
                """
                SELECT file_path FROM ingest_files
                WHERE root_path = %s AND status = %s
                """,
                (root, FileStatus.ACTIVE.value),
            )
            return {row[0] for row in cursor.fetchall()}

    def get_file_entry(self, root: str, file_path: str) -> Optional[FileEntry]:
        """Get file entry if it exists."""
        self._ensure_schema()

        with self._manager.connection(INGEST_STATE_COLLECTION) as conn:
            result = conn.execute(
                """
                SELECT content_hash, ext, size_bytes, mtime_epoch, status,
                       ingested_at, chunker_id, parser_id, embedding_id,
                       vector_db_id, enricher_id, collection
                FROM ingest_files
                WHERE file_path = %s AND root_path = %s
                """,
                (file_path, root),
            ).fetchone()

            if not result:
                return None

            return FileEntry(
                content_hash=result[0],
                ext=result[1],
                size_bytes=result[2],
                mtime_epoch=result[3],
                status=FileStatus(result[4]),
                ingested_at=result[5],
                chunker_id=result[6],
                parser_id=result[7],
                embedding_id=result[8],
                vector_db_id=result[9],
                enricher_id=result[10],
                collection=result[11],
            )

    def set_embedding_config(
        self,
        provider: str,
        model: str,
        **kwargs,
    ) -> None:
        """
        Set the current embedding configuration.

        Args:
            provider: Embedding provider name.
            model: Model name.
            **kwargs: Additional config (dimension, etc.).
        """
        self._ensure_schema()
        self._embedding = EmbeddingConfig.create(provider, model, **kwargs)

        with self._manager.connection(INGEST_STATE_COLLECTION) as conn:
            conn.execute(
                """
                UPDATE ingest_project SET
                    embedding_provider = %s,
                    embedding_model = %s,
                    embedding_dimension = %s,
                    embedding_id = %s,
                    updated_at = %s
                WHERE id = 1
                """,
                (
                    provider,
                    model,
                    kwargs.get("dimension"),
                    self._embedding.id,
                    datetime.now(timezone.utc),
                ),
            )
            conn.commit()

        self._dirty = True

    # Properties for backwards compatibility
    @property
    def state(self):
        """Get state-like object for backwards compatibility."""
        return self

    @property
    def embedding(self) -> Optional[EmbeddingConfig]:
        """Get current embedding config (backwards compatibility)."""
        return self._embedding

    @property
    def schema_version(self) -> int:
        """Schema version (backwards compatibility)."""
        return 1

    @property
    def project_id(self) -> str:
        """Project ID (backwards compatibility)."""
        if self._project_id is None:
            self._ensure_schema()
            self.load()
        return self._project_id or ""


__all__ = ["IngestStateManager"]
