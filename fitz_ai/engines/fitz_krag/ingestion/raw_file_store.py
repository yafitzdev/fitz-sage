# fitz_ai/engines/fitz_krag/ingestion/raw_file_store.py
"""CRUD operations for krag_raw_files table."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fitz_ai.engines.fitz_krag.ingestion.schema import TABLE_PREFIX

if TYPE_CHECKING:
    from fitz_ai.storage.postgres import PostgresConnectionManager

logger = logging.getLogger(__name__)

TABLE = f"{TABLE_PREFIX}raw_files"


class RawFileStore:
    """CRUD for raw file storage."""

    def __init__(self, connection_manager: "PostgresConnectionManager", collection: str):
        self._cm = connection_manager
        self._collection = collection

    def upsert(
        self,
        file_id: str,
        path: str,
        content: str,
        content_hash: str,
        file_type: str,
        size_bytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a raw file."""
        meta_json = json.dumps(metadata or {})
        sql = f"""
            INSERT INTO {TABLE} (id, path, content, content_hash, file_type, size_bytes, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                path = EXCLUDED.path,
                content = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash,
                file_type = EXCLUDED.file_type,
                size_bytes = EXCLUDED.size_bytes,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """
        with self._cm.connection(self._collection) as conn:
            conn.execute(
                sql, (file_id, path, content, content_hash, file_type, size_bytes, meta_json)
            )
            conn.commit()

    def get(self, file_id: str) -> dict[str, Any] | None:
        """Get a raw file by ID."""
        sql = f"SELECT id, path, content, content_hash, file_type, size_bytes, metadata FROM {TABLE} WHERE id = %s"
        with self._cm.connection(self._collection) as conn:
            row = conn.execute(sql, (file_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "path": row[1],
            "content": row[2],
            "content_hash": row[3],
            "file_type": row[4],
            "size_bytes": row[5],
            "metadata": row[6] if isinstance(row[6], dict) else json.loads(row[6] or "{}"),
        }

    def get_by_path(self, path: str) -> dict[str, Any] | None:
        """Get a raw file by path."""
        sql = f"SELECT id, path, content, content_hash, file_type, size_bytes, metadata FROM {TABLE} WHERE path = %s"
        with self._cm.connection(self._collection) as conn:
            row = conn.execute(sql, (path,)).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "path": row[1],
            "content": row[2],
            "content_hash": row[3],
            "file_type": row[4],
            "size_bytes": row[5],
            "metadata": row[6] if isinstance(row[6], dict) else json.loads(row[6] or "{}"),
        }

    def delete(self, file_id: str) -> None:
        """Delete a raw file (cascades to symbols + imports)."""
        sql = f"DELETE FROM {TABLE} WHERE id = %s"
        with self._cm.connection(self._collection) as conn:
            conn.execute(sql, (file_id,))
            conn.commit()

    def list_hashes(self) -> dict[str, str]:
        """Return {path: content_hash} for all stored files."""
        sql = f"SELECT path, content_hash FROM {TABLE}"
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql).fetchall()
        return {row[0]: row[1] for row in rows}

    def list_ids_by_path(self) -> dict[str, str]:
        """Return {path: file_id} for all stored files."""
        sql = f"SELECT path, id FROM {TABLE}"
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql).fetchall()
        return {row[0]: row[1] for row in rows}
