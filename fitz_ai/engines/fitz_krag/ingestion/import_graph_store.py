# fitz_ai/engines/fitz_krag/ingestion/import_graph_store.py
"""CRUD operations for krag_import_graph table."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fitz_ai.engines.fitz_krag.ingestion.schema import TABLE_PREFIX

if TYPE_CHECKING:
    from fitz_ai.storage.postgres import PostgresConnectionManager

logger = logging.getLogger(__name__)

TABLE = f"{TABLE_PREFIX}import_graph"


class ImportGraphStore:
    """CRUD for the import graph."""

    def __init__(self, connection_manager: "PostgresConnectionManager", collection: str):
        self._cm = connection_manager
        self._collection = collection

    def upsert_batch(self, edges: list[dict[str, Any]]) -> None:
        """Insert or update a batch of import edges."""
        if not edges:
            return

        sql = f"""
            INSERT INTO {TABLE} (source_file_id, target_module, target_file_id, import_names)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (source_file_id, target_module) DO UPDATE SET
                target_file_id = EXCLUDED.target_file_id,
                import_names = EXCLUDED.import_names
        """
        with self._cm.connection(self._collection) as conn:
            for edge in edges:
                conn.execute(
                    sql,
                    (
                        edge["source_file_id"],
                        edge["target_module"],
                        edge.get("target_file_id"),
                        edge.get("import_names", []),
                    ),
                )
            conn.commit()

    def get_imports(self, file_id: str) -> list[dict[str, Any]]:
        """Get all imports from a file."""
        sql = f"""
            SELECT source_file_id, target_module, target_file_id, import_names
            FROM {TABLE} WHERE source_file_id = %s
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (file_id,)).fetchall()
        return [
            {
                "source_file_id": row[0],
                "target_module": row[1],
                "target_file_id": row[2],
                "import_names": row[3] or [],
            }
            for row in rows
        ]

    def get_importers(self, file_id: str) -> list[dict[str, Any]]:
        """Get all files that import this file."""
        sql = f"""
            SELECT source_file_id, target_module, target_file_id, import_names
            FROM {TABLE} WHERE target_file_id = %s
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (file_id,)).fetchall()
        return [
            {
                "source_file_id": row[0],
                "target_module": row[1],
                "target_file_id": row[2],
                "import_names": row[3] or [],
            }
            for row in rows
        ]

    def delete_by_file(self, file_id: str) -> None:
        """Delete all import edges for a file (as source)."""
        sql = f"DELETE FROM {TABLE} WHERE source_file_id = %s"
        with self._cm.connection(self._collection) as conn:
            conn.execute(sql, (file_id,))
            conn.commit()
