# fitz_ai/engines/fitz_krag/ingestion/section_store.py
"""CRUD operations for krag_section_index table with BM25 + vector search."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fitz_ai.engines.fitz_krag.ingestion.schema import TABLE_PREFIX

if TYPE_CHECKING:
    from fitz_ai.storage.postgres import PostgresConnectionManager

logger = logging.getLogger(__name__)

TABLE = f"{TABLE_PREFIX}section_index"


class SectionStore:
    """CRUD for the section index."""

    def __init__(self, connection_manager: "PostgresConnectionManager", collection: str):
        self._cm = connection_manager
        self._collection = collection

    def upsert_batch(self, sections: list[dict[str, Any]]) -> None:
        """Insert or update a batch of sections."""
        if not sections:
            return

        sql = f"""
            INSERT INTO {TABLE}
                (id, raw_file_id, title, level, page_start, page_end,
                 content, summary, summary_vector, parent_section_id,
                 position, keywords, entities, metadata)
            VALUES
                (%s, %s, %s, %s, %s, %s,
                 %s, %s, %s::vector, %s,
                 %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                level = EXCLUDED.level,
                page_start = EXCLUDED.page_start,
                page_end = EXCLUDED.page_end,
                content = EXCLUDED.content,
                summary = EXCLUDED.summary,
                summary_vector = EXCLUDED.summary_vector,
                parent_section_id = EXCLUDED.parent_section_id,
                position = EXCLUDED.position,
                keywords = EXCLUDED.keywords,
                entities = EXCLUDED.entities,
                metadata = EXCLUDED.metadata
        """
        with self._cm.connection(self._collection) as conn:
            for sec in sections:
                vector_str = _vector_to_pg(sec.get("summary_vector"))
                conn.execute(
                    sql,
                    (
                        sec["id"],
                        sec["raw_file_id"],
                        sec["title"],
                        sec["level"],
                        sec.get("page_start"),
                        sec.get("page_end"),
                        sec["content"],
                        sec.get("summary"),
                        vector_str,
                        sec.get("parent_section_id"),
                        sec["position"],
                        sec.get("keywords", []),
                        json.dumps(sec.get("entities", [])),
                        json.dumps(sec.get("metadata", {})),
                    ),
                )
            conn.commit()

    def search_bm25(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search using ts_rank on content_tsv."""
        sql = f"""
            SELECT id, raw_file_id, title, level, page_start, page_end,
                   content, summary, parent_section_id, position, metadata,
                   ts_rank(content_tsv, plainto_tsquery('english', %s)) AS rank
            FROM {TABLE}
            WHERE content_tsv @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (query, query, limit)).fetchall()
        results = []
        for row in rows:
            d = _row_to_dict(row[:11])
            d["bm25_score"] = float(row[11]) if row[11] is not None else 0.0
            results.append(d)
        return results

    def search_by_vector(self, vector: list[float], limit: int = 20) -> list[dict[str, Any]]:
        """Semantic search using cosine distance on summary_vector."""
        vector_str = _vector_to_pg(vector)
        if not vector_str:
            return []

        sql = f"""
            SELECT id, raw_file_id, title, level, page_start, page_end,
                   content, summary, parent_section_id, position, metadata,
                   1 - (summary_vector <=> %s::vector) AS score
            FROM {TABLE}
            WHERE summary_vector IS NOT NULL
            ORDER BY summary_vector <=> %s::vector
            LIMIT %s
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (vector_str, vector_str, limit)).fetchall()
        results = []
        for row in rows:
            d = _row_to_dict(row[:11])
            d["score"] = float(row[11]) if row[11] is not None else 0.0
            results.append(d)
        return results

    def get(self, section_id: str) -> dict[str, Any] | None:
        """Get a section by ID."""
        sql = f"""
            SELECT id, raw_file_id, title, level, page_start, page_end,
                   content, summary, parent_section_id, position, metadata
            FROM {TABLE} WHERE id = %s
        """
        with self._cm.connection(self._collection) as conn:
            row = conn.execute(sql, (section_id,)).fetchone()
        if not row:
            return None
        return _row_to_dict(row)

    def get_by_file(self, raw_file_id: str) -> list[dict[str, Any]]:
        """Get all sections for a file, ordered by position."""
        sql = f"""
            SELECT id, raw_file_id, title, level, page_start, page_end,
                   content, summary, parent_section_id, position, metadata
            FROM {TABLE}
            WHERE raw_file_id = %s
            ORDER BY position
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (raw_file_id,)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_children(self, section_id: str) -> list[dict[str, Any]]:
        """Get child sections of a parent."""
        sql = f"""
            SELECT id, raw_file_id, title, level, page_start, page_end,
                   content, summary, parent_section_id, position, metadata
            FROM {TABLE}
            WHERE parent_section_id = %s
            ORDER BY position
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (section_id,)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_siblings(self, section_id: str) -> list[dict[str, Any]]:
        """Get sibling sections (same parent)."""
        section = self.get(section_id)
        if not section:
            return []
        parent_id = section.get("parent_section_id")
        if parent_id:
            return self.get_children(parent_id)
        # Top-level sections: get all level-1 sections for same file
        sql = f"""
            SELECT id, raw_file_id, title, level, page_start, page_end,
                   content, summary, parent_section_id, position, metadata
            FROM {TABLE}
            WHERE raw_file_id = %s AND parent_section_id IS NULL
            ORDER BY position
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (section["raw_file_id"],)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def delete_by_file(self, raw_file_id: str) -> None:
        """Delete all sections for a file."""
        sql = f"DELETE FROM {TABLE} WHERE raw_file_id = %s"
        with self._cm.connection(self._collection) as conn:
            conn.execute(sql, (raw_file_id,))
            conn.commit()


def _vector_to_pg(vector: list[float] | None) -> str | None:
    """Convert a float list to pgvector string format."""
    if not vector:
        return None
    return "[" + ",".join(str(v) for v in vector) + "]"


def _row_to_dict(row: tuple) -> dict[str, Any]:
    """Convert a query row to dict."""
    meta = row[10]
    if isinstance(meta, str):
        meta = json.loads(meta)
    elif meta is None:
        meta = {}
    return {
        "id": row[0],
        "raw_file_id": row[1],
        "title": row[2],
        "level": row[3],
        "page_start": row[4],
        "page_end": row[5],
        "content": row[6],
        "summary": row[7],
        "parent_section_id": row[8],
        "position": row[9],
        "metadata": meta,
    }
