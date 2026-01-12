# fitz_ai/retrieval/entity_graph/store.py
"""Persistent entity-to-chunk graph for related chunk discovery."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


class EntityGraphStore:
    """
    Persistent entity-to-chunk graph.

    Stores relationships between entities and the chunks that mention them.
    Enables fast discovery of related chunks via shared entities at query time.

    Storage: SQLite database per collection.
    """

    def __init__(self, collection: str):
        """
        Initialize entity graph store.

        Args:
            collection: Collection name for namespacing
        """
        self.collection = collection
        self._conn: sqlite3.Connection | None = None

    @property
    def db_path(self) -> Path:
        """Path to the SQLite database file."""
        return FitzPaths.entity_graph(self.collection)

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._ensure_schema()
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables schema if not exists."""
        self.conn.executescript(
            """
            -- Entity definitions
            CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                entity_type TEXT,
                mention_count INTEGER DEFAULT 1
            );

            -- Entity-to-chunk mapping (graph edges)
            CREATE TABLE IF NOT EXISTS entity_chunks (
                entity_name TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                PRIMARY KEY (entity_name, chunk_id)
            );

            -- Index for fast chunk → entities lookup
            CREATE INDEX IF NOT EXISTS idx_chunk_entities
            ON entity_chunks(chunk_id);

            -- Index for fast entity → chunks lookup
            CREATE INDEX IF NOT EXISTS idx_entity_chunks
            ON entity_chunks(entity_name);
        """
        )
        self.conn.commit()

    # =========================================================================
    # Write Operations (Ingestion Time)
    # =========================================================================

    def add_chunk_entities(
        self,
        chunk_id: str,
        entities: list[tuple[str, str]],
    ) -> None:
        """
        Register entities found in a chunk.

        Args:
            chunk_id: The chunk identifier
            entities: List of (display_name, entity_type) tuples
        """
        if not entities:
            return

        for display_name, entity_type in entities:
            normalized = self._normalize(display_name)
            if not normalized:
                continue

            # Upsert entity
            self.conn.execute(
                """
                INSERT INTO entities (name, display_name, entity_type, mention_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(name) DO UPDATE SET mention_count = mention_count + 1
            """,
                (normalized, display_name, entity_type),
            )

            # Add edge
            self.conn.execute(
                """
                INSERT OR IGNORE INTO entity_chunks (entity_name, chunk_id)
                VALUES (?, ?)
            """,
                (normalized, chunk_id),
            )

        self.conn.commit()
        logger.debug(f"Added {len(entities)} entities for chunk {chunk_id[:8]}...")

    def remove_chunk(self, chunk_id: str) -> None:
        """
        Remove all entity associations for a chunk.

        Used when re-ingesting a file to clear stale associations.

        Args:
            chunk_id: The chunk identifier to remove
        """
        self.conn.execute(
            "DELETE FROM entity_chunks WHERE chunk_id = ?", (chunk_id,)
        )
        self.conn.commit()

    def remove_chunks(self, chunk_ids: list[str]) -> None:
        """
        Remove entity associations for multiple chunks.

        Args:
            chunk_ids: List of chunk identifiers to remove
        """
        if not chunk_ids:
            return

        placeholders = ",".join("?" * len(chunk_ids))
        self.conn.execute(
            f"DELETE FROM entity_chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        self.conn.commit()

    # =========================================================================
    # Read Operations (Query Time)
    # =========================================================================

    def get_related_chunks(
        self,
        chunk_ids: list[str],
        max_total: int = 20,
        min_shared_entities: int = 1,
    ) -> list[str]:
        """
        Find chunks related to given chunks via shared entities.

        Algorithm:
        1. Get all entities from input chunks
        2. Find other chunks mentioning those entities
        3. Rank by entity overlap count
        4. Return top chunks (excluding input chunks)

        Args:
            chunk_ids: Input chunk IDs to find relations for
            max_total: Maximum number of related chunks to return
            min_shared_entities: Minimum shared entities to consider related

        Returns:
            List of related chunk IDs, ordered by relevance (shared entity count)
        """
        if not chunk_ids:
            return []

        # Get entities from input chunks
        placeholders = ",".join("?" * len(chunk_ids))
        cursor = self.conn.execute(
            f"""
            SELECT DISTINCT entity_name
            FROM entity_chunks
            WHERE chunk_id IN ({placeholders})
        """,
            chunk_ids,
        )
        entities = [row[0] for row in cursor.fetchall()]

        if not entities:
            return []

        # Find related chunks, ranked by number of shared entities
        entity_placeholders = ",".join("?" * len(entities))
        chunk_placeholders = ",".join("?" * len(chunk_ids))

        cursor = self.conn.execute(
            f"""
            SELECT chunk_id, COUNT(DISTINCT entity_name) as shared_count
            FROM entity_chunks
            WHERE entity_name IN ({entity_placeholders})
              AND chunk_id NOT IN ({chunk_placeholders})
            GROUP BY chunk_id
            HAVING shared_count >= ?
            ORDER BY shared_count DESC
            LIMIT ?
        """,
            [*entities, *chunk_ids, min_shared_entities, max_total],
        )

        return [row[0] for row in cursor.fetchall()]

    def get_chunks_for_entity(self, entity: str, limit: int = 10) -> list[str]:
        """
        Get chunks mentioning a specific entity.

        Args:
            entity: Entity name to search for
            limit: Maximum chunks to return

        Returns:
            List of chunk IDs mentioning this entity
        """
        normalized = self._normalize(entity)
        cursor = self.conn.execute(
            """
            SELECT chunk_id FROM entity_chunks
            WHERE entity_name = ?
            LIMIT ?
        """,
            (normalized, limit),
        )
        return [row[0] for row in cursor.fetchall()]

    def get_chunks_for_entities(
        self,
        entities: list[str],
        limit: int = 20,
    ) -> list[str]:
        """
        Get chunks mentioning any of the given entities.

        Args:
            entities: Entity names to search for
            limit: Maximum chunks to return

        Returns:
            List of chunk IDs, ordered by number of matching entities
        """
        if not entities:
            return []

        normalized = [self._normalize(e) for e in entities if self._normalize(e)]
        if not normalized:
            return []

        placeholders = ",".join("?" * len(normalized))
        cursor = self.conn.execute(
            f"""
            SELECT chunk_id, COUNT(DISTINCT entity_name) as match_count
            FROM entity_chunks
            WHERE entity_name IN ({placeholders})
            GROUP BY chunk_id
            ORDER BY match_count DESC
            LIMIT ?
        """,
            [*normalized, limit],
        )
        return [row[0] for row in cursor.fetchall()]

    def get_entities_for_chunk(self, chunk_id: str) -> list[dict]:
        """
        Get all entities mentioned in a chunk.

        Args:
            chunk_id: The chunk identifier

        Returns:
            List of entity dicts with name, display_name, type
        """
        cursor = self.conn.execute(
            """
            SELECT e.name, e.display_name, e.entity_type
            FROM entity_chunks ec
            JOIN entities e ON ec.entity_name = e.name
            WHERE ec.chunk_id = ?
        """,
            (chunk_id,),
        )
        return [
            {"name": row[0], "display_name": row[1], "type": row[2]}
            for row in cursor.fetchall()
        ]

    def get_entities_for_chunks(self, chunk_ids: list[str]) -> dict[str, list[str]]:
        """
        Get entities for multiple chunks.

        Args:
            chunk_ids: List of chunk identifiers

        Returns:
            Dict mapping chunk_id → list of entity display names
        """
        if not chunk_ids:
            return {}

        placeholders = ",".join("?" * len(chunk_ids))
        cursor = self.conn.execute(
            f"""
            SELECT ec.chunk_id, e.display_name
            FROM entity_chunks ec
            JOIN entities e ON ec.entity_name = e.name
            WHERE ec.chunk_id IN ({placeholders})
        """,
            chunk_ids,
        )

        result: dict[str, list[str]] = {cid: [] for cid in chunk_ids}
        for row in cursor.fetchall():
            result[row[0]].append(row[1])
        return result

    # =========================================================================
    # Utilities
    # =========================================================================

    def _normalize(self, name: str) -> str:
        """
        Normalize entity name for matching.

        Lowercases and strips whitespace for consistent lookups.
        """
        if not name:
            return ""
        return name.lower().strip()

    def stats(self) -> dict:
        """
        Get graph statistics.

        Returns:
            Dict with entity count, edge count, top entities
        """
        entities = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        edges = self.conn.execute("SELECT COUNT(*) FROM entity_chunks").fetchone()[0]

        # Top entities by mention count
        top_entities = self.conn.execute(
            """
            SELECT display_name, mention_count
            FROM entities
            ORDER BY mention_count DESC
            LIMIT 10
        """
        ).fetchall()

        return {
            "entities": entities,
            "edges": edges,
            "top_entities": [
                {"name": row[0], "mentions": row[1]} for row in top_entities
            ],
        }

    def clear(self) -> None:
        """Clear all data from the graph."""
        self.conn.execute("DELETE FROM entity_chunks")
        self.conn.execute("DELETE FROM entities")
        self.conn.commit()
        logger.info(f"Cleared entity graph for collection {self.collection}")

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
