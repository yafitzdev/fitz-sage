# fitz_ai/backends/local_vector_db/pgvector.py
"""
pgvector-backed Vector Database.

Design principles:
- Unified storage: vectors + payloads in one PostgreSQL instance
- HNSW indexing: 99% recall with zero maintenance
- Per-collection databases for natural sharding
- Support for hybrid search (vector + full-text via tsvector)
- Implements standard VectorDBPlugin contract
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import VECTOR_DB
from fitz_ai.storage import StorageConfig, StorageMode, get_connection_manager
from fitz_ai.vector_db.base import SearchResult

logger = get_logger(__name__)


@dataclass
class _PgRecord:
    """Simple record object for scroll results."""

    id: str
    payload: Dict[str, Any]


class PgVectorDB:
    """
    pgvector-backed VectorDB plugin.

    Key features:
    - HNSW index for fast approximate nearest neighbor search
    - Full payload storage in JSONB columns
    - Per-collection databases (natural sharding)
    - Hybrid search support (vector + BM25 via tsvector)

    Usage:
        db = PgVectorDB()

        # Upsert vectors
        db.upsert("my_collection", [
            {"id": "1", "vector": [0.1] * 1536, "payload": {"content": "..."}}
        ])

        # Search
        results = db.search("my_collection", query_vector, limit=10)
    """

    plugin_name = "pgvector"
    plugin_type = "vector_db"

    # SQL templates
    CREATE_CHUNKS_TABLE = """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            vector vector({dim}),
            payload JSONB NOT NULL DEFAULT '{{}}',
            content_tsv tsvector GENERATED ALWAYS AS (
                to_tsvector('english', COALESCE(payload->>'content', ''))
            ) STORED,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """

    CREATE_HNSW_INDEX = """
        CREATE INDEX IF NOT EXISTS chunks_vector_hnsw_idx
        ON chunks USING hnsw (vector vector_cosine_ops)
        WITH (m = {m}, ef_construction = {ef_construction})
    """

    CREATE_FTS_INDEX = """
        CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx
        ON chunks USING gin (content_tsv)
    """

    CREATE_ID_INDEX = """
        CREATE INDEX IF NOT EXISTS chunks_id_idx ON chunks (id)
    """

    def __init__(
        self,
        *,
        mode: str = "local",
        data_dir: Optional[str | Path] = None,
        connection_string: Optional[str] = None,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
        **kwargs,  # Accept extra kwargs for compatibility
    ):
        """
        Initialize pgvector database.

        Args:
            mode: "local" (pgserver) or "external" (connection_string)
            data_dir: Data directory for pgserver (local mode only)
            connection_string: PostgreSQL URI (external mode only)
            hnsw_m: HNSW max connections per layer (higher = better recall, more memory)
            hnsw_ef_construction: HNSW construction ef (higher = better recall, slower builds)
        """
        # Build storage config
        config = StorageConfig(
            mode=StorageMode(mode),
            data_dir=Path(data_dir) if data_dir else None,
            connection_string=connection_string,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
        )

        self._manager = get_connection_manager(config)
        self._manager.start()

        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._initialized_collections: Dict[str, int] = {}  # collection -> dim

        logger.info(f"{VECTOR_DB} pgvector initialized (mode={mode})")

    def _ensure_schema(self, collection: str, dim: int) -> None:
        """Ensure collection schema exists with correct dimension."""
        if collection in self._initialized_collections:
            existing_dim = self._initialized_collections[collection]
            if existing_dim != dim:
                raise ValueError(
                    f"Dimension mismatch: collection '{collection}' has dim={existing_dim}, "
                    f"but got vectors with dim={dim}"
                )
            return

        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as e:
            raise ImportError(
                "psycopg and pgvector required. Install with: pip install 'psycopg[binary]' pgvector"
            ) from e

        with self._manager.connection(collection) as conn:
            register_vector(conn)

            # Check if table exists and get current dimension
            result = conn.execute(
                """
                SELECT atttypmod - 4 as dim
                FROM pg_attribute
                WHERE attrelid = 'chunks'::regclass
                AND attname = 'vector'
                """
            ).fetchone()

            if result:
                # Table exists, verify dimension
                existing_dim = result[0]
                if existing_dim != dim:
                    raise ValueError(
                        f"Dimension mismatch: collection '{collection}' has dim={existing_dim}, "
                        f"but got vectors with dim={dim}. Drop the collection to change dimensions."
                    )
                self._initialized_collections[collection] = existing_dim
                logger.debug(f"{VECTOR_DB} Using existing collection '{collection}' (dim={existing_dim})")
                return

        # Table doesn't exist, create it
        with self._manager.connection(collection) as conn:
            register_vector(conn)

            # Create table
            conn.execute(self.CREATE_CHUNKS_TABLE.format(dim=dim))

            # Create HNSW index
            conn.execute(
                self.CREATE_HNSW_INDEX.format(
                    m=self._hnsw_m, ef_construction=self._hnsw_ef_construction
                )
            )

            # Create full-text search index
            conn.execute(self.CREATE_FTS_INDEX)

            # Create ID index for fast lookups
            conn.execute(self.CREATE_ID_INDEX)

            conn.commit()

        self._initialized_collections[collection] = dim
        logger.info(f"{VECTOR_DB} Created collection '{collection}' (dim={dim})")

    def _build_filter_clause(
        self, filter_cond: Optional[Dict[str, Any]]
    ) -> Tuple[str, List[Any]]:
        """Build SQL WHERE clause from Qdrant-style filter."""
        if not filter_cond:
            return "", []

        conditions = []
        params = []

        def process_condition(cond: Dict[str, Any]) -> None:
            # Handle nested must/should
            if "must" in cond:
                for sub_cond in cond["must"]:
                    process_condition(sub_cond)
                return

            if "should" in cond:
                or_clauses = []
                for sub_cond in cond["should"]:
                    clause, sub_params = self._build_single_condition(sub_cond)
                    if clause:
                        or_clauses.append(clause)
                        params.extend(sub_params)
                if or_clauses:
                    conditions.append(f"({' OR '.join(or_clauses)})")
                return

            # Direct condition
            clause, sub_params = self._build_single_condition(cond)
            if clause:
                conditions.append(clause)
                params.extend(sub_params)

        process_condition(filter_cond)

        if conditions:
            return "WHERE " + " AND ".join(conditions), params
        return "", []

    def _build_single_condition(
        self, cond: Dict[str, Any]
    ) -> Tuple[str, List[Any]]:
        """Build single filter condition."""
        key = cond.get("key")
        if not key:
            return "", []

        # Support nested key lookup via JSONB path
        if "." in key:
            parts = key.split(".")
            json_path = "payload"
            for part in parts[:-1]:
                json_path = f"{json_path}->'{part}'"
            json_path = f"{json_path}->>'{parts[-1]}'"
        else:
            json_path = f"payload->>'{key}'"

        if "match" in cond:
            value = cond["match"].get("value")
            return f"{json_path} = %s", [str(value)]

        if "range" in cond:
            range_cond = cond["range"]
            clauses = []
            params = []
            if "gte" in range_cond:
                clauses.append(f"({json_path})::numeric >= %s")
                params.append(range_cond["gte"])
            if "gt" in range_cond:
                clauses.append(f"({json_path})::numeric > %s")
                params.append(range_cond["gt"])
            if "lte" in range_cond:
                clauses.append(f"({json_path})::numeric <= %s")
                params.append(range_cond["lte"])
            if "lt" in range_cond:
                clauses.append(f"({json_path})::numeric < %s")
                params.append(range_cond["lt"])
            return " AND ".join(clauses), params

        return "", []

    # =========================================================================
    # Public API - Standard VectorDBPlugin Contract
    # =========================================================================

    def upsert(
        self,
        collection: str,
        points: List[Dict[str, Any]],
        defer_persist: bool = False,
    ) -> None:
        """
        Upsert points into a collection.

        Args:
            collection: Collection name
            points: List of points with 'id', 'vector', and 'payload' keys
            defer_persist: Ignored for pgvector (auto-committed)
        """
        if not points:
            return

        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as e:
            raise ImportError(
                "psycopg and pgvector required. Install with: pip install 'psycopg[binary]' pgvector"
            ) from e

        # Auto-detect dimension from first vector
        first_vector = points[0]["vector"]
        dim = len(first_vector)
        self._ensure_schema(collection, dim)

        with self._manager.connection(collection) as conn:
            register_vector(conn)

            # Batch upsert using executemany
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks (id, vector, payload)
                    VALUES (%(id)s, %(vector)s, %(payload)s)
                    ON CONFLICT (id) DO UPDATE SET
                        vector = EXCLUDED.vector,
                        payload = EXCLUDED.payload
                    """,
                    [
                        {
                            "id": str(p["id"]),
                            "vector": p["vector"],
                            "payload": psycopg.types.json.Json(p.get("payload", {})),
                        }
                        for p in points
                    ],
                )
            conn.commit()

        logger.debug(f"{VECTOR_DB} Upserted {len(points)} points to '{collection}'")

    def flush(self) -> None:
        """No-op for pgvector (auto-committed)."""
        pass

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int,
        with_payload: bool = True,
        query_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors using HNSW index.

        Args:
            collection_name: Collection to search
            query_vector: Query embedding vector
            limit: Maximum number of results
            with_payload: Whether to include payload in results
            query_filter: Optional Qdrant-style metadata filter

        Returns:
            List of SearchResult objects, ordered by similarity (highest first)
        """
        if collection_name not in self._initialized_collections:
            # Try to discover existing collection
            self._discover_collection(collection_name)
            if collection_name not in self._initialized_collections:
                return []

        try:
            from pgvector.psycopg import register_vector
        except ImportError:
            return []

        with self._manager.connection(collection_name) as conn:
            register_vector(conn)

            # Build filter clause
            where_clause, filter_params = self._build_filter_clause(query_filter)

            # Cosine similarity: 1 - cosine_distance
            # pgvector <=> operator returns cosine distance
            if where_clause:
                sql = f"""
                    SELECT id, 1 - (vector <=> %s) as score, payload
                    FROM chunks
                    {where_clause}
                    ORDER BY vector <=> %s
                    LIMIT %s
                """
                params = [query_vector] + filter_params + [query_vector, limit]
            else:
                sql = """
                    SELECT id, 1 - (vector <=> %s) as score, payload
                    FROM chunks
                    ORDER BY vector <=> %s
                    LIMIT %s
                """
                params = [query_vector, query_vector, limit]

            cursor = conn.execute(sql, params)
            results = []

            for row in cursor:
                results.append(
                    SearchResult(
                        id=row[0],
                        score=float(row[1]) if row[1] is not None else 0.0,
                        payload=dict(row[2]) if with_payload and row[2] else {},
                    )
                )

            return results

    def hybrid_search(
        self,
        collection_name: str,
        query_vector: List[float],
        query_text: str,
        limit: int,
        alpha: float = 0.5,
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector similarity and BM25.

        Uses RRF (Reciprocal Rank Fusion) to combine results.

        Args:
            collection_name: Collection to search
            query_vector: Dense embedding vector
            query_text: Text query for BM25
            limit: Number of results
            alpha: Weight for vector vs text (0=text only, 1=vector only)

        Returns:
            List of SearchResult objects
        """
        if collection_name not in self._initialized_collections:
            return []

        try:
            from pgvector.psycopg import register_vector
        except ImportError:
            return []

        with self._manager.connection(collection_name) as conn:
            register_vector(conn)

            # RRF fusion query
            sql = """
                WITH vector_results AS (
                    SELECT id, ROW_NUMBER() OVER (ORDER BY vector <=> %s) as rank
                    FROM chunks
                    LIMIT %s
                ),
                text_results AS (
                    SELECT id, ROW_NUMBER() OVER (
                        ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) DESC
                    ) as rank
                    FROM chunks
                    WHERE content_tsv @@ plainto_tsquery('english', %s)
                    LIMIT %s
                ),
                rrf AS (
                    SELECT
                        COALESCE(v.id, t.id) as id,
                        (COALESCE(%s / (60.0 + v.rank), 0) +
                         COALESCE(%s / (60.0 + t.rank), 0)) as rrf_score
                    FROM vector_results v
                    FULL OUTER JOIN text_results t ON v.id = t.id
                )
                SELECT c.id, r.rrf_score, c.payload
                FROM rrf r
                JOIN chunks c ON r.id = c.id
                ORDER BY r.rrf_score DESC
                LIMIT %s
            """

            vector_weight = alpha
            text_weight = 1 - alpha
            search_multiplier = 2  # Search more to account for RRF fusion

            cursor = conn.execute(
                sql,
                [
                    query_vector,
                    limit * search_multiplier,
                    query_text,
                    query_text,
                    limit * search_multiplier,
                    vector_weight,
                    text_weight,
                    limit,
                ],
            )

            return [
                SearchResult(
                    id=row[0],
                    score=float(row[1]) if row[1] else 0.0,
                    payload=dict(row[2]) if row[2] else {},
                )
                for row in cursor
            ]

    def retrieve(
        self,
        collection_name: str,
        ids: List[str],
        with_payload: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve points by their IDs."""
        if not ids or collection_name not in self._initialized_collections:
            return []

        with self._manager.connection(collection_name) as conn:
            cursor = conn.execute(
                "SELECT id, payload FROM chunks WHERE id = ANY(%s)", (ids,)
            )
            return [
                {"id": row[0], "payload": dict(row[1]) if with_payload and row[1] else {}}
                for row in cursor
            ]

    def list_collections(self) -> List[str]:
        """List all known collections."""
        return sorted(self._initialized_collections.keys())

    def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        if collection not in self._initialized_collections:
            self._discover_collection(collection)
            if collection not in self._initialized_collections:
                return {
                    "points_count": 0,
                    "vectors_count": 0,
                    "status": "not_found",
                    "vector_size": None,
                }

        with self._manager.connection(collection) as conn:
            result = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            count = result[0] if result else 0

            return {
                "points_count": count,
                "vectors_count": count,
                "status": "ready",
                "vector_size": self._initialized_collections.get(collection),
            }

    def scroll(
        self,
        collection: str,
        limit: int = 10,
        offset: int = 0,
    ) -> Tuple[List[_PgRecord], Optional[int]]:
        """Scroll through records in a collection."""
        if collection not in self._initialized_collections:
            return [], None

        with self._manager.connection(collection) as conn:
            cursor = conn.execute(
                "SELECT id, payload FROM chunks ORDER BY id LIMIT %s OFFSET %s",
                (limit, offset),
            )
            records = [_PgRecord(row[0], dict(row[1]) if row[1] else {}) for row in cursor]

            next_offset = offset + limit if len(records) == limit else None
            return records, next_offset

    def scroll_with_vectors(
        self,
        collection: str,
        limit: int = 10,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """Scroll through records including vectors."""
        if collection not in self._initialized_collections:
            return [], None

        try:
            from pgvector.psycopg import register_vector
        except ImportError:
            return [], None

        with self._manager.connection(collection) as conn:
            register_vector(conn)

            cursor = conn.execute(
                "SELECT id, payload, vector FROM chunks ORDER BY id LIMIT %s OFFSET %s",
                (limit, offset),
            )
            records = [
                {
                    "id": row[0],
                    "payload": dict(row[1]) if row[1] else {},
                    "vector": list(row[2]) if row[2] else [],
                }
                for row in cursor
            ]

            next_offset = offset + limit if len(records) == limit else None
            return records, next_offset

    def count(self, collection: Optional[str] = None) -> int:
        """Return the number of vectors."""
        if collection is None:
            total = 0
            for coll in self._initialized_collections:
                total += self.count(coll)
            return total

        if collection not in self._initialized_collections:
            return 0

        with self._manager.connection(collection) as conn:
            result = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return result[0] if result else 0

    def delete_collection(self, collection: str) -> int:
        """Delete a collection entirely."""
        if collection not in self._initialized_collections:
            return 0

        with self._manager.connection(collection) as conn:
            # Count before delete
            result = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            count = result[0] if result else 0

            # Drop tables
            conn.execute("DROP TABLE IF EXISTS chunks CASCADE")
            conn.execute("DROP TABLE IF EXISTS tables CASCADE")
            conn.commit()

        del self._initialized_collections[collection]
        logger.info(f"{VECTOR_DB} Deleted collection '{collection}' ({count} vectors)")
        return count

    def _discover_collection(self, collection: str) -> bool:
        """Try to discover an existing collection's dimension."""
        try:
            with self._manager.connection(collection) as conn:
                # Check if chunks table exists and get dimension
                result = conn.execute(
                    """
                    SELECT atttypmod - 4 as dim
                    FROM pg_attribute
                    WHERE attrelid = 'chunks'::regclass
                    AND attname = 'vector'
                    """
                ).fetchone()

                if result and result[0]:
                    self._initialized_collections[collection] = result[0]
                    logger.debug(
                        f"{VECTOR_DB} Discovered collection '{collection}' (dim={result[0]})"
                    )
                    return True
        except Exception:
            pass
        return False


__all__ = ["PgVectorDB"]
