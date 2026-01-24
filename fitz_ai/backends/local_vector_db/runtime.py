from fitz_ai.backends.local_vector_db.config import LocalVectorDBConfig
from fitz_ai.backends.local_vector_db.pgvector import PgVectorDB
from fitz_ai.vector_db.base import VectorDBPlugin


class LocalVectorDBRuntime:
    """
    Thin runtime wrapper to build a local VectorDB.

    Uses pgvector for unified PostgreSQL storage.
    """

    def __init__(self, config: LocalVectorDBConfig | None = None):
        self._config = config or LocalVectorDBConfig()

    def build(self, *, embedding_dim: int = 0) -> VectorDBPlugin:
        # pgvector auto-detects dimension on first upsert
        return PgVectorDB()
