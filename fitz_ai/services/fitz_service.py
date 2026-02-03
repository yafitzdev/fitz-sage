# fitz_ai/services/fitz_service.py
"""
FitzService - Unified service layer for all Fitz operations.

This is THE single API that CLI, SDK, and REST API should all use.
By centralizing business logic here, we:
1. Test once, confidence in all interfaces
2. Eliminate code duplication across CLI/SDK/API
3. Ensure consistent behavior everywhere

Design Principles:
- Stateless: No instance state, all state passed as parameters
- Synchronous: Async wrappers added by callers (API)
- Config-driven: FitzConfig passed to operations that need it
- Exception-based: Raises domain exceptions, interfaces translate

Usage:
    from fitz_ai.services import FitzService

    service = FitzService()

    # Query
    answer = service.query("What is RAG?", collection="docs")

    # Ingest
    result = service.ingest("/path/to/docs", collection="docs")

    # Collections
    collections = service.list_collections()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_ai.core import Answer
from fitz_ai.logging.logger import get_logger

if TYPE_CHECKING:
    from fitz_ai.core.chunk import Chunk
    from fitz_ai.retrieval.rewriter.types import ConversationContext

logger = get_logger(__name__)


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class IngestResult:
    """Result of an ingestion operation."""

    collection: str
    documents_processed: int
    chunks_created: int
    chunks_new: int = 0
    chunks_updated: int = 0
    chunks_deleted: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class CollectionInfo:
    """Information about a collection."""

    name: str
    chunk_count: int
    vector_dimensions: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""

    valid: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class HealthCheckResult:
    """Result of system health check."""

    healthy: bool
    components: dict[str, bool] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


# =============================================================================
# Exceptions
# =============================================================================


class FitzServiceError(Exception):
    """Base exception for service errors."""

    pass


class CollectionNotFoundError(FitzServiceError):
    """Collection does not exist."""

    def __init__(self, collection: str):
        self.collection = collection
        super().__init__(f"Collection not found: {collection}")


class ConfigurationError(FitzServiceError):
    """Configuration is invalid or missing."""

    pass


class IngestError(FitzServiceError):
    """Ingestion failed."""

    pass


class QueryError(FitzServiceError):
    """Query failed."""

    pass


# =============================================================================
# FitzService
# =============================================================================


class FitzService:
    """
    Unified service layer for all Fitz operations.

    This class is the single source of truth for business logic.
    CLI, SDK, and API should all call these methods.

    The service is stateless - configuration and collection are passed
    to each method that needs them.
    """

    # =========================================================================
    # Query Operations
    # =========================================================================

    def query(
        self,
        question: str,
        collection: str,
        *,
        top_k: int | None = None,
        conversation_context: "ConversationContext | None" = None,
        engine: str = "fitz_rag",
    ) -> Answer:
        """
        Query the knowledge base.

        Args:
            question: The question to ask
            collection: Collection to query
            top_k: Number of chunks to retrieve (uses config default if None)
            conversation_context: For query rewriting (pronoun resolution)
            engine: Engine to use (default: fitz_rag)

        Returns:
            Answer with text, provenance, and mode

        Raises:
            QueryError: If query fails
            CollectionNotFoundError: If collection doesn't exist
        """
        from fitz_ai.runtime import create_engine

        if not question or not question.strip():
            raise QueryError("Question cannot be empty")

        # Verify collection exists
        if not self._collection_exists(collection):
            raise CollectionNotFoundError(collection)

        try:
            # Create engine with collection override
            engine_instance = create_engine(
                engine,
                collection=collection,
                top_k=top_k,
            )

            # Run query
            answer = engine_instance.answer(
                question,
                conversation_context=conversation_context,
            )

            return answer

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise QueryError(f"Query failed: {e}") from e

    def retrieve(
        self,
        question: str,
        collection: str,
        *,
        top_k: int | None = None,
    ) -> list["Chunk"]:
        """
        Retrieve relevant chunks without generating an answer.

        Useful for debugging retrieval or building custom pipelines.

        Args:
            question: The query
            collection: Collection to search
            top_k: Number of chunks to retrieve

        Returns:
            List of relevant chunks
        """
        from fitz_ai.runtime import create_engine

        if not self._collection_exists(collection):
            raise CollectionNotFoundError(collection)

        try:
            engine = create_engine("fitz_rag", collection=collection, top_k=top_k)

            # Access retrieval directly if available
            if hasattr(engine, "retrieve"):
                return engine.retrieve(question)

            # Fallback: run full pipeline and return chunks from provenance
            answer = engine.answer(question)
            return [p.metadata.get("chunk") for p in answer.provenance if p.metadata.get("chunk")]

        except Exception as e:
            raise QueryError(f"Retrieval failed: {e}") from e

    # =========================================================================
    # Ingestion Operations
    # =========================================================================

    def ingest(
        self,
        source: str | Path,
        collection: str,
        *,
        clear_existing: bool = False,
        enable_enrichment: bool = True,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestResult:
        """
        Ingest documents into a collection.

        Args:
            source: Path to file or directory
            collection: Target collection name
            clear_existing: Delete existing data before ingesting
            enable_enrichment: Run LLM enrichment (summaries, entities, etc.)
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap

        Returns:
            IngestResult with statistics

        Raises:
            IngestError: If ingestion fails
            ValueError: If source doesn't exist
        """
        from fitz_ai.ingestion.diff import run_diff_ingest

        source_path = Path(source)
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")

        try:
            if clear_existing:
                self.delete_collection(collection)

            # Run diff-based ingestion
            result = run_diff_ingest(
                source=str(source_path),
                collection=collection,
                enable_enrichment=enable_enrichment,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            return IngestResult(
                collection=collection,
                documents_processed=result.documents_processed,
                chunks_created=result.new_chunks + result.updated_chunks,
                chunks_new=result.new_chunks,
                chunks_updated=result.updated_chunks,
                chunks_deleted=result.deleted_chunks,
            )

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise IngestError(f"Ingestion failed: {e}") from e

    def ingest_text(
        self,
        text: str,
        collection: str,
        *,
        source_id: str = "direct_text",
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """
        Ingest raw text directly (without file).

        Args:
            text: Text content to ingest
            collection: Target collection
            source_id: Identifier for the text source
            metadata: Additional metadata to attach

        Returns:
            IngestResult with statistics
        """
        from fitz_ai.ingestion.chunking.router import ChunkingRouter
        from fitz_ai.llm import get_embedder
        from fitz_ai.vector_db.registry import get_vector_db_plugin
        from fitz_ai.vector_db.writer import VectorDBWriter

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Create chunks from text
            chunker = ChunkingRouter.from_defaults().get_chunker(".txt")
            from fitz_ai.ingestion.parser.base import ParsedDocument

            doc = ParsedDocument(
                full_text=text,
                sections=[],
                metadata={"source_id": source_id, **(metadata or {})},
            )
            chunks = chunker.chunk(doc)

            # Embed chunks
            embedder = get_embedder()
            vectors = [embedder.embed(c.content) for c in chunks]

            # Store
            vdb = get_vector_db_plugin()
            writer = VectorDBWriter(client=vdb)
            writer.upsert(collection=collection, chunks=chunks, vectors=vectors)

            return IngestResult(
                collection=collection,
                documents_processed=1,
                chunks_created=len(chunks),
                chunks_new=len(chunks),
            )

        except Exception as e:
            raise IngestError(f"Text ingestion failed: {e}") from e

    # =========================================================================
    # Collection Operations
    # =========================================================================

    def list_collections(self) -> list[CollectionInfo]:
        """
        List all collections.

        Returns:
            List of CollectionInfo with names and stats
        """
        from fitz_ai.vector_db.registry import get_vector_db_plugin

        vdb = get_vector_db_plugin()

        if not hasattr(vdb, "list_collections"):
            return []

        names = vdb.list_collections()
        result = []

        for name in names:
            info = CollectionInfo(name=name, chunk_count=0)

            if hasattr(vdb, "count"):
                try:
                    info.chunk_count = vdb.count(name)
                except Exception:
                    pass

            if hasattr(vdb, "get_collection_stats"):
                try:
                    stats = vdb.get_collection_stats(name)
                    info.vector_dimensions = stats.get("vector_size")
                    info.metadata = stats
                except Exception:
                    pass

            result.append(info)

        return result

    def get_collection(self, name: str) -> CollectionInfo:
        """
        Get detailed information about a collection.

        Args:
            name: Collection name

        Returns:
            CollectionInfo with full stats

        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """
        if not self._collection_exists(name):
            raise CollectionNotFoundError(name)

        from fitz_ai.vector_db.registry import get_vector_db_plugin

        vdb = get_vector_db_plugin()

        info = CollectionInfo(name=name, chunk_count=0)

        if hasattr(vdb, "count"):
            info.chunk_count = vdb.count(name)

        if hasattr(vdb, "get_collection_stats"):
            stats = vdb.get_collection_stats(name)
            info.vector_dimensions = stats.get("vector_size")
            info.metadata = stats

        return info

    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.

        Args:
            name: Collection to delete

        Returns:
            True if deleted, False if didn't exist
        """
        from fitz_ai.vector_db.registry import get_vector_db_plugin

        vdb = get_vector_db_plugin()

        if not hasattr(vdb, "delete_collection"):
            raise FitzServiceError("Vector DB does not support deleting collections")

        if not self._collection_exists(name):
            return False

        vdb.delete_collection(name)
        logger.info(f"Deleted collection: {name}")
        return True

    def _collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        from fitz_ai.vector_db.registry import get_vector_db_plugin

        vdb = get_vector_db_plugin()

        if hasattr(vdb, "list_collections"):
            return name in vdb.list_collections()

        # Fallback: try to count
        if hasattr(vdb, "count"):
            try:
                vdb.count(name)
                return True
            except Exception:
                return False

        return True  # Assume exists if we can't check

    # =========================================================================
    # Configuration Operations
    # =========================================================================

    def validate_config(self) -> ConfigValidationResult:
        """
        Validate the current configuration.

        Checks:
        - Config file exists and parses
        - Required plugins are available
        - API keys are set for configured providers
        - Vector DB is accessible

        Returns:
            ConfigValidationResult with issues and warnings
        """
        issues = []
        warnings = []

        # Check config exists
        from fitz_ai.core.paths import FitzPaths

        config_path = FitzPaths.engine_config("fitz_rag")
        if not config_path.exists():
            issues.append(f"Config not found: {config_path}")
            return ConfigValidationResult(valid=False, issues=issues)

        # Try to load
        try:
            from fitz_ai.cli.context import CLIContext

            ctx = CLIContext.load()
        except Exception as e:
            issues.append(f"Config parse error: {e}")
            return ConfigValidationResult(valid=False, issues=issues)

        # Check plugins
        try:
            from fitz_ai.llm import get_chat_factory

            get_chat_factory(ctx.chat_plugin)
        except Exception as e:
            issues.append(f"Chat plugin '{ctx.chat_plugin}' not available: {e}")

        try:
            from fitz_ai.llm import get_embedder

            get_embedder(ctx.embedding_plugin)
        except Exception as e:
            issues.append(f"Embedding plugin '{ctx.embedding_plugin}' not available: {e}")

        # Check vector DB
        try:
            from fitz_ai.vector_db.registry import get_vector_db_plugin

            get_vector_db_plugin(ctx.vector_db_plugin)
        except Exception as e:
            issues.append(f"Vector DB '{ctx.vector_db_plugin}' not available: {e}")

        return ConfigValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
        )

    def get_config_summary(self) -> dict[str, Any]:
        """
        Get a summary of current configuration.

        Returns:
            Dict with plugin names, models, and settings
        """
        try:
            from fitz_ai.cli.context import CLIContext

            ctx = CLIContext.load()

            return {
                "chat": ctx.chat_plugin,
                "chat_model_smart": ctx.chat_model_smart,
                "chat_model_fast": ctx.chat_model_fast,
                "embedding": ctx.embedding_plugin,
                "embedding_model": ctx.embedding_model,
                "rerank": ctx.rerank_plugin,
                "vector_db": ctx.vector_db_plugin,
                "retrieval_plugin": ctx.retrieval_plugin,
                "collection": ctx.retrieval_collection,
                "chunk_size": ctx.chunk_size,
                "chunk_overlap": ctx.chunk_overlap,
            }
        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # Health & Diagnostics
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """
        Check system health.

        Tests connectivity to:
        - Vector database
        - LLM providers (if configured)

        Returns:
            HealthCheckResult with component status
        """
        components = {}
        issues = []

        # Check vector DB
        try:
            from fitz_ai.vector_db.registry import get_vector_db_plugin

            vdb = get_vector_db_plugin()
            if hasattr(vdb, "list_collections"):
                vdb.list_collections()
            components["vector_db"] = True
        except Exception as e:
            components["vector_db"] = False
            issues.append(f"Vector DB: {e}")

        # Check chat provider
        try:
            from fitz_ai.cli.context import CLIContext
            from fitz_ai.llm import get_chat_factory

            ctx = CLIContext.load()
            get_chat_factory(ctx.chat_plugin)  # Verify factory works
            components["chat"] = True
        except Exception as e:
            components["chat"] = False
            issues.append(f"Chat provider: {e}")

        # Check embedder
        try:
            from fitz_ai.llm import get_embedder

            get_embedder()
            components["embedding"] = True
        except Exception as e:
            components["embedding"] = False
            issues.append(f"Embedding provider: {e}")

        return HealthCheckResult(
            healthy=all(components.values()),
            components=components,
            issues=issues,
        )


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_service: FitzService | None = None


def get_service() -> FitzService:
    """Get the default FitzService instance."""
    global _default_service
    if _default_service is None:
        _default_service = FitzService()
    return _default_service
