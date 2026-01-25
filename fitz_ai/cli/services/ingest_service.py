# fitz_ai/cli/services/ingest_service.py
"""Service layer for ingest command - handles document ingestion logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ContentDetectionResult:
    """Result of content type detection."""

    content_type: str  # "codebase" or "documents"
    reason: str


@dataclass
class IngestResult:
    """Result of ingestion operation."""

    chunks_added: int
    chunks_updated: int
    chunks_deleted: int
    total_docs: int
    duration_sec: float
    collection: str


class IngestService:
    """
    Business logic for ingest command.

    Handles document ingestion, content detection, and artifact generation
    without UI concerns. Can be used by CLI, SDK, or API.
    """

    def suggest_collection_name(self, source: str) -> str:
        """Suggest a collection name from source path."""
        path = Path(source).resolve()
        name = path.name if path.is_dir() else path.parent.name
        # Sanitize: replace spaces/special chars with underscores
        return name.replace(" ", "_").replace("-", "_").lower()

    def detect_content_type(self, source: str) -> ContentDetectionResult:
        """
        Detect whether source is a codebase or document corpus.

        Returns:
            ContentDetectionResult with content_type and reason
        """
        from fitz_ai.ingestion.detection import detect_content_type

        result = detect_content_type(Path(source))
        return ContentDetectionResult(
            content_type=result.content_type,
            reason=result.reason,
        )

    def is_code_project(self, source: str) -> bool:
        """Check if source is a codebase."""
        result = self.detect_content_type(source)
        return result.content_type == "codebase"

    def get_available_artifacts(self, has_llm: bool = False) -> list[tuple[str, str]]:
        """
        Get available artifact plugins.

        Args:
            has_llm: Whether an LLM client is available

        Returns:
            List of (name, description) tuples
        """
        from fitz_ai.ingestion.enrichment.artifacts.registry import get_artifact_registry

        registry = get_artifact_registry()
        result = []

        for name in registry.list_plugin_names():
            info = registry.get_plugin(name)
            if info is None:
                continue

            # Mark LLM-requiring artifacts
            if info.requires_llm and not has_llm:
                desc = f"{info.description} (requires LLM)"
            else:
                desc = info.description

            result.append((name, desc))

        return result

    def parse_artifact_selection(
        self, artifacts_arg: str | None, available: list[str]
    ) -> list[str] | None:
        """
        Parse artifact selection from CLI argument.

        Args:
            artifacts_arg: Comma-separated artifact names or "all" / "none"
            available: List of available artifact names

        Returns:
            List of selected artifacts, or None if not specified
        """
        if not artifacts_arg:
            return None

        if artifacts_arg.lower() == "none":
            return []

        if artifacts_arg.lower() == "all":
            return available

        # Parse comma-separated list
        selected = [name.strip() for name in artifacts_arg.split(",")]

        # Validate all are available
        invalid = [name for name in selected if name not in available]
        if invalid:
            raise ValueError(f"Invalid artifact(s): {', '.join(invalid)}")

        return selected

    def run_ingestion(
        self,
        source: str,
        collection: str,
        engine_name: str,
        artifacts: list[str] | None = None,
        hierarchy: bool = False,
        force_full: bool = False,
    ) -> IngestResult:
        """
        Run document ingestion.

        Args:
            source: Path to ingest or literal text
            collection: Collection name
            engine_name: Engine to use
            artifacts: List of artifact plugins to run
            hierarchy: Whether to enable hierarchical summarization
            force_full: Whether to force full ingestion (no incremental)

        Returns:
            IngestResult with statistics
        """
        import time

        from fitz_ai.engines.fitz_rag.config import FitzRagConfig
        from fitz_ai.engines.fitz_rag.config.schema import (
            ChunkingRouterConfig,
            ExtensionChunkerConfig,
        )
        from fitz_ai.ingestion.chunking.router import ChunkingRouter
        from fitz_ai.ingestion.diff import run_diff_ingest
        from fitz_ai.ingestion.enrichment.config import ArtifactConfig, EnrichmentConfig
        from fitz_ai.ingestion.enrichment.pipeline import EnrichmentPipeline
        from fitz_ai.ingestion.parser import ParserRouter
        from fitz_ai.ingestion.state import IngestStateManager
        from fitz_ai.llm.factory import get_chat_factory
        from fitz_ai.llm.registry import get_llm_plugin
        from fitz_ai.runtime import get_engine_registry
        from fitz_ai.vector_db.registry import get_vector_db_plugin

        start_time = time.time()

        # Load engine config
        registry = get_engine_registry()
        _ = registry.get_metadata(engine_name)  # Validate engine exists

        if engine_name == "fitz_rag":
            # Load config
            from fitz_ai.core.paths import FitzPaths

            paths = FitzPaths()
            config_path = paths.config_dir / "fitz_rag.yaml"

            if config_path.exists():
                import yaml

                with config_path.open("r", encoding="utf-8") as f:
                    config_dict = yaml.safe_load(f) or {}
            else:
                # Use defaults
                from fitz_ai.engines.fitz_rag.config import load_default_config

                config_dict = load_default_config()

            # Override collection
            if "retrieval" not in config_dict:
                config_dict["retrieval"] = {}
            config_dict["retrieval"]["collection"] = collection

            cfg = FitzRagConfig.model_validate(config_dict)

            # Initialize components
            vector_client = get_vector_db_plugin(cfg.vector_db.plugin_name, **cfg.vector_db.kwargs)
            embedder = get_llm_plugin(
                plugin_type="embedding",
                plugin_name=cfg.embedding.plugin_name,
                **cfg.embedding.kwargs,
            )

            # Determine if we have an LLM for enrichment
            has_llm = False
            chat_factory = None
            try:
                chat_factory = get_chat_factory(
                    plugin_name=cfg.chat.plugin_name,
                    **cfg.chat.kwargs,
                )
                has_llm = True
            except Exception:
                pass

            # Parser and chunking
            parser_router = ParserRouter(docling_parser="docling")

            simple_chunker = ExtensionChunkerConfig(
                plugin_name="simple",
                kwargs={"chunk_size": 1500, "chunk_overlap": 200},
            )
            router_config = ChunkingRouterConfig(
                default=simple_chunker,
                by_extension={
                    ".md": simple_chunker,
                    ".py": simple_chunker,
                    ".txt": simple_chunker,
                },
            )
            chunking_router = ChunkingRouter.from_config(router_config)

            # Enrichment config
            enrichment_cfg = EnrichmentConfig(
                artifacts=ArtifactConfig(
                    auto=False,
                    enabled=artifacts or [],
                )
            )

            # Create enrichment pipeline if we have chat
            enrichment_pipeline = None
            if has_llm and chat_factory:
                enrichment_pipeline = EnrichmentPipeline(
                    config=enrichment_cfg,
                    project_root=Path(source).resolve(),
                    chat_factory=chat_factory,
                )

            # State manager
            state_manager = IngestStateManager()
            state_manager.load()

            # Vector DB writer adapter
            class VectorDBWriterAdapter:
                def __init__(self, client):
                    self._client = client

                def upsert(self, collection: str, points: list, defer_persist: bool = False):
                    self._client.upsert(collection, points, defer_persist=defer_persist)

                def flush(self):
                    if hasattr(self._client, "flush"):
                        self._client.flush()

            writer = VectorDBWriterAdapter(vector_client)

            # Run ingestion
            summary = run_diff_ingest(
                source=source,
                state_manager=state_manager,
                vector_db_writer=writer,
                embedder=embedder,
                parser_router=parser_router,
                chunking_router=chunking_router,
                enrichment_pipeline=enrichment_pipeline,
                collection=collection,
                force_full=force_full,
            )

            duration = time.time() - start_time

            return IngestResult(
                chunks_added=summary.chunks_added,
                chunks_updated=summary.chunks_updated,
                chunks_deleted=summary.chunks_deleted,
                total_docs=summary.total_docs,
                duration_sec=duration,
                collection=collection,
            )

        else:
            raise ValueError(f"Ingestion not supported for engine: {engine_name}")

    def validate_source(self, source: str) -> tuple[bool, str]:
        """
        Validate ingestion source.

        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(source)

        if not path.exists():
            # Could be literal text
            if len(source) < 10:
                return False, f"Source path does not exist: {source}"
            # Assume literal text
            return True, ""

        if not path.is_dir() and not path.is_file():
            return False, f"Source must be a file or directory: {source}"

        return True, ""
