# fitz_ai/sdk/fitz.py
"""
Fitz class - Stateful SDK for the Fitz RAG framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fitz_ai.core import Answer, ConfigurationError, Provenance
from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IngestStats:
    """Statistics from an ingestion operation."""

    documents: int
    chunks: int
    collection: str


class fitz:
    """
    Stateful SDK for the Fitz RAG framework.

    Provides a simple two-step workflow:
    1. Ingest documents: fitz.ingest("./docs")
    2. Ask questions: answer = fitz.ask("question?")

    The Fitz object remembers its collection and configuration,
    allowing multiple queries without re-specifying settings.

    Examples:
        Simple usage:
        >>> f = fitz()
        >>> f.ingest("./docs")
        >>> answer = f.ask("What is the refund policy?")
        >>> print(answer.text)

        With collection name:
        >>> f = fitz(collection="physics")
        >>> f.ingest("./physics_papers")
        >>> answer = f.ask("Explain entanglement")

        With custom config:
        >>> f = fitz(config_path="my_config.yaml")

        Access provenance:
        >>> for source in answer.provenance:
        ...     print(source.excerpt)
    """

    def __init__(
        self,
        collection: str = "default",
        config_path: Optional[Union[str, Path]] = None,
        auto_init: bool = True,
    ) -> None:
        """
        Initialize the Fitz SDK.

        Args:
            collection: Name for the vector DB collection. Documents ingested
                       with this Fitz instance will be stored in this collection.
            config_path: Path to a YAML config file. If not provided, uses
                        the default config at .fitz/config.yaml or creates one.
            auto_init: If True and no config exists, create a default one.
                      If False, raise ConfigurationError when config missing.
        """
        self._collection = collection
        self._config_path = Path(config_path) if config_path else None
        self._auto_init = auto_init

        # Lazy-loaded components
        self._pipeline = None
        self._pipeline_config_hash = None

    @property
    def collection(self) -> str:
        """The vector DB collection name."""
        return self._collection

    @property
    def config_path(self) -> Path:
        """Path to the configuration file."""
        if self._config_path:
            return self._config_path
        return FitzPaths.config()

    def ingest(
        self,
        source: Union[str, Path],
        clear_existing: bool = False,
    ) -> IngestStats:
        """
        Ingest documents into the knowledge base.

        Args:
            source: Path to a file or directory to ingest.
            clear_existing: If True, clear the collection before ingesting.

        Returns:
            IngestStats with document and chunk counts.

        Raises:
            ConfigurationError: If config cannot be loaded/created.
            ValueError: If source path doesn't exist or no documents found.
        """
        import yaml

        from fitz_ai.ingestion.chunking.router import ChunkingRouter
        from fitz_ai.ingestion.diff.scanner import FileScanner
        from fitz_ai.ingestion.parser import ParserRouter
        from fitz_ai.ingestion.source.base import SourceFile
        from fitz_ai.llm.registry import get_llm_plugin
        from fitz_ai.vector_db.registry import get_vector_db_plugin
        from fitz_ai.vector_db.writer import VectorDBWriter

        source_path = Path(source)
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")

        # Ensure config exists
        self._ensure_config()

        # Load config
        with self.config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if "fitz_rag" in raw:
            config = raw["fitz_rag"]
        else:
            config = raw

        # Get plugin configs
        embedding_config = config.get("embedding", {})
        vector_db_config = config.get("vector_db", {})

        # Step 1: Discover and parse documents
        logger.info(f"Discovering documents from {source_path}")
        scanner = FileScanner()
        scan_result = scanner.scan(str(source_path))

        if not scan_result.files:
            raise ValueError(f"No documents found in {source_path}")

        parser_router = ParserRouter()
        parsed_docs = []
        for file_info in scan_result.files:
            source_file = SourceFile(
                uri=Path(file_info.path).as_uri(),
                local_path=Path(file_info.path),
                metadata={},
            )
            try:
                parsed_doc = parser_router.parse(source_file)
                if parsed_doc.full_text.strip():
                    parsed_docs.append(parsed_doc)
            except Exception as e:
                logger.warning(f"Failed to parse {file_info.path}: {e}")

        if not parsed_docs:
            raise ValueError(f"No documents could be parsed in {source_path}")

        logger.info(f"Parsed {len(parsed_docs)} documents")

        # Step 2: Chunk documents
        chunking_config = self._build_chunking_config(config)
        chunking_router = ChunkingRouter.from_config(chunking_config)

        chunks: List[Any] = []
        for parsed_doc in parsed_docs:
            ext = Path(parsed_doc.metadata.get("source_file", ".txt")).suffix or ".txt"
            chunker = chunking_router.get_chunker(ext)
            doc_chunks = chunker.chunk(parsed_doc)
            chunks.extend(doc_chunks)

        if not chunks:
            raise ValueError("No chunks created from documents")

        logger.info(f"Created {len(chunks)} chunks")

        # Step 3: Embed chunks
        logger.info("Generating embeddings...")
        embedder = get_llm_plugin(
            plugin_type="embedding",
            plugin_name=embedding_config.get("plugin_name", "cohere"),
            **embedding_config.get("kwargs", {}),
        )

        vectors = []
        for chunk in chunks:
            vec = embedder.embed(chunk.content)
            vectors.append(vec)

        # Step 4: Store in vector DB
        logger.info("Storing vectors...")
        vdb_plugin = get_vector_db_plugin(
            vector_db_config.get("plugin_name", "local_faiss"),
            **vector_db_config.get("kwargs", {}),
        )

        if clear_existing and hasattr(vdb_plugin, "delete_collection"):
            vdb_plugin.delete_collection(self._collection)

        writer = VectorDBWriter(client=vdb_plugin)
        writer.upsert(collection=self._collection, chunks=chunks, vectors=vectors)

        # Invalidate cached pipeline (collection may have changed)
        self._pipeline = None

        logger.info(
            f"Ingested {len(parsed_docs)} documents ({len(chunks)} chunks) "
            f"into collection '{self._collection}'"
        )

        return IngestStats(
            documents=len(parsed_docs),
            chunks=len(chunks),
            collection=self._collection,
        )

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        conversation_context: Optional[Any] = None,
    ) -> Answer:
        """
        Ask a question about the ingested documents.

        Args:
            question: The question to ask.
            top_k: Override the number of chunks to retrieve.
            conversation_context: Optional ConversationContext for query rewriting.
                Enables conversational pronoun resolution (e.g., "their" → "TechCorp's").

        Returns:
            Answer object with text and provenance.

        Raises:
            ConfigurationError: If not configured.
            ValueError: If question is empty.
        """
        import yaml

        from fitz_ai.engines.fitz_rag.config import FitzRagConfig
        from fitz_ai.engines.fitz_rag.pipeline.engine import RAGPipeline

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        # Ensure config exists
        self._ensure_config()

        # Load typed config
        with self.config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if "fitz_rag" in raw:
            config_dict = raw["fitz_rag"]
        else:
            config_dict = raw

        config = FitzRagConfig(**config_dict)

        # Override collection to match this Fitz instance
        config.retrieval.collection = self._collection

        # Override top_k if provided
        if top_k is not None:
            config.retrieval.top_k = top_k

        # Create pipeline (cache for efficiency)
        config_hash = hash((str(self.config_path), self._collection, top_k))
        if self._pipeline is None or self._pipeline_config_hash != config_hash:
            self._pipeline = RAGPipeline.from_config(config)
            self._pipeline_config_hash = config_hash

        # Run query
        logger.info(f"Querying: {question[:50]}...")
        rgs_answer = self._pipeline.run(question, conversation_context=conversation_context)

        # Convert RGSAnswer to core Answer
        return self._convert_to_answer(rgs_answer)

    def query(self, question: str, conversation_context: Optional[Any] = None, **kwargs) -> Answer:
        """Alias for ask(). Provided for API consistency."""
        return self.ask(question, conversation_context=conversation_context, **kwargs)

    def _ensure_config(self) -> None:
        """Ensure configuration file exists, creating if needed."""
        config_path = self.config_path

        if config_path.exists():
            return

        if not self._auto_init:
            raise ConfigurationError(
                f"Config file not found: {config_path}. Run 'fitz init' or pass auto_init=True."
            )

        # Create default config
        self._create_default_config(config_path)

    def _create_default_config(self, config_path: Path) -> None:
        """Create default configuration file."""
        config_content = f"""\
# Fitz RAG Configuration
# Generated by: Fitz SDK

chat:
  plugin_name: cohere
  kwargs:
    model: command-a-03-2025
    temperature: 0.2

embedding:
  plugin_name: cohere
  kwargs:
    model: embed-english-v3.0

vector_db:
  plugin_name: local_faiss
  kwargs: {{}}

retrieval:
  plugin_name: dense
  collection: {self._collection}
  top_k: 5

rerank:
  enabled: true
  plugin_name: cohere
  kwargs:
    model: rerank-v3.5

rgs:
  enable_citations: true
  strict_grounding: true
  max_chunks: 8

logging:
  level: INFO
"""

        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_content)
        logger.info(f"Created default config at {config_path}")

    def _build_chunking_config(self, config: Dict[str, Any]):
        """Build ChunkingRouterConfig from config dict."""
        from pathlib import Path as _Path

        import yaml

        from fitz_ai.engines.fitz_rag.config import (
            ChunkingRouterConfig,
            ExtensionChunkerConfig,
        )

        chunking_cfg = config.get("chunking") or config.get("ingest", {}).get("chunking")

        if chunking_cfg:
            default_cfg = chunking_cfg.get("default", {})
            return ChunkingRouterConfig(
                default=ExtensionChunkerConfig(
                    plugin_name=default_cfg.get("plugin_name", "recursive"),
                    kwargs=default_cfg.get("kwargs", {}),
                ),
                by_extension={
                    ext: ExtensionChunkerConfig(
                        plugin_name=ext_cfg.get("plugin_name", "simple"),
                        kwargs=ext_cfg.get("kwargs", {}),
                    )
                    for ext, ext_cfg in chunking_cfg.get("by_extension", {}).items()
                },
                warn_on_fallback=chunking_cfg.get("warn_on_fallback", False),
            )

        # Fall back to package defaults
        defaults_path = (
            _Path(__file__).parent.parent / "engines" / "fitz_rag" / "config" / "default.yaml"
        )
        with defaults_path.open("r", encoding="utf-8") as f:
            default_config = yaml.safe_load(f) or {}

        default_ingest = default_config.get("ingest", {})
        default_chunking = default_ingest.get("chunking", {}).get("default", {})

        return ChunkingRouterConfig(
            default=ExtensionChunkerConfig(
                plugin_name=default_chunking.get("plugin_name", "recursive"),
                kwargs=default_chunking.get("kwargs", {}),
            ),
        )

    def _convert_to_answer(self, rgs_answer) -> Answer:
        """Convert RGSAnswer to core Answer type."""
        provenance = []
        for source in rgs_answer.sources:
            provenance.append(
                Provenance(
                    source_id=source.source_id,
                    excerpt=source.content,
                    metadata={
                        "doc_id": source.doc_id,
                        "index": source.index,
                        **source.metadata,
                    },
                )
            )

        return Answer(
            text=rgs_answer.answer,
            provenance=provenance,
            mode=rgs_answer.mode,
            metadata={"engine": "fitz_rag"},
        )
