# fitz_ai/engines/fitz_krag/engine.py
"""
FitzKragEngine - Knowledge Routing Augmented Generation engine.

Uses knowledge-type-aware access strategies (code symbols, document sections)
instead of uniform chunk-based retrieval. Retrieval returns addresses (pointers),
content is read on demand after ranking.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fitz_ai.core import (
    Answer,
    ConfigurationError,
    GenerationError,
    KnowledgeError,
    Provenance,
    Query,
    QueryError,
)
from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_ai.logging.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class FitzKragEngine:
    """
    Fitz KRAG engine implementation.

    Flow:
    1. Retrieve addresses (pointers to code symbols / document sections)
    2. Read content for top-ranked addresses
    3. Expand with context (imports, class context, same-file refs)
    4. Assemble LLM context
    5. Generate grounded answer with file:line provenance
    """

    def __init__(self, config: FitzKragConfig):
        try:
            self._config = config
            self._init_components()
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Fitz KRAG engine: {e}") from e

    def _init_components(self) -> None:
        """Initialize engine components lazily."""
        from fitz_ai.llm.client import get_chat, get_embedder
        from fitz_ai.storage.postgres import PostgresConnectionManager

        self._chat = get_chat(
            self._config.chat,
            config=self._config.chat_kwargs.model_dump(exclude_none=True) or None,
        )
        self._embedder = get_embedder(
            self._config.embedding,
            config=self._config.embedding_kwargs.model_dump(exclude_none=True) or None,
        )
        self._connection_manager = PostgresConnectionManager.get_instance()

        # Ingestion stores
        from fitz_ai.engines.fitz_krag.ingestion.import_graph_store import ImportGraphStore
        from fitz_ai.engines.fitz_krag.ingestion.raw_file_store import RawFileStore
        from fitz_ai.engines.fitz_krag.ingestion.schema import ensure_schema
        from fitz_ai.engines.fitz_krag.ingestion.section_store import SectionStore
        from fitz_ai.engines.fitz_krag.ingestion.symbol_store import SymbolStore

        self._raw_store = RawFileStore(self._connection_manager, self._config.collection)
        self._symbol_store = SymbolStore(self._connection_manager, self._config.collection)
        self._import_store = ImportGraphStore(self._connection_manager, self._config.collection)
        self._section_store = SectionStore(self._connection_manager, self._config.collection)

        # Ensure schema exists
        embedding_dim = self._embedder.dimensions
        ensure_schema(self._connection_manager, self._config.collection, embedding_dim)

        # Retrieval
        from fitz_ai.engines.fitz_krag.retrieval.expander import CodeExpander
        from fitz_ai.engines.fitz_krag.retrieval.reader import ContentReader
        from fitz_ai.engines.fitz_krag.retrieval.router import RetrievalRouter
        from fitz_ai.engines.fitz_krag.retrieval.strategies.code_search import (
            CodeSearchStrategy,
        )
        from fitz_ai.engines.fitz_krag.retrieval.strategies.section_search import (
            SectionSearchStrategy,
        )

        code_strategy = CodeSearchStrategy(self._symbol_store, self._embedder, self._config)
        section_strategy = SectionSearchStrategy(self._section_store, self._embedder, self._config)
        self._retrieval_router = RetrievalRouter(
            code_strategy=code_strategy,
            chunk_strategy=None,
            config=self._config,
            section_strategy=section_strategy,
        )
        self._reader = ContentReader(self._raw_store, section_store=self._section_store)
        self._expander = CodeExpander(
            self._raw_store,
            self._symbol_store,
            self._import_store,
            self._config,
        )

        # Context + Generation
        from fitz_ai.engines.fitz_krag.context.assembler import ContextAssembler
        from fitz_ai.engines.fitz_krag.generation.synthesizer import CodeSynthesizer

        self._assembler = ContextAssembler(self._config)
        self._synthesizer = CodeSynthesizer(self._chat, self._config)

    def answer(self, query: Query) -> Answer:
        """
        Execute a query using KRAG approach.

        Flow: retrieve addresses -> read content -> expand -> assemble -> generate

        Args:
            query: Query object with question text

        Returns:
            Answer with file:line provenance
        """
        if not query.text or not query.text.strip():
            raise QueryError("Query text cannot be empty")

        try:
            # 1. Retrieve addresses
            addresses = self._retrieval_router.retrieve(query.text)

            if not addresses:
                return Answer(
                    text="No relevant code or documents found for this query.",
                    provenance=[],
                    metadata={"engine": "fitz_krag", "query": query.text},
                )

            # 2. Read content for top addresses
            read_results = self._reader.read(addresses, self._config.top_read)

            if not read_results:
                return Answer(
                    text="Found matching symbols but could not read their content.",
                    provenance=[],
                    metadata={"engine": "fitz_krag", "query": query.text},
                )

            # 3. Expand with context
            expanded = self._expander.expand(read_results)

            # 4. Assemble context
            context = self._assembler.assemble(query.text, expanded)

            # 5. Generate answer
            return self._synthesizer.generate(query.text, context, expanded)

        except Exception as e:
            error_msg = str(e).lower()
            if "retriev" in error_msg or "search" in error_msg:
                raise KnowledgeError(f"Retrieval failed: {e}") from e
            elif "generat" in error_msg or "llm" in error_msg:
                raise GenerationError(f"Generation failed: {e}") from e
            else:
                raise KnowledgeError(f"KRAG pipeline error: {e}") from e

    def ingest(self, source: Path, collection: str | None = None) -> dict:
        """
        Ingest source files into the KRAG knowledge store.

        Args:
            source: Path to source directory or file
            collection: Collection name override (uses config default if None)

        Returns:
            Stats dict with files, symbols, imports counts
        """
        from fitz_ai.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline

        col = collection or self._config.collection
        pipeline = KragIngestPipeline(
            config=self._config,
            chat=self._chat,
            embedder=self._embedder,
            connection_manager=self._connection_manager,
            collection=col,
        )
        return pipeline.ingest(source)

    @property
    def config(self) -> FitzKragConfig:
        """Get the engine's configuration."""
        return self._config
