# fitz_ai/engines/fitz_krag/engine.py
"""
FitzKragEngine - Knowledge Routing Augmented Generation engine.

Uses knowledge-type-aware access strategies (code symbols, document sections)
instead of uniform chunk-based retrieval. Retrieval returns addresses (pointers),
content is read on demand after ranking.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fitz_ai.core import (
    Answer,
    ConfigurationError,
    GenerationError,
    KnowledgeError,
    Query,
    QueryError,
)
from fitz_ai.core.answer_mode import AnswerMode
from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

# Fitz Cloud optimizer version for cache key computation
CLOUD_OPTIMIZER_VERSION = "1.0"


class FitzKragEngine:
    """
    Fitz KRAG engine implementation.

    Flow:
    1. Analyze query intent (+ optional detection)
    2. Retrieve addresses (pointers to code symbols / document sections)
    3. (Optional) Check cloud cache — early return on hit
    4. Read content for top-ranked addresses
    5. Expand with context (imports, class context, same-file refs)
    6. Run epistemic guardrails — determine AnswerMode
    7. Assemble LLM context
    8. Generate grounded answer with file:line provenance
    9. (Optional) Store in cloud cache
    """

    def __init__(self, config: FitzKragConfig):
        try:
            self._config = config
            self._init_components()
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Fitz KRAG engine: {e}") from e

    def load(self, collection: str) -> None:
        """Load a collection, reinitializing collection-dependent components."""
        self._config.collection = collection
        self._init_components()

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

        # Table stores
        from fitz_ai.engines.fitz_krag.ingestion.table_store import TableStore
        from fitz_ai.tabular.store.postgres import PostgresTableStore

        self._table_store = TableStore(self._connection_manager, self._config.collection)
        self._pg_table_store = PostgresTableStore(self._config.collection)

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
        from fitz_ai.engines.fitz_krag.retrieval.strategies.table_search import (
            TableSearchStrategy,
        )

        code_strategy = CodeSearchStrategy(self._symbol_store, self._embedder, self._config)
        section_strategy = SectionSearchStrategy(self._section_store, self._embedder, self._config)
        table_strategy = TableSearchStrategy(self._table_store, self._embedder, self._config)
        self._retrieval_router = RetrievalRouter(
            code_strategy=code_strategy,
            chunk_strategy=None,
            config=self._config,
            section_strategy=section_strategy,
            table_strategy=table_strategy,
            chat_factory=None,  # Set after chat_factory is created
        )
        self._reader = ContentReader(
            self._raw_store,
            section_store=self._section_store,
            config=self._config,
            table_store=self._table_store,
            pg_table_store=self._pg_table_store,
        )
        self._expander = CodeExpander(
            self._raw_store,
            self._symbol_store,
            self._import_store,
            self._config,
        )

        # Query Analysis
        from fitz_ai.engines.fitz_krag.query_analyzer import QueryAnalyzer

        self._query_analyzer = QueryAnalyzer(self._chat)

        # Context + Generation
        from fitz_ai.engines.fitz_krag.context.assembler import ContextAssembler
        from fitz_ai.engines.fitz_krag.generation.synthesizer import CodeSynthesizer

        self._assembler = ContextAssembler(self._config)
        self._synthesizer = CodeSynthesizer(self._chat, self._config)

        # Guardrails (epistemic constraints)
        self._constraints: list[Any] = []
        self._governor: Any = None
        if self._config.enable_guardrails:
            from fitz_ai.governance import AnswerGovernor, create_default_constraints

            self._constraints = create_default_constraints(chat=self._chat)
            self._governor = AnswerGovernor()

        # Cloud cache
        self._cloud_client: Any = None
        if self._config.cloud.get("enabled", False):
            from fitz_ai.cloud.client import CloudClient
            from fitz_ai.cloud.config import CloudConfig

            cloud_config = CloudConfig(**self._config.cloud)
            cloud_config.validate_config()
            self._cloud_client = CloudClient(cloud_config, cloud_config.org_id or "")

        # Table query handler
        from fitz_ai.engines.fitz_krag.retrieval.table_handler import TableQueryHandler

        self._table_handler = TableQueryHandler(self._chat, self._pg_table_store, self._config)

        # Chat factory (shared by detection, rewriter, HyDE, multi-hop, enrichment)
        from fitz_ai.llm.factory import get_chat_factory

        self._chat_factory = get_chat_factory(self._config.chat)

        # Shared detection
        self._detection_orchestrator: Any = None
        if self._config.enable_detection:
            from fitz_ai.retrieval.detection.registry import DetectionOrchestrator

            self._detection_orchestrator = DetectionOrchestrator(chat_factory=self._chat_factory)

        # Query rewriting
        self._query_rewriter: Any = None
        if self._config.enable_query_rewriting:
            from fitz_ai.retrieval.rewriter.rewriter import QueryRewriter

            self._query_rewriter = QueryRewriter(chat_factory=self._chat_factory)

        # HyDE generator (passed to strategies)
        self._hyde_generator: Any = None
        if self._config.enable_hyde:
            from fitz_ai.retrieval.hyde.generator import HydeGenerator

            self._hyde_generator = HydeGenerator(chat_factory=self._chat_factory)

        # Reranker (activated by config.rerank provider presence)
        self._address_reranker: Any = None
        if self._config.rerank:
            from fitz_ai.llm.client import get_reranker

            reranker = get_reranker(self._config.rerank)
            if reranker:
                from fitz_ai.engines.fitz_krag.retrieval.reranker import AddressReranker

                self._address_reranker = AddressReranker(
                    reranker=reranker,
                    k=self._config.rerank_k,
                    min_addresses=self._config.rerank_min_addresses,
                )

        # Wire chat_factory + HyDE into router and strategies
        if self._config.enable_multi_query:
            self._retrieval_router._chat_factory = self._chat_factory
        if self._hyde_generator:
            code_strategy._hyde_generator = self._hyde_generator
            section_strategy._hyde_generator = self._hyde_generator

        # Vocabulary store + keyword matcher
        self._vocabulary_store: Any = None
        self._keyword_matcher: Any = None
        if self._config.enable_enrichment:
            try:
                from fitz_ai.retrieval.vocabulary.matcher import KeywordMatcher
                from fitz_ai.retrieval.vocabulary.store import VocabularyStore

                self._vocabulary_store = VocabularyStore(collection=self._config.collection)
                keywords = self._vocabulary_store.load()
                if keywords:
                    self._keyword_matcher = KeywordMatcher(keywords)
                    self._retrieval_router._keyword_matcher = self._keyword_matcher
            except Exception as e:
                logger.debug(f"Vocabulary store init: {e}")

        # Entity graph store
        self._entity_graph_store: Any = None
        if self._config.enable_enrichment:
            try:
                from fitz_ai.retrieval.entity_graph.store import EntityGraphStore

                self._entity_graph_store = EntityGraphStore(collection=self._config.collection)
                self._expander._entity_graph_store = self._entity_graph_store
            except Exception as e:
                logger.debug(f"Entity graph store init: {e}")

        # Multi-hop controller
        self._hop_controller: Any = None
        if self._config.enable_multi_hop:
            from fitz_ai.engines.fitz_krag.retrieval.multihop import KragHopController

            self._hop_controller = KragHopController(
                router=self._retrieval_router,
                reader=self._reader,
                chat_factory=self._chat_factory,
                max_hops=self._config.max_hops,
            )

    def answer(self, query: Query) -> Answer:
        """
        Execute a query using KRAG approach.

        Flow: analyze → detect → retrieve → (cache check) → read → expand →
              guardrails → assemble → generate → (cache store)

        Args:
            query: Query object with question text

        Returns:
            Answer with file:line provenance
        """
        if not query.text or not query.text.strip():
            raise QueryError("Query text cannot be empty")

        try:
            # 0. Sanitize and normalize query
            import re

            sanitized = re.sub(r"<[^>]+>", "", query.text).strip()
            if not sanitized:
                sanitized = query.text.strip()

            # Truncate excessively long queries
            MAX_QUERY_LENGTH = 500
            if len(sanitized) > MAX_QUERY_LENGTH:
                sanitized = sanitized[:MAX_QUERY_LENGTH]
                logger.debug(f"Query truncated to {MAX_QUERY_LENGTH} chars")

            # 0.5. Rewrite query for retrieval optimization
            retrieval_query = sanitized
            if self._query_rewriter:
                try:
                    result = self._query_rewriter.rewrite(sanitized)
                    if result.rewritten_query != sanitized:
                        retrieval_query = result.rewritten_query
                        logger.debug(
                            f"Query rewritten: '{sanitized[:50]}' -> '{retrieval_query[:50]}'"
                        )
                except Exception as e:
                    logger.warning(f"Query rewriting failed, using original: {e}")

            # 1. Analyze query intent
            analysis = self._query_analyzer.analyze(retrieval_query)

            # 1.5. Run shared detection (temporal, comparison, expansion)
            detection = None
            if self._detection_orchestrator:
                detection = self._detection_orchestrator.detect_for_retrieval(retrieval_query)

            # 2. Retrieve addresses (or multi-hop)
            if self._hop_controller:
                # Multi-hop: iterative retrieve → read → evaluate → bridge
                read_results = self._hop_controller.execute(retrieval_query, analysis, detection)
                addresses = [r.address for r in read_results] if read_results else []
            else:
                addresses = self._retrieval_router.retrieve(
                    retrieval_query, analysis, detection=detection
                )

            if not addresses:
                return Answer(
                    text="No information found. The available documents do not contain relevant content for this query.",
                    provenance=[],
                    metadata={"engine": "fitz_krag", "query": query.text, "mode": "abstain"},
                )

            # 2.5. Rerank addresses (when reranker configured)
            if self._address_reranker and not self._hop_controller:
                addresses = self._address_reranker.rerank(retrieval_query, addresses)

            # 2.6. Check cloud cache (early return on hit)
            if self._cloud_client:
                cached = self._check_cloud_cache(query.text, addresses)
                if cached:
                    return cached

            # 3. Read content for top addresses (skip if multi-hop already read)
            if self._hop_controller:
                pass  # read_results already populated by hop controller
            else:
                read_results = self._reader.read(addresses, self._config.top_read)

            if not read_results:
                return Answer(
                    text="Found matching symbols but could not read their content.",
                    provenance=[],
                    metadata={"engine": "fitz_krag", "query": query.text},
                )

            # 4. Expand with context
            expanded = self._expander.expand(read_results)

            # 4.5. Execute table queries (SQL generation + execution)
            expanded = self._table_handler.process(sanitized, expanded)

            # 5. Run guardrails (ReadResult satisfies EvidenceItem protocol)
            answer_mode = AnswerMode.TRUSTWORTHY
            if self._constraints and self._governor:
                from fitz_ai.governance import run_constraints

                constraint_results = run_constraints(sanitized, expanded, self._constraints)
                governance = self._governor.decide(constraint_results)
                answer_mode = governance.mode

            # 6. Assemble context
            context = self._assembler.assemble(sanitized, expanded)

            # 7. Generate answer with answer mode
            answer = self._synthesizer.generate(
                sanitized, context, expanded, answer_mode=answer_mode
            )

            # 7.5. Store in cloud cache
            if self._cloud_client:
                self._store_cloud_cache(query.text, addresses, answer)

            return answer

        except Exception as e:
            error_msg = str(e).lower()
            if "retriev" in error_msg or "search" in error_msg:
                raise KnowledgeError(f"Retrieval failed: {e}") from e
            elif "generat" in error_msg or "llm" in error_msg:
                raise GenerationError(f"Generation failed: {e}") from e
            else:
                raise KnowledgeError(f"KRAG pipeline error: {e}") from e

    def _check_cloud_cache(self, query_text: str, addresses: list) -> Answer | None:
        """Check cloud cache for a previously cached answer."""
        from fitz_ai.cloud.cache_key import compute_retrieval_fingerprint

        try:
            query_embedding = self._embedder.embed(query_text)
            fingerprint = compute_retrieval_fingerprint([a.source_id for a in addresses])
            versions = self._build_cache_versions()

            result = self._cloud_client.lookup_cache(
                query_text=query_text,
                query_embedding=query_embedding,
                retrieval_fingerprint=fingerprint,
                versions=versions,
            )

            if result.hit:
                logger.info("Cloud cache hit for KRAG query")
                self._cached_query_embedding = query_embedding
                return result.answer

            self._cached_query_embedding = query_embedding
            return None
        except Exception as e:
            logger.warning(f"Cloud cache lookup failed: {e}")
            return None

    def _store_cloud_cache(self, query_text: str, addresses: list, answer: Answer) -> None:
        """Store an answer in the cloud cache."""
        from fitz_ai.cloud.cache_key import compute_retrieval_fingerprint

        try:
            query_embedding = getattr(self, "_cached_query_embedding", None)
            if query_embedding is None:
                query_embedding = self._embedder.embed(query_text)

            fingerprint = compute_retrieval_fingerprint([a.source_id for a in addresses])
            versions = self._build_cache_versions()

            self._cloud_client.store_cache(
                query_text=query_text,
                query_embedding=query_embedding,
                retrieval_fingerprint=fingerprint,
                versions=versions,
                answer=answer,
            )
        except Exception as e:
            logger.warning(f"Cloud cache store failed: {e}")

    def _build_cache_versions(self) -> Any:
        """Build CacheVersions for cloud cache operations."""
        import fitz_ai
        from fitz_ai.cloud.cache_key import CacheVersions

        return CacheVersions(
            optimizer=CLOUD_OPTIMIZER_VERSION,
            engine=fitz_ai.__version__,
            collection=self._config.collection,
            llm_model=self._config.chat,
            prompt_template="default",
        )

    def ingest(
        self,
        source: Path,
        collection: str | None = None,
        force: bool = False,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        """
        Ingest source files into the KRAG knowledge store.

        Args:
            source: Path to source directory or file
            collection: Collection name override (uses config default if None)
            force: If True, re-ingest all files regardless of hash state
            on_progress: Optional callback(current, total, file_path) for progress

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
            table_store=self._table_store,
            pg_table_store=self._pg_table_store,
            vocabulary_store=self._vocabulary_store,
            entity_graph_store=self._entity_graph_store,
        )
        return pipeline.ingest(source, force=force, on_progress=on_progress)

    @property
    def config(self) -> FitzKragConfig:
        """Get the engine's configuration."""
        return self._config
