# fitz_ai/engines/fitz_rag/pipeline/engine.py
"""
RAGPipeline - Core orchestration for retrieval-augmented generation.

Flow: retrieval → constraints → answer_mode → context-processing → rgs → llm → answer
"""

from __future__ import annotations

from typing import Sequence

from fitz_ai.core.answer_mode import AnswerMode
from fitz_ai.core.answer_mode_resolver import resolve_answer_mode
from fitz_ai.core.guardrails import (
    ConstraintPlugin,
    ConstraintResult,
    SemanticMatcher,
    create_default_constraints,
)
from fitz_ai.engines.fitz_rag.config import FitzRagConfig
from fitz_ai.engines.fitz_rag.exceptions import (
    LLMError,
    PipelineError,
    RGSGenerationError,
)
from fitz_ai.engines.fitz_rag.generation.answer_mode.instructions import (
    get_mode_instruction,
)
from fitz_ai.engines.fitz_rag.generation.retrieval_guided.synthesis import (
    RGS,
    RGSAnswer,
)
from fitz_ai.engines.fitz_rag.generation.retrieval_guided.synthesis import (
    RGSConfig as RGSRuntimeConfig,
)
from fitz_ai.engines.fitz_rag.pipeline.components import (
    CloudComponents,
    GuardrailComponents,
    PipelineComponents,
    RoutingComponents,
    StructuredComponents,
)
from fitz_ai.engines.fitz_rag.pipeline.pipeline import ContextPipeline
from fitz_ai.engines.fitz_rag.retrieval.multihop import (
    BridgeExtractor,
    EvidenceEvaluator,
    HopController,
)
from fitz_ai.engines.fitz_rag.retrieval.registry import get_retrieval_plugin
from fitz_ai.engines.fitz_rag.routing import QueryIntent, QueryRouter
from fitz_ai.llm.registry import get_llm_plugin
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE, VECTOR_DB
from fitz_ai.retrieval.entity_graph import EntityGraphStore
from fitz_ai.retrieval.vocabulary import create_matcher_from_store
from fitz_ai.tabular.store import get_table_store
from fitz_ai.vector_db.registry import get_vector_db_plugin

# Structured data imports (optional, fail gracefully if not available)
try:
    from fitz_ai.structured import (
        DerivedStore,
    )
    from fitz_ai.structured import QueryRouter as StructuredQueryRouter
    from fitz_ai.structured import (
        ResultFormatter,
        SchemaStore,
        SQLGenerator,
        StructuredExecutor,
        StructuredRoute,
    )

    STRUCTURED_AVAILABLE = True
except ImportError:
    STRUCTURED_AVAILABLE = False

logger = get_logger(__name__)

# Fitz Cloud optimizer version for cache key computation
CLOUD_OPTIMIZER_VERSION = "1.0"


# =============================================================================
# RAGPipeline
# =============================================================================


class RAGPipeline:
    """
    RAG Pipeline orchestrator.

    Flow: retrieval → constraints → context-processing → rgs → llm → answer
    """

    def __init__(self, components: PipelineComponents):
        """
        Initialize RAG pipeline with component groups.

        Args:
            components: Grouped pipeline dependencies
        """
        # Required core
        self.retrieval = components.retrieval
        self.chat = components.chat
        self.rgs = components.rgs
        self.context = components.context or ContextPipeline()

        # Unpack guardrail components
        guardrails = components.guardrails or GuardrailComponents()
        semantic_matcher = guardrails.semantic_matcher

        # Set up constraints with defaults
        if guardrails.constraints is None:
            if semantic_matcher is None:
                self.constraints: list[ConstraintPlugin] = []
                logger.warning(
                    f"{PIPELINE} No semantic_matcher provided, constraints disabled. "
                    "Use RAGPipeline.from_config() for full constraint support."
                )
            else:
                self.constraints = create_default_constraints(semantic_matcher)
        else:
            self.constraints = list(guardrails.constraints)

        # Unpack routing components
        routing = components.routing or RoutingComponents()
        self.query_router = routing.query_router
        self.keyword_matcher = routing.keyword_matcher
        self.hop_controller = routing.hop_controller

        # Unpack cloud components
        cloud = components.cloud or CloudComponents()
        self.embedder = cloud.embedder
        self.cloud_client = cloud.client
        self.fast_chat = cloud.fast_chat

        # Unpack structured components
        structured = components.structured or StructuredComponents()
        self.structured_router = structured.router
        self.structured_executor = structured.executor
        self.sql_generator = structured.sql_generator
        self.result_formatter = structured.result_formatter
        self.derived_store = structured.derived_store

        # Log initialization status
        routing_status = (
            "enabled" if self.query_router and self.query_router.enabled else "disabled"
        )
        keyword_status = "enabled" if self.keyword_matcher else "disabled"
        multihop_status = (
            f"enabled (max {self.hop_controller.max_hops})" if self.hop_controller else "disabled"
        )
        cloud_routing_status = "enabled" if cloud.is_enabled() else "disabled"
        structured_status = "enabled" if structured.is_enabled() else "disabled"
        logger.info(
            f"{PIPELINE} RAGPipeline initialized with {self.retrieval.plugin_name} retrieval, "
            f"{len(self.constraints)} constraint(s), routing={routing_status}, "
            f"keywords={keyword_status}, multihop={multihop_status}, "
            f"cloud_routing={cloud_routing_status}, structured={structured_status}"
        )

    def _wrap_step(self, name: str, fn, *args, error_class=None, **kwargs):
        """
        Execute pipeline step with consistent error handling.

        Args:
            name: Human-readable step name for logging
            fn: Function to execute
            *args: Positional arguments for fn
            error_class: Exception class to raise on failure (default: PipelineError)
            **kwargs: Keyword arguments for fn

        Returns:
            Result of fn(*args, **kwargs)

        Raises:
            error_class: If fn raises any exception
        """
        error_class = error_class or PipelineError
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.error(f"{PIPELINE} {name} failed: {exc}")
            raise error_class(f"{name} failed") from exc

    def run(self, query: str, conversation_context=None) -> RGSAnswer:
        """
        Execute the RAG pipeline for a query.

        Args:
            query: The user's question
            conversation_context: Optional ConversationContext for query rewriting
        """
        # Early validation: empty or whitespace-only queries
        if not query or not query.strip():
            logger.warning(f"{PIPELINE} Empty query received, returning empty answer")
            return RGSAnswer(
                answer="Please provide a question to search for.",
                sources=[],
            )

        logger.info(f"{PIPELINE} Running pipeline for query='{query[:50]}...'")

        # Step -1: Check for structured query (tables/CSV)
        # If structured, execute SQL and ingest derived sentences before retrieval
        if self.structured_router:
            self._handle_structured_prefetch(query)

        # Step 0: Route query to appropriate retrieval target
        filter_override = None
        if self.query_router:
            intent = self.query_router.classify(query)
            if intent == QueryIntent.GLOBAL:
                filter_override = self.query_router.get_l2_filter()
                logger.info(f"{PIPELINE} Query routed to L2 corpus summaries (global intent)")

        # Step 1: Retrieve relevant chunks (using multi-hop or single retrieval)
        def _do_retrieval():
            if self.hop_controller:
                raw_chunks, hop_metadata = self.hop_controller.retrieve(
                    query, filter_override=filter_override
                )
                logger.info(
                    f"{PIPELINE} Multi-hop retrieval: {hop_metadata.total_hops} hop(s), "
                    f"{len(raw_chunks)} total chunks"
                )
                return raw_chunks
            else:
                return self.retrieval.retrieve(
                    query,
                    filter_override=filter_override,
                    conversation_context=conversation_context,
                )

        raw_chunks = self._wrap_step("Retrieval", _do_retrieval)

        # Step 1.25: Check cloud cache (early return on hit)
        if self.cloud_client and self.embedder:
            cached_answer = self._check_cloud_cache(query, raw_chunks)
            if cached_answer:
                logger.info(f"{PIPELINE} Cloud cache hit, returning cached answer")
                return cached_answer

        # Step 1.5: Apply keyword filtering (if vocabulary exists)
        if self.keyword_matcher:
            filtered_chunks = self.keyword_matcher.filter_chunks(query, raw_chunks)
            if len(filtered_chunks) < len(raw_chunks):
                logger.info(
                    f"{PIPELINE} Keyword filter: {len(raw_chunks)} → {len(filtered_chunks)} chunks"
                )
            raw_chunks = filtered_chunks

        # Step 2: Apply constraints
        constraint_results = self._apply_all_constraints(query, raw_chunks)

        # Step 3: Resolve answer mode from constraint signals
        answer_mode = resolve_answer_mode(constraint_results)
        logger.info(f"{PIPELINE} Answer mode resolved: {answer_mode.value}")

        # Step 4: Process context (dedupe, group, merge, pack)
        chunks = self._wrap_step("Context processing", self.context.process, raw_chunks)

        # Step 5: Build RGS prompt with answer mode instruction
        def _build_prompt():
            prompt = self.rgs.build_prompt(query, chunks)
            return self._apply_answer_mode_to_prompt(prompt, answer_mode)

        prompt = self._wrap_step("Build RGS prompt", _build_prompt, error_class=RGSGenerationError)

        # Step 6: Generate answer via LLM
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        # Select chat model based on cloud routing advice (fail-open: default to smart tier)
        chat_client = self._select_chat_for_routing()
        raw = self._wrap_step("LLM chat", chat_client.chat, messages, error_class=LLMError)

        # Step 7: Structure the answer with mode
        def _build_answer():
            answer = self.rgs.build_answer(raw, chunks, mode=answer_mode)
            logger.info(f"{PIPELINE} Pipeline run completed (mode={answer_mode.value})")

            # Step 8: Store in cloud cache
            if self.cloud_client and self.embedder:
                self._store_in_cloud_cache(query, raw_chunks, answer)

            return answer

        return self._wrap_step("Build RGS answer", _build_answer, error_class=RGSGenerationError)

    def _apply_all_constraints(
        self,
        query: str,
        chunks,
    ) -> list[ConstraintResult]:
        """Apply all constraints and return individual results."""
        results: list[ConstraintResult] = []

        for constraint in self.constraints:
            try:
                result = constraint.apply(query, chunks)
                results.append(result)
            except Exception as e:
                logger.warning(f"{PIPELINE} Constraint '{constraint.name}' raised exception: {e}")
                # Fail-safe: continue without this constraint's result
                continue

        return results

    def _handle_structured_prefetch(self, query: str) -> None:
        """
        Handle structured data pre-fetch for table/CSV queries.

        Routes query to structured path if applicable, executes SQL via metadata
        filtering, formats results as natural language, and ingests derived
        sentences for semantic retrieval.

        This is a pre-fetch step - derived sentences are added before retrieval
        so they can be included in the semantic search results.
        """
        if not STRUCTURED_AVAILABLE:
            logger.warning(f"{PIPELINE} Structured module not available, skipping")
            return

        if not self.structured_router:
            return

        try:
            # Route the query
            route = self.structured_router.route(query)

            # Only handle structured routes
            if not isinstance(route, StructuredRoute):
                logger.debug(f"{PIPELINE} Query routed to semantic path: {route.reason}")
                return

            logger.info(
                f"{PIPELINE} Query routed to structured path: "
                f"table={route.primary_table.table_name}, confidence={route.confidence:.2f}"
            )

            # Generate SQL queries
            if not self.sql_generator:
                logger.warning(f"{PIPELINE} No SQL generator configured, skipping structured")
                return

            generation_result = self.sql_generator.generate(query, route.tables)
            if generation_result.error:
                logger.warning(f"{PIPELINE} SQL generation failed: {generation_result.error}")
                return

            if not generation_result.queries:
                logger.debug(f"{PIPELINE} No SQL queries generated")
                return

            logger.info(f"{PIPELINE} Generated {len(generation_result.queries)} SQL queries")

            # Execute queries and format results
            if not self.structured_executor or not self.result_formatter:
                logger.warning(f"{PIPELINE} Missing executor or formatter, skipping structured")
                return

            derived_sentences = []
            for sql_query in generation_result.queries:
                try:
                    # Execute SQL via metadata filtering
                    exec_result = self.structured_executor.execute(sql_query)
                    if not exec_result.is_success:
                        logger.warning(f"{PIPELINE} SQL execution failed: {exec_result.error}")
                        continue

                    # Format result as natural language
                    formatted = self.result_formatter.format(exec_result)
                    derived_sentences.append(
                        (
                            formatted.sentence,
                            sql_query.raw_sql or str(sql_query),
                            sql_query.table,
                        )
                    )

                    logger.debug(f"{PIPELINE} Formatted result: {formatted.sentence[:50]}...")

                except Exception as e:
                    logger.warning(f"{PIPELINE} SQL execution/formatting failed: {e}")
                    continue

            # Ingest derived sentences
            if derived_sentences and self.derived_store:
                for sentence, sql, table in derived_sentences:
                    # Get table version from route
                    table_schema = next((t for t in route.tables if t.table_name == table), None)
                    table_version = table_schema.version if table_schema else "unknown"

                    self.derived_store.ingest(
                        sentence=sentence,
                        source_table=table,
                        source_query=sql,
                        table_version=table_version,
                    )

                logger.info(
                    f"{PIPELINE} Ingested {len(derived_sentences)} derived sentences "
                    f"from structured query"
                )

        except Exception as e:
            # Fail-safe: log error but don't block semantic retrieval
            logger.warning(f"{PIPELINE} Structured handling failed: {e}")

    def _apply_answer_mode_to_prompt(self, prompt, answer_mode: AnswerMode):
        """Prepend answer mode instruction to system prompt."""
        from fitz_ai.engines.fitz_rag.generation.retrieval_guided.synthesis import (
            RGSPrompt,
        )

        instruction = get_mode_instruction(answer_mode)
        modified_system = f"{instruction}\n\n{prompt.system}"

        return RGSPrompt(
            system=modified_system,
            user=prompt.user,
        )

    def _select_chat_for_routing(self):
        """
        Select chat client based on cloud routing advice.

        Fail-open design:
        - No routing advice → use smart tier (self.chat)
        - No fast_chat available → use smart tier (self.chat)
        - recommended_model="fast" → use fast tier (self.fast_chat)
        - recommended_model="smart" or None → use smart tier (self.chat)
        """
        # Default: use smart tier
        if not hasattr(self, "_routing_advice") or self._routing_advice is None:
            return self.chat

        # Check if routing recommends fast tier and we have fast_chat available
        recommended = self._routing_advice.recommended_model
        if recommended == "fast" and self.fast_chat is not None:
            model_name = getattr(self.fast_chat, "params", {}).get("model", "unknown")
            logger.info(f"{PIPELINE} Using fast tier model '{model_name}' per cloud routing advice")
            return self.fast_chat

        # Default to smart tier
        return self.chat

    def _get_collection_version(self) -> str:
        """Compute collection version hash from ingestion state."""
        import hashlib

        from fitz_ai.ingestion.state import IngestStateManager

        try:
            manager = IngestStateManager()
            manager.load()

            # Get collection name from retrieval config
            collection = (
                self.retrieval.collection if hasattr(self.retrieval, "collection") else "default"
            )

            file_hashes = []
            for root_entry in manager.state.roots.values():
                for file_entry in root_entry.files.values():
                    if file_entry.is_active() and file_entry.collection == collection:
                        meta = ":".join(
                            [
                                file_entry.content_hash,
                                file_entry.chunker_id,
                                file_entry.parser_id,
                                file_entry.embedding_id,
                            ]
                        )
                        file_hashes.append(meta)

            collection_state = ":".join(sorted(file_hashes))
            return hashlib.sha256(collection_state.encode()).hexdigest()[:16]
        except Exception:
            return "unknown"

    def _get_llm_model_id(self) -> str:
        """Get LLM model identifier from chat client."""
        # Chat plugins store model in self.params dict
        try:
            if hasattr(self.chat, "params") and hasattr(self.chat, "plugin_name"):
                params = getattr(self.chat, "params", {})
                model = params.get("model")
                plugin_name = getattr(self.chat, "plugin_name", None)
                if model and plugin_name:
                    return f"{plugin_name}:{model}"
        except (AttributeError, KeyError):
            pass
        return "unknown"

    def _answer_to_rgs_answer(self, answer) -> "RGSAnswer":
        """Convert standard Answer to RGSAnswer format."""
        from fitz_ai.engines.fitz_rag.generation.retrieval_guided.synthesis import (
            RGSAnswer,
            RGSSourceRef,
        )

        sources = [
            RGSSourceRef(
                source_id=prov.source_id,
                index=idx,
                doc_id=prov.source_id,
                content=prov.excerpt or "",
                metadata=prov.metadata,
            )
            for idx, prov in enumerate(answer.provenance)
        ]

        return RGSAnswer(
            answer=answer.text,
            sources=sources,
            mode=answer.mode,
        )

    def _check_cloud_cache(self, query: str, raw_chunks: list) -> "RGSAnswer | None":
        """
        Check cloud cache for existing answer.

        Returns cached RGSAnswer if hit, None if miss.
        Also caches query embedding for reuse in _store_in_cloud_cache.
        On cache miss, stores routing advice in self._routing_advice for model selection.
        """
        # Reset routing advice (fail-open: default to None = use smart tier)
        self._routing_advice = None

        if not self.cloud_client or not self.embedder:
            return None

        import fitz_ai
        from fitz_ai.cloud.cache_key import CacheVersions, compute_retrieval_fingerprint

        try:
            # 1. Get query embedding (cache for reuse in storage)
            query_embedding = self.embedder.embed(query)
            self._cached_query_embedding = query_embedding

            # 2. Compute retrieval fingerprint from chunk IDs
            chunk_ids = [chunk.id for chunk in raw_chunks]
            retrieval_fingerprint = compute_retrieval_fingerprint(chunk_ids)

            # 3. Get collection version (lazy compute)
            if not hasattr(self, "_collection_version"):
                self._collection_version = self._get_collection_version()

            # 4. Build cache versions
            versions = CacheVersions(
                optimizer=CLOUD_OPTIMIZER_VERSION,
                engine=fitz_ai.__version__,
                collection=self._collection_version,
                llm_model=self._get_llm_model_id(),
                prompt_template="default",
            )

            # 5. Get chunk embeddings for routing advice (Pro+ tiers)
            chunk_embeddings = None
            if raw_chunks and hasattr(raw_chunks[0], "embedding"):
                chunk_embeddings = [
                    chunk.embedding for chunk in raw_chunks if hasattr(chunk, "embedding")
                ]

            # 6. Lookup cache
            result = self.cloud_client.lookup_cache(
                query_text=query,
                query_embedding=query_embedding,
                retrieval_fingerprint=retrieval_fingerprint,
                versions=versions,
                chunk_embeddings=chunk_embeddings,
            )

            if result.hit:
                # Convert Answer back to RGSAnswer
                return self._answer_to_rgs_answer(result.answer)

            # Cache miss: store routing advice for model selection
            if result.routing:
                self._routing_advice = result.routing
                if result.routing.recommended_model:
                    logger.info(
                        f"{PIPELINE} Cloud routing advice: complexity={result.routing.complexity}, "
                        f"recommended_model={result.routing.recommended_model}"
                    )

            return None
        except Exception as e:
            logger.warning(f"{PIPELINE} Cloud cache lookup failed: {e}")
            return None

    def _store_in_cloud_cache(self, query: str, raw_chunks: list, rgs_answer: "RGSAnswer") -> None:
        """Store answer in cloud cache for future lookups."""
        if not self.cloud_client or not self.embedder:
            return

        import fitz_ai
        from fitz_ai.cloud.cache_key import CacheVersions, compute_retrieval_fingerprint
        from fitz_ai.core import Answer, Provenance

        try:
            # 1. Reuse cached query embedding from _check_cloud_cache if available
            query_embedding = getattr(self, "_cached_query_embedding", None)
            if query_embedding is None:
                query_embedding = self.embedder.embed(query)

            # 2. Compute retrieval fingerprint
            chunk_ids = [chunk.id for chunk in raw_chunks]
            retrieval_fingerprint = compute_retrieval_fingerprint(chunk_ids)

            # 3. Get versions (reuse cached collection version)
            if not hasattr(self, "_collection_version"):
                self._collection_version = self._get_collection_version()

            versions = CacheVersions(
                optimizer=CLOUD_OPTIMIZER_VERSION,
                engine=fitz_ai.__version__,
                collection=self._collection_version,
                llm_model=self._get_llm_model_id(),
                prompt_template="default",
            )

            # 4. Convert RGSAnswer to Answer
            provenance = [
                Provenance(
                    source_id=src.source_id,
                    excerpt=src.content,
                    metadata=src.metadata,
                )
                for src in rgs_answer.sources
            ]

            answer = Answer(
                text=rgs_answer.answer,
                provenance=provenance,
                mode=rgs_answer.mode,
                metadata={"engine": "fitz_rag"},
            )

            # 5. Store in cache
            stored = self.cloud_client.store_cache(
                query_text=query,
                query_embedding=query_embedding,
                retrieval_fingerprint=retrieval_fingerprint,
                versions=versions,
                answer=answer,
            )

            if stored:
                logger.info(f"{PIPELINE} Answer stored in cloud cache")
        except Exception as e:
            logger.warning(f"{PIPELINE} Failed to store in cache: {e}")

    @classmethod
    def from_config(
        cls,
        cfg: FitzRagConfig,
        constraints: Sequence[ConstraintPlugin] | None = None,
        enable_keywords: bool = True,
        cloud_client=None,  # CloudClient for cloud cache operations
    ) -> "RAGPipeline":
        """
        Create a RAGPipeline from configuration.

        Args:
            cfg: RAG configuration (flat schema with string plugin names)
            constraints: Optional list of constraints to override defaults
            enable_keywords: Whether to load keyword matcher from vocabulary store
        """
        logger.info(f"{PIPELINE} Constructing RAGPipeline from config")

        # Vector DB
        vector_db_kwargs = {
            k: v for k, v in cfg.vector_db_kwargs.model_dump().items() if v is not None
        }
        vector_client = get_vector_db_plugin(cfg.vector_db, **vector_db_kwargs)
        logger.info(f"{VECTOR_DB} Using vector DB plugin='{cfg.vector_db}'")

        # Chat LLM - use "smart" tier for user-facing query responses
        chat_kwargs = {k: v for k, v in cfg.chat_kwargs.model_dump().items() if v is not None}
        chat_plugin = get_llm_plugin(
            plugin_type="chat",
            plugin_name=cfg.chat,
            tier="smart",
            **chat_kwargs,
        )
        model_name = getattr(chat_plugin, "params", {}).get("model", "unknown")
        logger.info(f"{PIPELINE} Using chat plugin='{cfg.chat}' model='{model_name}' (smart tier)")

        # Embedding
        embedding_kwargs = {
            k: v for k, v in cfg.embedding_kwargs.model_dump().items() if v is not None
        }
        embedder = get_llm_plugin(
            plugin_type="embedding",
            plugin_name=cfg.embedding,
            **embedding_kwargs,
        )
        logger.info(f"{PIPELINE} Using embedding plugin='{cfg.embedding}'")

        # Rerank (optional - None means disabled in flat schema)
        reranker = None
        if cfg.rerank:
            rerank_kwargs = {
                k: v for k, v in cfg.rerank_kwargs.model_dump().items() if v is not None
            }
            reranker = get_llm_plugin(
                plugin_type="rerank",
                plugin_name=cfg.rerank,
                **rerank_kwargs,
            )
            logger.info(f"{PIPELINE} Using rerank plugin='{cfg.rerank}'")

        # Fast chat for multi-query expansion (uses same plugin, fast tier)
        fast_chat = get_llm_plugin(
            plugin_type="chat",
            plugin_name=cfg.chat,
            tier="fast",
            **chat_kwargs,
        )
        fast_model = getattr(fast_chat, "params", {}).get("model", "unknown")
        logger.info(
            f"{PIPELINE} Using chat plugin='{cfg.chat}' model='{fast_model}' "
            "(fast tier for multi-query expansion)"
        )

        # Keyword matcher (auto-loaded from collection's vocabulary if exists)
        # Must be created before retrieval pipeline so multi-query can use it
        keyword_matcher = None
        if enable_keywords:
            keyword_matcher = create_matcher_from_store(collection=cfg.collection)
        if keyword_matcher:
            logger.info(
                f"{PIPELINE} Loaded vocabulary [{cfg.collection}] "
                f"with {len(keyword_matcher.keywords)} keywords"
            )

        # Table store for CSV file queries
        # Uses GenericTableStore for remote vector DBs, SqliteTableStore for local
        table_store = get_table_store(
            collection=cfg.collection,
            vector_db_plugin=cfg.vector_db,
            vector_plugin_instance=vector_client,
        )
        logger.info(f"{PIPELINE} Using table store for plugin='{cfg.vector_db}'")

        # Entity graph for related chunk discovery (auto-loaded from collection)
        # Defaults: enabled=True, max_expansion=5
        entity_graph = None
        max_entity_expansion = 5  # Default
        try:
            entity_graph = EntityGraphStore(collection=cfg.collection)
            if entity_graph is not None:
                graph_stats = entity_graph.stats()
                if graph_stats["entities"] > 0:
                    logger.info(
                        f"{PIPELINE} Loaded entity graph [{cfg.collection}] "
                        f"with {graph_stats['entities']} entities, {graph_stats['edges']} edges"
                    )
                else:
                    entity_graph = None  # No entities, don't use expansion
        except Exception as e:
            logger.debug(f"{PIPELINE} Entity graph not available: {e}")
            entity_graph = None

        # Retrieval (YAML-based plugin)
        retrieval = get_retrieval_plugin(
            plugin_name=cfg.retrieval_plugin,
            vector_client=vector_client,
            embedder=embedder,
            collection=cfg.collection,
            reranker=reranker,
            chat=fast_chat,
            keyword_matcher=keyword_matcher,
            entity_graph=entity_graph,
            max_entity_expansion=max_entity_expansion,
            table_store=table_store,
            top_k=cfg.top_k,
            fetch_artifacts=cfg.fetch_artifacts,
        )
        logger.info(f"{PIPELINE} Using retrieval plugin='{cfg.retrieval_plugin}'")

        # RGS - uses flattened config fields
        rgs_cfg = RGSRuntimeConfig(
            enable_citations=cfg.enable_citations,
            strict_grounding=cfg.strict_grounding,
            answer_style="default",  # Default value
            max_chunks=cfg.max_chunks,
            max_answer_chars=cfg.max_answer_chars,
            include_query_in_context=cfg.include_query_in_context,
            source_label_prefix="S",  # Default value
        )
        rgs = RGS(config=rgs_cfg)

        # Create semantic matcher for constraints using the embedder
        semantic_matcher = SemanticMatcher(embedder=embedder.embed)

        # Query router (routes global queries to L2 summaries)
        # Defaults: enabled=True, threshold=0.7
        query_router = QueryRouter(
            enabled=True,
            embedder=embedder.embed,
            threshold=0.7,
        )

        # Multi-hop controller (iterative evidence gathering)
        # Only activated when max_hops > 1. Default disabled for simpler test setup.
        hop_controller = None
        max_hops = 1  # Default (disabled). E2E tests use max_hops=3.
        if fast_chat and max_hops > 1:
            evaluator = EvidenceEvaluator(chat=fast_chat)
            extractor = BridgeExtractor(chat=fast_chat)
            hop_controller = HopController(
                retrieval_pipeline=retrieval,
                evaluator=evaluator,
                extractor=extractor,
                max_hops=max_hops,
            )
            logger.info(f"{PIPELINE} Multi-hop retrieval enabled (max {max_hops} hops)")

        # Structured data components (tables/CSV queries via SQL)
        # Default: enabled=True
        structured_router = None
        structured_executor = None
        sql_generator = None
        result_formatter = None
        derived_store = None

        if STRUCTURED_AVAILABLE:
            try:
                # Schema store for table discovery
                schema_store = SchemaStore(
                    vector_db=vector_client,
                    embedding=embedder,
                    base_collection=cfg.collection,
                )

                # Structured query router (LLM-based classification)
                # Default thresholds: schema_match=0.7, confidence=0.5
                structured_router = StructuredQueryRouter(
                    schema_store=schema_store,
                    chat_client=fast_chat,
                    schema_match_threshold=0.7,
                    structured_confidence_threshold=0.5,
                )

                # SQL generator (NL -> SQL via LLM)
                sql_generator = SQLGenerator(chat_client=fast_chat)

                # SQL executor (via metadata filtering)
                structured_executor = StructuredExecutor(
                    vector_db=vector_client,
                    base_collection=cfg.collection,
                )

                # Result formatter (SQL results -> NL sentences)
                result_formatter = ResultFormatter(chat_client=fast_chat)

                # Derived store (sentence storage with provenance)
                derived_store = DerivedStore(
                    vector_db=vector_client,
                    embedding=embedder,
                    base_collection=cfg.collection,
                )

                logger.info(
                    f"{PIPELINE} Structured data handling enabled for collection='{cfg.collection}'"
                )
            except Exception as e:
                logger.warning(f"{PIPELINE} Failed to initialize structured components: {e}")
                structured_router = None

        logger.info(f"{PIPELINE} RAGPipeline successfully created")

        # Build component groups
        components = PipelineComponents(
            retrieval=retrieval,
            chat=chat_plugin,
            rgs=rgs,
            context=ContextPipeline(),
            guardrails=GuardrailComponents(
                constraints=constraints,
                semantic_matcher=semantic_matcher,
            ),
            routing=RoutingComponents(
                query_router=query_router,
                keyword_matcher=keyword_matcher,
                hop_controller=hop_controller,
            ),
            cloud=CloudComponents(
                embedder=embedder,
                client=cloud_client,
                fast_chat=fast_chat,
            ),
            structured=StructuredComponents(
                router=structured_router,
                executor=structured_executor,
                sql_generator=sql_generator,
                result_formatter=result_formatter,
                derived_store=derived_store,
            ),
        )

        return cls(components)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RAGPipeline":
        """Create a RAGPipeline from a configuration dictionary."""
        cfg = FitzRagConfig(**config_dict)
        return cls.from_config(cfg)

    @classmethod
    def from_yaml(cls, config_path: str) -> "RAGPipeline":
        """Create a RAGPipeline from a YAML configuration file."""
        from pathlib import Path

        import yaml

        with Path(config_path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Unwrap fitz_rag key if present
        if "fitz_rag" in raw:
            config_dict = raw["fitz_rag"]
        else:
            config_dict = raw

        cfg = FitzRagConfig(**config_dict)
        return cls.from_config(cfg)


def create_pipeline_from_yaml(path: str | None = None) -> RAGPipeline:
    """Create a RAGPipeline from a YAML config file."""
    if path is None:
        from fitz_ai.config import load_engine_config

        cfg = load_engine_config("fitz_rag")
    else:
        from pathlib import Path

        import yaml

        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Unwrap fitz_rag key if present
        if "fitz_rag" in raw:
            config_dict = raw["fitz_rag"]
        else:
            config_dict = raw

        cfg = FitzRagConfig(**config_dict)

    return RAGPipeline.from_config(cfg)
