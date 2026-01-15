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
from fitz_ai.engines.fitz_rag.config import FitzRagConfig, load_config
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
from fitz_ai.retrieval.vocabulary import KeywordMatcher, create_matcher_from_store
from fitz_ai.tabular.store import get_table_store
from fitz_ai.vector_db.registry import get_vector_db_plugin

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

    def __init__(
        self,
        retrieval,  # RetrievalPipelineFromYaml
        chat,
        rgs: RGS,
        context: ContextPipeline | None = None,
        constraints: Sequence[ConstraintPlugin] | None = None,
        semantic_matcher: SemanticMatcher | None = None,
        query_router: QueryRouter | None = None,
        keyword_matcher: KeywordMatcher | None = None,
        hop_controller: HopController | None = None,
        embedder=None,  # Embedder for cloud cache query embeddings
        cloud_client=None,  # CloudClient for cloud cache operations
    ):
        self.retrieval = retrieval
        self.chat = chat
        self.rgs = rgs
        self.context = context or ContextPipeline()
        self.query_router = query_router
        self.keyword_matcher = keyword_matcher
        self.hop_controller = hop_controller
        self.embedder = embedder
        self.cloud_client = cloud_client

        # Default constraints: ConflictAware + InsufficientEvidence + CausalAttribution
        # Uses semantic embedding similarity for language-agnostic detection.
        # Users can override by passing constraints=[] to disable
        if constraints is None:
            if semantic_matcher is None:
                # No constraints if no semantic matcher provided
                # (caller should use from_config which provides embedder)
                self.constraints: list[ConstraintPlugin] = []
                logger.warning(
                    f"{PIPELINE} No semantic_matcher provided, constraints disabled. "
                    "Use RAGPipeline.from_config() for full constraint support."
                )
            else:
                self.constraints = create_default_constraints(semantic_matcher)
        else:
            self.constraints = list(constraints)

        routing_status = "enabled" if query_router and query_router.enabled else "disabled"
        keyword_status = "enabled" if keyword_matcher else "disabled"
        multihop_status = (
            f"enabled (max {hop_controller.max_hops})" if hop_controller else "disabled"
        )
        logger.info(
            f"{PIPELINE} RAGPipeline initialized with {retrieval.plugin_name} retrieval, "
            f"{len(self.constraints)} constraint(s), routing={routing_status}, "
            f"keywords={keyword_status}, multihop={multihop_status}"
        )

    def run(self, query: str) -> RGSAnswer:
        """Execute the RAG pipeline for a query."""
        logger.info(f"{PIPELINE} Running pipeline for query='{query[:50]}...'")

        # Step 0: Route query to appropriate retrieval target
        filter_override = None
        if self.query_router:
            intent = self.query_router.classify(query)
            if intent == QueryIntent.GLOBAL:
                filter_override = self.query_router.get_l2_filter()
                logger.info(f"{PIPELINE} Query routed to L2 corpus summaries (global intent)")

        # Step 1: Retrieve relevant chunks (using multi-hop or single retrieval)
        try:
            if self.hop_controller:
                raw_chunks, hop_metadata = self.hop_controller.retrieve(
                    query, filter_override=filter_override
                )
                logger.info(
                    f"{PIPELINE} Multi-hop retrieval: {hop_metadata.total_hops} hop(s), "
                    f"{len(raw_chunks)} total chunks"
                )
            else:
                raw_chunks = self.retrieval.retrieve(query, filter_override=filter_override)
        except Exception as exc:
            logger.error(f"{PIPELINE} Retrieval failed: {exc}")
            raise PipelineError("Retrieval failed") from exc

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
        try:
            chunks = self.context.process(raw_chunks)
        except Exception as exc:
            logger.error(f"{PIPELINE} Context processing failed: {exc}")
            raise PipelineError("Context processing failed") from exc

        # Step 5: Build RGS prompt with answer mode instruction
        try:
            prompt = self.rgs.build_prompt(query, chunks)
            prompt = self._apply_answer_mode_to_prompt(prompt, answer_mode)
        except Exception as exc:
            logger.error(f"{PIPELINE} Failed to build RGS prompt: {exc}")
            raise RGSGenerationError("Failed to build RGS prompt") from exc

        # Step 6: Generate answer via LLM
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        try:
            raw = self.chat.chat(messages)
        except Exception as exc:
            logger.error(f"{PIPELINE} LLM chat failed: {exc}")
            raise LLMError("LLM chat operation failed") from exc

        # Step 7: Structure the answer with mode
        try:
            answer = self.rgs.build_answer(raw, chunks, mode=answer_mode)
            logger.info(f"{PIPELINE} Pipeline run completed (mode={answer_mode.value})")

            # Step 8: Store in cloud cache
            if self.cloud_client and self.embedder:
                self._store_in_cloud_cache(query, raw_chunks, answer)

            return answer
        except Exception as exc:
            logger.error(f"{PIPELINE} Failed to structure RGS answer: {exc}")
            raise RGSGenerationError("Failed to build RGS answer") from exc

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

    def _get_collection_version(self) -> str:
        """Compute collection version hash from ingestion state."""
        from fitz_ai.ingestion.state import IngestStateManager
        import hashlib

        try:
            manager = IngestStateManager()
            manager.load()

            # Get collection name from retrieval config
            collection = (
                self.retrieval.collection
                if hasattr(self.retrieval, "collection")
                else "default"
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
        """
        if not self.cloud_client or not self.embedder:
            return None

        from fitz_ai.cloud.cache_key import CacheVersions, compute_retrieval_fingerprint
        import fitz_ai

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

            # 5. Lookup cache
            result = self.cloud_client.lookup_cache(
                query_text=query,
                query_embedding=query_embedding,
                retrieval_fingerprint=retrieval_fingerprint,
                versions=versions,
            )

            if result.hit:
                # Convert Answer back to RGSAnswer
                return self._answer_to_rgs_answer(result.answer)

            return None
        except Exception as e:
            logger.warning(f"{PIPELINE} Cloud cache lookup failed: {e}")
            return None

    def _store_in_cloud_cache(self, query: str, raw_chunks: list, rgs_answer: "RGSAnswer") -> None:
        """Store answer in cloud cache for future lookups."""
        if not self.cloud_client or not self.embedder:
            return

        from fitz_ai.cloud.cache_key import CacheVersions, compute_retrieval_fingerprint
        from fitz_ai.core import Answer, Provenance
        import fitz_ai

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
            cfg: RAG configuration
            constraints: Optional list of constraints to override defaults
            enable_keywords: Whether to load keyword matcher from vocabulary store
        """
        logger.info(f"{PIPELINE} Constructing RAGPipeline from config")

        # Vector DB
        vector_client = get_vector_db_plugin(cfg.vector_db.plugin_name, **cfg.vector_db.kwargs)
        logger.info(f"{VECTOR_DB} Using vector DB plugin='{cfg.vector_db.plugin_name}'")

        # Chat LLM - use "smart" tier for user-facing query responses
        chat_plugin = get_llm_plugin(
            plugin_type="chat",
            plugin_name=cfg.chat.plugin_name,
            tier="smart",
            **cfg.chat.kwargs,
        )
        model_name = getattr(chat_plugin, "params", {}).get("model", "unknown")
        logger.info(
            f"{PIPELINE} Using chat plugin='{cfg.chat.plugin_name}' model='{model_name}' (smart tier)"
        )

        # Embedding
        embedder = get_llm_plugin(
            plugin_type="embedding",
            plugin_name=cfg.embedding.plugin_name,
            **cfg.embedding.kwargs,
        )
        logger.info(f"{PIPELINE} Using embedding plugin='{cfg.embedding.plugin_name}'")

        # Rerank (optional)
        reranker = None
        if cfg.rerank.enabled and cfg.rerank.plugin_name:
            reranker = get_llm_plugin(
                plugin_type="rerank",
                plugin_name=cfg.rerank.plugin_name,
                **cfg.rerank.kwargs,
            )
            logger.info(f"{PIPELINE} Using rerank plugin='{cfg.rerank.plugin_name}'")

        # Fast chat for multi-query expansion (uses same plugin, fast tier)
        fast_chat = get_llm_plugin(
            plugin_type="chat",
            plugin_name=cfg.chat.plugin_name,
            tier="fast",
            **cfg.chat.kwargs,
        )
        fast_model = getattr(fast_chat, "params", {}).get("model", "unknown")
        logger.info(
            f"{PIPELINE} Using chat plugin='{cfg.chat.plugin_name}' model='{fast_model}' "
            "(fast tier for multi-query expansion)"
        )

        # Keyword matcher (auto-loaded from collection's vocabulary if exists)
        # Must be created before retrieval pipeline so multi-query can use it
        keyword_matcher = None
        if enable_keywords:
            keyword_matcher = create_matcher_from_store(collection=cfg.retrieval.collection)
        if keyword_matcher:
            logger.info(
                f"{PIPELINE} Loaded vocabulary [{cfg.retrieval.collection}] "
                f"with {len(keyword_matcher.keywords)} keywords"
            )

        # Table store for CSV file queries
        # Uses GenericTableStore for remote vector DBs, SqliteTableStore for local
        table_store = get_table_store(
            collection=cfg.retrieval.collection,
            vector_db_plugin=cfg.vector_db.plugin_name,
            vector_plugin_instance=vector_client,
        )
        logger.info(f"{PIPELINE} Using table store for plugin='{cfg.vector_db.plugin_name}'")

        # Entity graph for related chunk discovery (auto-loaded from collection)
        entity_graph = None
        max_entity_expansion = cfg.entity_graph.max_expansion
        if cfg.entity_graph.enabled:
            entity_graph = EntityGraphStore(collection=cfg.retrieval.collection)
            graph_stats = entity_graph.stats()
            if graph_stats["entities"] > 0:
                logger.info(
                    f"{PIPELINE} Loaded entity graph [{cfg.retrieval.collection}] "
                    f"with {graph_stats['entities']} entities, {graph_stats['edges']} edges"
                )
            else:
                entity_graph = None  # No entities, don't use expansion
        else:
            logger.info(f"{PIPELINE} Entity graph disabled by config")

        # Retrieval (YAML-based plugin)
        retrieval = get_retrieval_plugin(
            plugin_name=cfg.retrieval.plugin_name,
            vector_client=vector_client,
            embedder=embedder,
            collection=cfg.retrieval.collection,
            reranker=reranker,
            chat=fast_chat,
            keyword_matcher=keyword_matcher,
            entity_graph=entity_graph,
            max_entity_expansion=max_entity_expansion,
            table_store=table_store,
            top_k=cfg.retrieval.top_k,
            fetch_artifacts=cfg.retrieval.fetch_artifacts,
        )
        logger.info(f"{PIPELINE} Using retrieval plugin='{cfg.retrieval.plugin_name}'")

        # RGS
        rgs_cfg = RGSRuntimeConfig(
            enable_citations=cfg.rgs.enable_citations,
            strict_grounding=cfg.rgs.strict_grounding,
            answer_style=cfg.rgs.answer_style,
            max_chunks=cfg.rgs.max_chunks,
            max_answer_chars=cfg.rgs.max_answer_chars,
            include_query_in_context=cfg.rgs.include_query_in_context,
            source_label_prefix=cfg.rgs.source_label_prefix,
        )
        rgs = RGS(config=rgs_cfg)

        # Create semantic matcher for constraints using the embedder
        semantic_matcher = SemanticMatcher(embedder=embedder.embed)

        # Query router (routes global queries to L2 summaries)
        query_router = QueryRouter(
            enabled=cfg.routing.enabled,
            embedder=embedder.embed,
            threshold=cfg.routing.threshold,
        )

        # Multi-hop controller (iterative evidence gathering)
        # Always on when fast_chat is available, but naturally stops after 1 hop
        # when evidence is sufficient
        hop_controller = None
        if fast_chat and cfg.multihop.max_hops > 1:
            evaluator = EvidenceEvaluator(chat=fast_chat)
            extractor = BridgeExtractor(chat=fast_chat)
            hop_controller = HopController(
                retrieval_pipeline=retrieval,
                evaluator=evaluator,
                extractor=extractor,
                max_hops=cfg.multihop.max_hops,
            )
            logger.info(
                f"{PIPELINE} Multi-hop retrieval enabled (max {cfg.multihop.max_hops} hops)"
            )

        logger.info(f"{PIPELINE} RAGPipeline successfully created")
        return cls(
            retrieval=retrieval,
            chat=chat_plugin,
            rgs=rgs,
            context=ContextPipeline(),
            constraints=constraints,
            semantic_matcher=semantic_matcher,
            query_router=query_router,
            keyword_matcher=keyword_matcher,
            hop_controller=hop_controller,
            embedder=embedder,
            cloud_client=cloud_client,
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RAGPipeline":
        """Create a RAGPipeline from a configuration dictionary."""
        cfg = FitzRagConfig.from_dict(config_dict)
        return cls.from_config(cfg)

    @classmethod
    def from_yaml(cls, config_path: str) -> "RAGPipeline":
        """Create a RAGPipeline from a YAML configuration file."""
        cfg = load_config(config_path)
        return cls.from_config(cfg)


def create_pipeline_from_yaml(path: str | None = None) -> RAGPipeline:
    """Create a RAGPipeline from a YAML config file."""
    cfg = load_config(path)
    return RAGPipeline.from_config(cfg)
