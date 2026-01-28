# fitz_ai/integrations/llamaindex/query_engine.py
"""LlamaIndex integration with pre-LLM caching.

This module provides FitzQueryEngine, a LlamaIndex query engine that wraps
retrieval and LLM components with Fitz Cloud caching. The cache is checked
AFTER retrieval but BEFORE the LLM, enabling actual cost savings on repeated queries.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

try:
    from llama_index.core.base.base_query_engine import BaseQueryEngine
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.base.response.schema import Response
    from llama_index.core.llms import LLM
    from llama_index.core.schema import NodeWithScore, QueryBundle
except ImportError:
    raise ImportError(
        "LlamaIndex integration requires llama-index-core. "
        "Install with: pip install fitz-ai[llamaindex]"
    )

from fitz_ai.integrations.base import FitzOptimizer
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PROMPT_TEMPLATE = """Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""


class FitzQueryEngine(BaseQueryEngine):
    """
    LlamaIndex query engine with Fitz Cloud caching.

    Cache is checked AFTER retrieval but BEFORE synthesis, enabling actual cost savings.
    On cache hit, the LLM is NOT called - the cached answer is returned directly.

    Example:
        >>> from llama_index.core import VectorStoreIndex
        >>> from llama_index.embeddings.openai import OpenAIEmbedding
        >>> from llama_index.llms.openai import OpenAI
        >>> from fitz_ai.integrations.llamaindex import FitzQueryEngine
        >>>
        >>> # Your existing setup
        >>> embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        >>> index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        >>> retriever = index.as_retriever(similarity_top_k=5)
        >>> llm = OpenAI(model="gpt-4o")
        >>>
        >>> # Create cached query engine
        >>> engine = FitzQueryEngine(
        ...     retriever=retriever,
        ...     llm=llm,
        ...     api_key="fitz_abc123...",
        ...     org_key="<64-char-hex-key>",
        ...     embedding_fn=embed_model.get_query_embedding,
        ...     llm_model="gpt-4o",
        ... )
        >>>
        >>> # Use as normal - caching is automatic
        >>> response = engine.query("What is the refund policy?")
        >>>
        >>> # On cache hit: LLM is NOT called, answer returned from cache
        >>> # On cache miss: LLM runs, result is cached for next time
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        api_key: str,
        org_key: str,
        embedding_fn: Callable[[str], list[float]],
        llm_model: str,
        prompt_template: Optional[str] = None,
        org_id: Optional[str] = None,
        collection_version: str = "default",
        # Legacy parameter for backwards compatibility
        query_engine: Optional[BaseQueryEngine] = None,
    ):
        """
        Initialize cached query engine.

        Args:
            retriever: Document retriever (from index.as_retriever())
            llm: Language model for synthesis
            api_key: Fitz Cloud API key (fitz_xxx format)
            org_key: Encryption key (64-char hex, NEVER sent to server)
            embedding_fn: Function to embed query → embedding vector
            llm_model: LLM model name for cache versioning
            prompt_template: Optional custom prompt template (must have {context} and {question})
            org_id: Optional organization ID
            collection_version: Collection version for cache invalidation
            query_engine: DEPRECATED - use retriever + llm instead
        """
        super().__init__(callback_manager=None)

        # Handle legacy API
        if query_engine is not None:
            logger.warning(
                "query_engine parameter is deprecated. Use retriever + llm for proper caching."
            )
            # Fall back to legacy mode if query_engine provided
            self._legacy_mode = True
            self._query_engine = query_engine
            self.retriever = None
            self.llm = None
        else:
            self._legacy_mode = False
            self._query_engine = None
            self.retriever = retriever
            self.llm = llm

        self.embedding_fn = embedding_fn
        self.llm_model = llm_model
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE

        # Track metrics
        self._last_cache_hit = False
        self._last_routing_advice: Optional[dict] = None

        self.optimizer = FitzOptimizer(
            api_key=api_key,
            org_key=org_key,
            org_id=org_id,
            collection_version=collection_version,
        )

        logger.info(
            "FitzQueryEngine initialized",
            extra={"llm_model": llm_model, "collection_version": collection_version},
        )

    def _query(self, query_bundle: QueryBundle) -> Response:
        """
        Query with caching.

        Flow:
        1. Run retrieval (cheap, local)
        2. Check cloud cache with query + chunk_ids
        3. HIT: Return cached answer (skip LLM!)
        4. MISS: Run LLM, store result, return
        """
        query = query_bundle.query_str

        # Legacy mode: can't short-circuit, just run and cache
        if self._legacy_mode:
            return self._query_legacy(query_bundle)

        # Step 1: Get query embedding
        try:
            query_embedding = self.embedding_fn(query)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}, falling back to uncached execution")
            return self._run_without_cache(query_bundle)

        # Step 2: Run retrieval (cheap, local operation)
        nodes = self.retriever.retrieve(query_bundle)
        chunk_ids = self._extract_chunk_ids(nodes)

        # Step 3: Check cloud cache
        cache_result = self.optimizer.lookup(
            query=query,
            query_embedding=query_embedding,
            chunk_ids=chunk_ids,
            llm_model=self.llm_model,
        )

        # Step 4a: Cache HIT - return cached answer, skip LLM!
        if cache_result.hit:
            self._last_cache_hit = True
            self._last_routing_advice = None
            logger.info("Cache hit - skipping LLM call")
            return Response(
                response=cache_result.answer,
                source_nodes=nodes,
                metadata={
                    "_fitz_cache_hit": True,
                    "_fitz_sources": cache_result.sources,
                },
            )

        # Step 4b: Cache MISS - run LLM
        self._last_cache_hit = False
        self._last_routing_advice = cache_result.routing_advice
        logger.info("Cache miss - running LLM")

        # Format context and run LLM
        context = self._format_nodes(nodes)
        prompt = self.prompt_template.format(context=context, question=query)
        response_obj = self.llm.complete(prompt)
        response_text = str(response_obj)

        # Step 5: Store in cache for future lookups
        sources = [self._node_to_source(node) for node in nodes]
        self.optimizer.store(
            query=query,
            query_embedding=query_embedding,
            chunk_ids=chunk_ids,
            llm_model=self.llm_model,
            answer_text=response_text,
            sources=sources,
        )

        metadata: dict[str, Any] = {"_fitz_cache_hit": False}
        if cache_result.routing_advice:
            metadata["_fitz_routing"] = cache_result.routing_advice

        return Response(
            response=response_text,
            source_nodes=nodes,
            metadata=metadata,
        )

    def _query_legacy(self, query_bundle: QueryBundle) -> Response:
        """Legacy mode: query engine wrapping (no LLM short-circuit)."""
        query = query_bundle.query_str

        try:
            query_embedding = self.embedding_fn(query)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return self._query_engine.query(query_bundle)

        # Run the query engine (LLM runs here - can't short-circuit)
        response = self._query_engine.query(query_bundle)

        # Extract chunk IDs and check cache
        source_nodes = getattr(response, "source_nodes", [])
        chunk_ids = self._extract_chunk_ids(source_nodes)

        cache_result = self.optimizer.lookup(
            query=query,
            query_embedding=query_embedding,
            chunk_ids=chunk_ids,
            llm_model=self.llm_model,
        )

        self._last_routing_advice = cache_result.routing_advice

        # Store in cache if miss
        response_text = str(response) if response else ""
        if not cache_result.hit and response_text:
            sources = [self._node_to_source(node) for node in source_nodes]
            self.optimizer.store(
                query=query,
                query_embedding=query_embedding,
                chunk_ids=chunk_ids,
                llm_model=self.llm_model,
                answer_text=response_text,
                sources=sources,
            )

        # Add routing metadata
        if cache_result.routing_advice and hasattr(response, "metadata"):
            if response.metadata is None:
                response.metadata = {}
            response.metadata["_fitz_routing"] = cache_result.routing_advice

        return response

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        """Async query - delegates to sync for now."""
        # TODO: Add async cloud client support
        return self._query(query_bundle)

    def _run_without_cache(self, query_bundle: QueryBundle) -> Response:
        """Fallback: run without caching."""
        nodes = self.retriever.retrieve(query_bundle)
        context = self._format_nodes(nodes)
        prompt = self.prompt_template.format(context=context, question=query_bundle.query_str)
        response_obj = self.llm.complete(prompt)
        response_text = str(response_obj)
        return Response(response=response_text, source_nodes=nodes)

    def _format_nodes(self, nodes: list[NodeWithScore]) -> str:
        """Format nodes for prompt context."""
        return "\n\n".join(node.node.get_content() for node in nodes)

    def _extract_chunk_ids(self, nodes: list[NodeWithScore]) -> list[str]:
        """Extract chunk IDs from nodes."""
        ids = []
        for i, node in enumerate(nodes):
            node_id = node.node.node_id or f"node_{i}"
            ids.append(str(node_id))
        return ids

    def _node_to_source(self, node: NodeWithScore) -> dict:
        """Convert LlamaIndex node to source dict."""
        return {
            "source_id": node.node.node_id or "unknown",
            "excerpt": node.node.get_content()[:500],
            "metadata": node.node.metadata,
        }

    @property
    def was_cache_hit(self) -> bool:
        """Whether the last query was a cache hit."""
        return self._last_cache_hit

    def get_routing_advice(self) -> Optional[dict]:
        """Get routing advice from last cache miss."""
        return self._last_routing_advice

    def close(self):
        """Close optimizer."""
        self.optimizer.close()

    def _get_prompt_modules(self) -> dict:
        """Return prompt modules (required by BaseQueryEngine)."""
        return {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
