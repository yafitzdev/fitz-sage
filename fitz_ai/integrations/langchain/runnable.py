# fitz_ai/integrations/langchain/runnable.py
"""LangChain integration with pre-LLM caching.

This module provides FitzRAGChain, a LangChain Runnable that wraps retrieval
and LLM components with Fitz Cloud caching. The cache is checked AFTER retrieval
but BEFORE the LLM, enabling actual cost savings on repeated queries.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

try:
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.runnables import Runnable, RunnableConfig
except ImportError:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install fitz-ai[langchain]"
    )

from fitz_ai.integrations.base import FitzOptimizer
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""
)


class FitzRAGChain(Runnable):
    """
    LangChain RAG chain with Fitz Cloud caching.

    Cache is checked AFTER retrieval but BEFORE the LLM, enabling actual cost savings.
    On cache hit, the LLM is NOT called - the cached answer is returned directly.

    Example:
        >>> from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        >>> from langchain_community.vectorstores import FAISS
        >>> from fitz_ai.integrations.langchain import FitzRAGChain
        >>>
        >>> # Your existing setup
        >>> embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        >>> vectorstore = FAISS.from_documents(docs, embeddings)
        >>> retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>>
        >>> # Create cached RAG chain
        >>> chain = FitzRAGChain(
        ...     retriever=retriever,
        ...     llm=llm,
        ...     api_key="fitz_abc123...",
        ...     org_key="<64-char-hex-key>",
        ...     embedding_fn=embeddings.embed_query,
        ...     llm_model="gpt-4o",
        ... )
        >>>
        >>> # Use as normal - caching is automatic
        >>> result = chain.invoke({"question": "What is the refund policy?"})
        >>>
        >>> # On cache hit: LLM is NOT called, answer returned from cache
        >>> # On cache miss: LLM runs, result is cached for next time
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        api_key: str,
        org_key: str,
        embedding_fn: Callable[[str], list[float]],
        llm_model: str,
        prompt: Optional[ChatPromptTemplate] = None,
        org_id: Optional[str] = None,
        collection_version: str = "default",
        input_key: str = "question",
    ):
        """
        Initialize cached RAG chain.

        Args:
            retriever: Document retriever (vector store, etc.)
            llm: Language model for generation
            api_key: Fitz Cloud API key (fitz_xxx format)
            org_key: Encryption key (64-char hex, NEVER sent to server)
            embedding_fn: Function to embed query → embedding vector
            llm_model: LLM model name for cache versioning
            prompt: Optional custom prompt template
            org_id: Optional organization ID
            collection_version: Collection version for cache invalidation
            input_key: Key for question in input dict
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt or DEFAULT_PROMPT
        self.embedding_fn = embedding_fn
        self.llm_model = llm_model
        self.input_key = input_key

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
            "FitzRAGChain initialized",
            extra={"llm_model": llm_model, "collection_version": collection_version},
        )

    def invoke(
        self,
        input: dict[str, Any],
        config: Optional[RunnableConfig] = None,
    ) -> dict[str, Any]:
        """
        Invoke with caching.

        Flow:
        1. Run retrieval (cheap, local)
        2. Check cloud cache with query + chunk_ids
        3. HIT: Return cached answer (skip LLM!)
        4. MISS: Run LLM, store result, return
        """
        query = input.get(self.input_key, "")
        if not query:
            # No question - can't cache, just run
            return self._run_without_cache(input, config)

        # Step 1: Get query embedding
        try:
            query_embedding = self.embedding_fn(query)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}, falling back to uncached execution")
            return self._run_without_cache(input, config)

        # Step 2: Run retrieval (cheap, local operation)
        docs = self.retriever.invoke(query, config)
        chunk_ids = self._extract_chunk_ids(docs)

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
            return {
                "answer": cache_result.answer,
                "source_documents": docs,
                "_fitz_cache_hit": True,
                "_fitz_sources": cache_result.sources,
            }

        # Step 4b: Cache MISS - run LLM
        self._last_cache_hit = False
        self._last_routing_advice = cache_result.routing_advice
        logger.info("Cache miss - running LLM")

        # Format context and run LLM
        context = self._format_docs(docs)
        prompt_value = self.prompt.invoke({"context": context, "question": query})
        response = self.llm.invoke(prompt_value, config)
        answer_text = response.content if hasattr(response, "content") else str(response)

        # Step 5: Store in cache for future lookups
        sources = [self._doc_to_source(doc) for doc in docs]
        self.optimizer.store(
            query=query,
            query_embedding=query_embedding,
            chunk_ids=chunk_ids,
            llm_model=self.llm_model,
            answer_text=answer_text,
            sources=sources,
        )

        result: dict[str, Any] = {
            "answer": answer_text,
            "source_documents": docs,
            "_fitz_cache_hit": False,
        }

        if cache_result.routing_advice:
            result["_fitz_routing"] = cache_result.routing_advice

        return result

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: Optional[RunnableConfig] = None,
    ) -> dict[str, Any]:
        """Async invoke - delegates to sync for now."""
        # TODO: Add async cloud client support
        return self.invoke(input, config)

    def _run_without_cache(
        self,
        input: dict[str, Any],
        config: Optional[RunnableConfig],
    ) -> dict[str, Any]:
        """Fallback: run without caching."""
        query = input.get(self.input_key, "")
        docs = self.retriever.invoke(query, config)
        context = self._format_docs(docs)
        prompt_value = self.prompt.invoke({"context": context, "question": query})
        response = self.llm.invoke(prompt_value, config)
        answer_text = response.content if hasattr(response, "content") else str(response)
        return {"answer": answer_text, "source_documents": docs}

    def _format_docs(self, docs: list[Document]) -> str:
        """Format documents for prompt context."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _extract_chunk_ids(self, docs: list[Document]) -> list[str]:
        """Extract chunk IDs from documents."""
        ids = []
        for i, doc in enumerate(docs):
            doc_id = (
                doc.metadata.get("id")
                or doc.metadata.get("chunk_id")
                or doc.metadata.get("source_id")
                or doc.metadata.get("doc_id")
                or f"chunk_{i}"
            )
            ids.append(str(doc_id))
        return ids

    def _doc_to_source(self, doc: Document) -> dict:
        """Convert LangChain Document to source dict."""
        return {
            "source_id": doc.metadata.get("id", doc.metadata.get("source", "unknown")),
            "excerpt": doc.page_content[:500],
            "metadata": doc.metadata,
        }

    @property
    def was_cache_hit(self) -> bool:
        """Whether the last invocation was a cache hit."""
        return self._last_cache_hit

    def get_routing_advice(self) -> Optional[dict]:
        """Get routing advice from last cache miss."""
        return self._last_routing_advice

    def close(self):
        """Close optimizer."""
        self.optimizer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Backwards compatibility alias
FitzCacheRunnable = FitzRAGChain
