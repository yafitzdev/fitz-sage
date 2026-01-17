# fitz_ai/integrations/llamaindex/__init__.py
"""LlamaIndex integration for Fitz Cloud optimization.

Install with: pip install fitz-ai[llamaindex]

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

try:
    from fitz_ai.integrations.llamaindex.query_engine import FitzQueryEngine
except ImportError as e:
    raise ImportError(
        "LlamaIndex integration requires llama-index-core. "
        "Install with: pip install fitz-ai[llamaindex]"
    ) from e

__all__ = ["FitzQueryEngine"]
