# fitz_ai/integrations/langchain/__init__.py
"""LangChain integration for Fitz Cloud optimization.

Install with: pip install fitz-ai[langchain]

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

try:
    from fitz_ai.integrations.langchain.runnable import FitzRAGChain

    # Backwards compatibility alias
    FitzCacheRunnable = FitzRAGChain
except ImportError as e:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install fitz-ai[langchain]"
    ) from e

__all__ = ["FitzRAGChain", "FitzCacheRunnable"]
