# fitz_ai/code/__init__.py
"""
Standalone code retrieval — no PostgreSQL, no pgvector, no docling.

Usage:
    from fitz_ai.code import CodeRetriever
    from fitz_ai.llm.factory import get_chat_factory

    retriever = CodeRetriever(
        source_dir="./myproject",
        chat_factory=get_chat_factory({"fast": "ollama/qwen2.5:3b", "smart": "ollama/qwen2.5:7b"}),
    )
    results = retriever.retrieve("How does authentication work?")
"""

from fitz_ai.code.retriever import CodeRetriever

__all__ = ["CodeRetriever"]
