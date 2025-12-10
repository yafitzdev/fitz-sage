"""
RAGPipeline for fitz_rag.

This class orchestrates:
  - dense retrieval
  - reranking
  - context building
  - prompt construction
  - LLM chat

Usage:
    pipeline = RAGPipeline(
        retriever=my_retriever,
        reranker=my_reranker,
        chat_client=my_chat_client,
    )

    answer = pipeline.run("What is this about?")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from fitz_rag.retriever.base import BaseRetriever
from fitz_rag.llm.rerank_client import RerankClient
from fitz_rag.llm.chat_client import ChatClient
from fitz_rag.context.builder import build_context


@dataclass
class RAGPipeline:
    retriever: BaseRetriever
    reranker: RerankClient
    chat_client: ChatClient

    system_prompt: str = "You are a helpful assistant."
    context_chars: int = 4000
    final_top_k: int = 5

    # ---------------------------------------------------------
    # Step 1: Dense retrieval
    # ---------------------------------------------------------
    def _retrieve(self, query: str) -> List[Dict[str, Any]]:
        results = self.retriever.retrieve(query)

        chunks = []
        for r in results:
            chunks.append(
                {
                    "text": getattr(r, "text", ""),  # direct field
                    "file": r.metadata.get("file", "unknown"),  # from metadata
                }
            )
        return chunks

    # ---------------------------------------------------------
    # Step 2: Rerank retrieved chunks
    # ---------------------------------------------------------
    def _rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        texts = [c["text"] for c in chunks]
        indices = self.reranker.rerank(query, texts, top_n=self.final_top_k)
        return [chunks[i] for i in indices]

    # ---------------------------------------------------------
    # Step 3: Build context (merged chunks)
    # ---------------------------------------------------------
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        return build_context(chunks, max_chars=self.context_chars)

    # ---------------------------------------------------------
    # Step 4: Prompt builder
    # ---------------------------------------------------------
    def _build_prompt(self, context: str, query: str) -> str:
        return f"""
CONTEXT:
{context}

QUESTION:
{query}

Answer ONLY using the context above. If unsure, say 'I don't know'.
""".strip()

    # ---------------------------------------------------------
    # Step 5: Chat with LLM
    # ---------------------------------------------------------
    def _chat(self, prompt: str) -> str:
        return self.chat_client.chat(
            system_prompt=self.system_prompt,
            user_content=prompt,
        )

    # ---------------------------------------------------------
    # PUBLIC ENTRYPOINT
    # ---------------------------------------------------------
    def run(self, query: str) -> str:
        chunks = self._retrieve(query)
        if not chunks:
            return "No relevant information found."

        reranked = self._rerank(query, chunks)
        context = self._build_context(reranked)
        prompt = self._build_prompt(context, query)

        return self._chat(prompt)
