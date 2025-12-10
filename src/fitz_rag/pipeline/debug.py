from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

from qdrant_client import QdrantClient

from fitz_rag.pipeline.engine import RAGPipeline
from fitz_rag.retriever.dense_retriever import RAGRetriever
from fitz_rag.generation.rgs import RGS, RGSConfig
from fitz_rag.llm.embedding_client import CohereEmbeddingClient
from fitz_rag.llm.rerank_client import CohereRerankClient
from fitz_rag.llm.chat_client import CohereChatClient


@dataclass
class DebugRAG:
    """
    Developer preset with introspection helpers.
    """

    collection: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    cohere_api_key: Optional[str] = None

    def __post_init__(self):
        key = self.cohere_api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("Missing Cohere API key.")

        qdrant = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

        embedder = CohereEmbeddingClient(api_key=key)
        reranker = CohereRerankClient(api_key=key)
        chat = CohereChatClient(api_key=key)

        retriever = RAGRetriever(
            client=qdrant,
            embedder=embedder,
            reranker=reranker,
            collection=self.collection,
            top_k=20,
        )

        rgs = RGS(RGSConfig())

        self.pipeline = RAGPipeline(
            retriever=retriever,
            llm=chat,
            rgs=rgs,
        )

    # -------- developer helpers ---------

    def explain(self, query: str):
        """Return chunks + prompt + answer for debugging."""
        chunks = self.pipeline.retriever.retrieve(query)
        prompt = self.pipeline.rgs.build_prompt(query, chunks)
        answer = self.pipeline.llm.chat(
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
        )
        return {
            "query": query,
            "chunks": chunks,
            "prompt": prompt,
            "answer": answer,
        }
