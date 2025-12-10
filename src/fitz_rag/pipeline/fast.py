from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

from qdrant_client import QdrantClient

from fitz_rag.pipeline.engine import RAGPipeline
from fitz_rag.retriever.dense_retriever import RAGRetriever
from fitz_rag.generation.rgs import RGS, RGSConfig
from fitz_rag.llm.embedding_client import CohereEmbeddingClient
from fitz_rag.llm.chat_client import CohereChatClient


@dataclass
class FastRAG:
    """
    Lowest-latency pipeline.
    No reranking → retriever returns top_k dense results only.
    """

    collection: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    cohere_api_key: Optional[str] = None

    def __post_init__(self):
        key = self.cohere_api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("Missing COHERE_API_KEY")

        qdrant = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

        embedder = CohereEmbeddingClient(api_key=key)
        chat = CohereChatClient(api_key=key)

        retriever = RAGRetriever(
            client=qdrant,
            embedder=embedder,
            reranker=None,      # no reranking
            collection=self.collection,
            top_k=5,
        )

        rgs = RGS(RGSConfig())

        self.pipeline = RAGPipeline(
            retriever=retriever,
            llm=chat,
            rgs=rgs,
        )

    def ask(self, query: str):
        return self.pipeline.run(query)
