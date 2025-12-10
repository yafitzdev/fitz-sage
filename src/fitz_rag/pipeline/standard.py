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
class StandardRAG:
    """
    Configurable, but still easy to use.
    Allows changing models, top_k, etc.
    """

    collection: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    cohere_api_key: Optional[str] = None
    embed_model: str = "embed-english-v3.0"
    rerank_model: str = "rerank-v3.5"

    top_k: int = 20
    final_top_k: int = 5

    def __post_init__(self):
        key = self.cohere_api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("Missing COHERE_API_KEY")

        qdrant = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

        embedder = CohereEmbeddingClient(api_key=key, model=self.embed_model)
        reranker = CohereRerankClient(api_key=key, model=self.rerank_model)
        chat = CohereChatClient(api_key=key)

        retriever = RAGRetriever(
            client=qdrant,
            embedder=embedder,
            reranker=reranker,
            collection=self.collection,
            top_k=self.top_k,
        )

        rgs = RGS(RGSConfig())

        self.pipeline = RAGPipeline(
            retriever=retriever,
            llm=chat,
            rgs=rgs,
        )

    def ask(self, query: str):
        return self.pipeline.run(query)
