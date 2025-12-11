from fitz_rag.retriever.plugins.dense import RAGRetriever

from fitz_rag.config.schema import (
    RetrieverConfig,
    EmbeddingConfig,
    RerankConfig,
)


# ---------------------------------------------------------
# Minimal mock Qdrant client for retriever tests
# ---------------------------------------------------------
class MockQdrantSearchClient:
    """
    Emulates minimal QdrantClient interface.
    Returns empty search results.
    """
    def search(self, collection_name, vector, limit, with_payload=True):
        return []


# ---------------------------------------------------------
# Dummy embedder
# ---------------------------------------------------------
class DummyEmbedder:
    def embed(self, text: str):
        return [0.0] * 10


def test_rag_retriever_import():
    from fitz_rag.retriever.plugins.dense import RAGRetriever
    assert RAGRetriever is not None


def test_rag_retriever_dummy(monkeypatch):
    embed_cfg = EmbeddingConfig(
        provider="cohere",
        model="embed-english-v3.0",
        api_key=None,
    )

    retriever_cfg = RetrieverConfig(
        collection="test_collection",
        top_k=3,
        qdrant_host="localhost",
        qdrant_port=6333,
    )

    rerank_cfg = RerankConfig(
        provider="cohere",
        model="rerank-english-v3.0",
        api_key=None,
        enabled=False,
    )

    client = MockQdrantSearchClient()

    retriever = RAGRetriever(
        client=client,
        embed_cfg=embed_cfg,
        retriever_cfg=retriever_cfg,
        rerank_cfg=rerank_cfg,
    )

    retriever.embedder = DummyEmbedder()

    result = retriever.retrieve("hello world")

    assert isinstance(result, list)
    assert len(result) == 0
