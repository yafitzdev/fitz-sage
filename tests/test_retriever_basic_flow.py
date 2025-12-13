import pytest
from dataclasses import dataclass

from rag.retrieval.plugins.dense import DenseRetrievalPlugin
from rag.config.schema import RetrieverConfig
from core.exceptions.llm import EmbeddingError


class DummyEmbedder:
    def embed(self, text):
        return [0.1, 0.2, 0.3]


class DummyClient:
    class Hit:
        id = "1"
        score = 0.9
        payload = {"content": "hello world", "metadata": {"file": "doc.txt"}}

    def search(self, *args, **kwargs):
        return [DummyClient.Hit()]


def test_retriever_basic_flow():
    r = DenseRetrievalPlugin(
        client=DummyClient(),
        retriever_cfg=RetrieverConfig(collection="demo", top_k=1),
        embedder=DummyEmbedder(),
        rerank_engine=None,
    )

    out = r.retrieve("hi")
    assert isinstance(out, list)
    assert getattr(out[0], "content") == "hello world"


def test_retriever_embedding_failure_raises_embedding_error():
    class BadEmbedder:
        def embed(self, _):
            raise RuntimeError("oh no")

    r = DenseRetrievalPlugin(
        client=DummyClient(),
        retriever_cfg=RetrieverConfig(collection="demo", top_k=1),
        embedder=BadEmbedder(),
        rerank_engine=None,
    )

    with pytest.raises(EmbeddingError):
        r.retrieve("hi")
