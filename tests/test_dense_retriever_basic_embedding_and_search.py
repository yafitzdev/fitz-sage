from dataclasses import dataclass

from rag.retrieval.plugins.dense import DenseRetrievalPlugin
from rag.config.schema import RetrieverConfig


class DummyEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text: str):
        self.calls.append(text)
        return [0.1, 0.2, 0.3]


class DummyHit:
    def __init__(self, id, content, score=0.5):
        self.id = id
        self.score = score
        self.payload = {"content": content, "metadata": {"file": "dummy.txt"}}


class DummyClient:
    def __init__(self):
        self.last_query = None

    def search(self, collection_name, query_vector, limit, with_payload=True):
        self.last_query = (collection_name, query_vector, limit, with_payload)
        return [
            DummyHit("h1", "Hello world"),
            DummyHit("h2", "Another doc"),
        ]


def test_dense_retriever_calls_embed_and_search():
    embedder = DummyEmbedder()
    client = DummyClient()

    retriever = DenseRetrievalPlugin(
        client=client,
        retriever_cfg=RetrieverConfig(collection="testcol", top_k=2),
        embedder=embedder,
        rerank_engine=None,
    )

    chunks = retriever.retrieve("hello world")

    assert embedder.calls == ["hello world"]

    collection_name, vec, limit, with_payload = client.last_query
    assert collection_name == "testcol"
    assert vec == [0.1, 0.2, 0.3]
    assert limit == 2
    assert with_payload is True

    assert isinstance(chunks, list)
    assert len(chunks) == 2
    # Don’t hard-pin the Chunk class; just validate the output contract.
    assert getattr(chunks[0], "content") == "Hello world"
