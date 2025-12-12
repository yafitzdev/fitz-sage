import pytest
from dataclasses import dataclass

from core.vector_db.writer import VectorDBWriter, compute_chunk_hash
from core.vector_db.base import VectorRecord


# Dummy embedding engine
class DummyEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [1.0, 2.0, 3.0]


# Dummy vector DB
class DummyVectorDB:
    def __init__(self):
        self.upserts = []

    def upsert(self, collection, records):
        self.upserts.append((collection, records))


@dataclass
class DummyChunk:
    id: str | None
    text: str
    metadata: dict


def test_writer_basic_dedupe_and_write():
    embedder = DummyEmbedder()
    vectordb = DummyVectorDB()
    writer = VectorDBWriter(embedder=embedder, vectordb=vectordb, deduplicate=True)

    chunks = [
        DummyChunk(None, "hello world", {"source": "s1"}),
        DummyChunk(None, "hello world", {"source": "s1"}),  # duplicate
    ]

    count = writer.write("colA", chunks)

    # Only one chunk written
    assert count == 1
    assert len(vectordb.upserts) == 1

    collection, records = vectordb.upserts[0]
    assert collection == "colA"
    assert isinstance(records[0], VectorRecord)

    # Embedding called once
    assert embedder.calls == ["hello world"]

    # Verify hash is included
    h = compute_chunk_hash(chunks[0])
    assert records[0].payload["chunk_hash"] == h
