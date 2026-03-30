# tests/unit/vector_db/test_writer.py
"""Tests for vector_db writer utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.tier1

from fitz_sage.core.chunk import Chunk
from fitz_sage.vector_db.writer import (
    VectorDBWriter,
    chunks_to_points,
    compute_chunk_hash,
)


def _make_chunk(
    id: str = "chunk-1",
    doc_id: str = "doc-1",
    content: str = "Hello world",
    chunk_index: int = 0,
    metadata: dict | None = None,
) -> Chunk:
    return Chunk(
        id=id,
        doc_id=doc_id,
        content=content,
        chunk_index=chunk_index,
        metadata=metadata or {},
    )


class TestComputeChunkHash:
    def test_deterministic(self):
        chunk = _make_chunk()
        assert compute_chunk_hash(chunk) == compute_chunk_hash(chunk)

    def test_different_content_different_hash(self):
        c1 = _make_chunk(content="hello")
        c2 = _make_chunk(content="world")
        assert compute_chunk_hash(c1) != compute_chunk_hash(c2)

    def test_different_doc_id_different_hash(self):
        c1 = _make_chunk(doc_id="doc-1")
        c2 = _make_chunk(doc_id="doc-2")
        assert compute_chunk_hash(c1) != compute_chunk_hash(c2)

    def test_different_chunk_index_different_hash(self):
        c1 = _make_chunk(chunk_index=0)
        c2 = _make_chunk(chunk_index=1)
        assert compute_chunk_hash(c1) != compute_chunk_hash(c2)

    def test_returns_hex_string(self):
        chunk = _make_chunk()
        h = compute_chunk_hash(chunk)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest


class TestChunksToPoints:
    def test_basic_conversion(self):
        chunks = [_make_chunk(id="c1", doc_id="d1", content="text", chunk_index=0)]
        vectors = [[1.0, 2.0, 3.0]]
        points = chunks_to_points(chunks, vectors)

        assert len(points) == 1
        point = points[0]
        assert point["id"] == "c1"
        assert point["vector"] == [1.0, 2.0, 3.0]
        assert point["payload"]["doc_id"] == "d1"
        assert point["payload"]["content"] == "text"
        assert point["payload"]["chunk_index"] == 0
        assert "chunk_hash" in point["payload"]

    def test_multiple_chunks(self):
        chunks = [_make_chunk(id=f"c{i}", chunk_index=i) for i in range(3)]
        vectors = [[float(i)] for i in range(3)]
        points = chunks_to_points(chunks, vectors)
        assert len(points) == 3
        assert [p["id"] for p in points] == ["c0", "c1", "c2"]

    def test_metadata_preserved(self):
        chunks = [_make_chunk(metadata={"source_file": "readme.md", "page": 5})]
        vectors = [[0.1]]
        points = chunks_to_points(chunks, vectors)
        assert points[0]["payload"]["metadata"]["source_file"] == "readme.md"
        assert points[0]["payload"]["metadata"]["page"] == 5

    def test_empty_input(self):
        assert chunks_to_points([], []) == []

    def test_none_metadata_becomes_empty_dict(self):
        chunk = _make_chunk()
        chunk.metadata = None
        points = chunks_to_points([chunk], [[0.1]])
        assert points[0]["payload"]["metadata"] == {}


class TestVectorDBWriter:
    def test_upsert_delegates_to_client(self):
        client = MagicMock()
        writer = VectorDBWriter(client=client)
        chunks = [_make_chunk(id="c1")]
        vectors = [[1.0, 2.0]]

        writer.upsert("my_collection", chunks, vectors)

        client.upsert.assert_called_once()
        call_args = client.upsert.call_args
        assert call_args[0][0] == "my_collection"
        assert len(call_args[0][1]) == 1
        assert call_args[0][1][0]["id"] == "c1"

    def test_upsert_multiple_chunks(self):
        client = MagicMock()
        writer = VectorDBWriter(client=client)
        chunks = [_make_chunk(id=f"c{i}", chunk_index=i) for i in range(5)]
        vectors = [[float(i)] for i in range(5)]

        writer.upsert("col", chunks, vectors)

        points = client.upsert.call_args[0][1]
        assert len(points) == 5
