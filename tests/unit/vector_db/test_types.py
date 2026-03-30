# tests/unit/vector_db/test_types.py
"""Tests for vector_db type models."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.tier1

from fitz_sage.vector_db.types import (
    CollectionInfo,
    CollectionStats,
    ScrollResponse,
    VectorPoint,
)


class TestVectorPoint:
    def test_to_dict(self):
        point = VectorPoint(id="abc", vector=[1.0, 2.0], payload={"key": "val"})
        d = point.to_dict()
        assert d["id"] == "abc"
        assert d["vector"] == [1.0, 2.0]
        assert d["payload"] == {"key": "val"}

    def test_to_dict_empty_payload(self):
        point = VectorPoint(id="abc", vector=[0.5])
        d = point.to_dict()
        assert d["payload"] == {}

    def test_default_payload_is_empty_dict(self):
        point = VectorPoint(id="x", vector=[])
        assert point.payload == {}

    def test_default_payload_not_shared(self):
        p1 = VectorPoint(id="a", vector=[])
        p2 = VectorPoint(id="b", vector=[])
        p1.payload["key"] = "val"
        assert "key" not in p2.payload


class TestCollectionStats:
    def test_defaults(self):
        stats = CollectionStats(points_count=100, vectors_count=100)
        assert stats.status == "green"
        assert stats.vector_size is None
        assert stats.indexed_vectors_count is None

    def test_full_construction(self):
        stats = CollectionStats(
            points_count=500,
            vectors_count=500,
            status="yellow",
            vector_size=768,
            indexed_vectors_count=450,
            segments_count=3,
        )
        assert stats.points_count == 500
        assert stats.segments_count == 3


class TestCollectionInfo:
    def test_defaults(self):
        info = CollectionInfo(name="test", vector_size=384)
        assert info.distance == "cosine"
        assert info.points_count == 0


class TestScrollResponse:
    def test_defaults(self):
        resp = ScrollResponse(points=[])
        assert resp.next_offset is None

    def test_with_points(self):
        resp = ScrollResponse(
            points=[{"id": "1", "payload": {}, "vector": None}],
            next_offset="abc",
        )
        assert len(resp.points) == 1
        assert resp.next_offset == "abc"
