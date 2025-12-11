from __future__ import annotations

from pathlib import Path

from fitz_ingest.ingester.engine import IngestionEngine
from fitz_ingest.chunker.engine import ChunkingEngine
from fitz_ingest.chunker.plugins.simple import SimpleChunker


class DummyClient:
    def __init__(self):
        self.created = False
        self.upserts = []

    def get_collections(self):
        class _C:
            collections = []   # force creation
        return _C()

    def create_collection(self, *args, **kwargs):
        self.created = True

    def upsert(self, name, points):
        self.upserts.append((name, points))


def test_ingestion_builds_points_correctly(tmp_path):
    file = tmp_path / "x.txt"
    file.write_text("abcdefghi")  # length 9

    engine = IngestionEngine(
        client=DummyClient(),
        collection="col",
        vector_size=4,
        chunker_engine=ChunkingEngine(SimpleChunker(chunk_size=3)),
        embedder=None,    # forces zero-vectors
    )

    engine.ingest_file(file)

    # Validate upserts
    upserts = engine.client.upserts[0][1]
    assert len(upserts) == 3  # 9 chars, chunk size 3 → 3 chunks

    p0 = upserts[0]
    assert p0["id"] == 0
    assert p0["vector"] == [0.0, 0.0, 0.0, 0.0]  # zero vector

    payload = p0["payload"]
    assert "text" in payload
    assert "file" in payload  # derived from metadata["source_file"]
