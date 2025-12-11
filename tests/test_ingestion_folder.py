from pathlib import Path
from fitz_ingest.ingester.engine import IngestionEngine
from fitz_ingest.chunker.engine import ChunkingEngine
from fitz_ingest.chunker.plugins.simple import SimpleChunker


class DummyClient:
    def __init__(self):
        self.calls = []
        self.collections = []
        self.created = False

    def get_collections(self):
        class C: collections = []
        return C()

    def create_collection(self, collection_name, vectors_config):
        self.created = True

    def upsert(self, collection_name, points):
        self.calls.append((collection_name, points))


def test_ingestion_folder(tmp_path):
    # create 2 small text files
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("hello world")
    f2.write_text("abc def ghi")

    client = DummyClient()

    engine = IngestionEngine(
        client=client,
        collection="col",
        vector_size=4,
        chunker_engine=ChunkingEngine(SimpleChunker(chunk_size=5)),
        embedder=None,  # force zero-vector
    )

    # ingest folder
    engine.ingest_path(tmp_path)

    # Validate
    assert client.created
    assert len(client.calls) == 2

    # Each call = (collection, points[])
    col, pts = client.calls[0]
    assert col == "col"
    assert isinstance(pts, list)
    assert pts[0]["vector"] == [0.0, 0.0, 0.0, 0.0]
