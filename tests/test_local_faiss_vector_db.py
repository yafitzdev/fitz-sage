# tests/test_local_faiss_vector_db.py

from pathlib import Path

import numpy as np

from fitz.backends.local_vector_db.config import LocalVectorDBConfig
from fitz.backends.local_vector_db.faiss import FaissLocalVectorDB
from fitz.core.models.chunk import Chunk


def _make_chunk(i: int, dim: int) -> Chunk:
    chunk = Chunk(
        id=f"c{i}",
        doc_id="doc",
        content=f"text {i}",
        chunk_index=i,
        metadata={"i": i},
    )

    object.__setattr__(
        chunk,
        "embedding",
        np.ones(dim, dtype="float32") * i,
    )

    return chunk


def test_local_faiss_vector_db_add_search_and_persist(tmp_path: Path):
    dim = 4
    config = LocalVectorDBConfig(path=tmp_path, persist=True)

    db = FaissLocalVectorDB(dim=dim, config=config)

    chunks = [_make_chunk(i, dim) for i in range(5)]
    db.add(chunks)

    assert db.count() == 5

    query = np.ones(dim, dtype="float32") * 3
    results = db.search(
        collection_name="default",
        query_vector=query.tolist(),
        limit=2,
    )

    assert len(results) == 2
    assert results[0].id in {"c2", "c3", "c4"}
    assert "i" in results[0].payload

    db.persist()

    db_reloaded = FaissLocalVectorDB(dim=dim, config=config)
    assert db_reloaded.count() == 5

    results_reloaded = db_reloaded.search(
        collection_name="default",
        query_vector=query.tolist(),
        limit=1,
    )

    assert results_reloaded[0].id == results[0].id
