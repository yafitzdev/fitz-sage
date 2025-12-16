# tests/test_local_faiss_vector_db_default_config.py

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")

from fitz.backends.local_vector_db.config import LocalVectorDBConfig
from fitz.backends.local_vector_db.faiss import FaissLocalVectorDB
from fitz.core.models.chunk import Chunk


def _make_chunk(i: int, dim: int) -> Chunk:
    chunk = Chunk(
        id=f"c{i}",
        doc_id="doc",
        content=f"text {i}",
        chunk_index=i,
        metadata={"idx": i},
    )

    # attach runtime embedding (as embedding engine would)
    object.__setattr__(
        chunk,
        "embedding",
        np.ones(dim, dtype="float32") * i,
    )
    return chunk


def test_local_faiss_vector_db_uses_default_path_and_persists(tmp_path):
    """
    Verifies that:
    - LocalVectorDBConfig defaults are usable
    - The DB creates its directory
    - Persistence works without custom path overrides
    """

    dim = 3

    # isolate HOME to avoid touching real user directory
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    Path.home = lambda: fake_home  # type: ignore

    config = LocalVectorDBConfig()  # default config
    db = FaissLocalVectorDB(dim=dim, config=config)

    # default base path must be ~/.fitz/vector_db
    expected_path = fake_home / ".fitz" / "vector_db"
    assert db._base_path == expected_path
    assert expected_path.exists()

    chunks = [_make_chunk(i, dim) for i in range(3)]
    db.add(chunks)
    db.persist()

    # verify files were created
    assert (expected_path / "index.faiss").exists()
    assert (expected_path / "payloads.npy").exists()

    # reload using default config again
    db_reloaded = FaissLocalVectorDB(dim=dim, config=config)
    assert db_reloaded.count() == 3
