from __future__ import annotations

from typing import Dict
from fitz_rag.vector_db.base import VectorDBPlugin
from fitz_rag.vector_db.plugins import QdrantVectorDB


_REGISTRY: Dict[str, VectorDBPlugin] = {
    "qdrant": QdrantVectorDB(),
}


def get_vector_db_plugin(name: str) -> VectorDBPlugin:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown vector DB provider: {name!r}") from exc


def available_vector_dbs() -> list[str]:
    return sorted(_REGISTRY.keys())
