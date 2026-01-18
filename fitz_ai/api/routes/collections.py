# fitz_ai/api/routes/collections.py
"""Collection management endpoints."""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from fitz_ai.api.dependencies import get_vector_db
from fitz_ai.api.error_handlers import handle_api_errors
from fitz_ai.api.models.schemas import CollectionInfo, CollectionStats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("", response_model=List[CollectionInfo])
@handle_api_errors
async def list_collections() -> List[CollectionInfo]:
    """
    List all available collections.

    Returns collection names and chunk counts.
    """
    vdb = get_vector_db()

    if not hasattr(vdb, "list_collections"):
        raise HTTPException(
            status_code=501,
            detail="Vector DB plugin does not support listing collections",
        )

    collection_names = vdb.list_collections()

    result = []
    for name in collection_names:
        chunk_count = 0
        if hasattr(vdb, "get_collection_stats"):
            try:
                stats = vdb.get_collection_stats(name)
                chunk_count = stats.get("count", stats.get("chunk_count", 0))
            except Exception as e:
                logger.debug(f"Failed to get stats for collection {name}: {e}")
        result.append(CollectionInfo(name=name, chunk_count=chunk_count))

    return result


@router.get("/{name}", response_model=CollectionStats)
@handle_api_errors
async def get_collection(name: str) -> CollectionStats:
    """
    Get statistics for a specific collection.
    """
    vdb = get_vector_db()

    if not hasattr(vdb, "get_collection_stats"):
        raise HTTPException(
            status_code=501,
            detail="Vector DB plugin does not support collection stats",
        )

    stats = vdb.get_collection_stats(name)

    return CollectionStats(
        name=name,
        chunk_count=stats.get("count", stats.get("chunk_count", 0)),
        metadata=stats,
    )


@router.delete("/{name}")
@handle_api_errors
async def delete_collection(name: str) -> dict:
    """
    Delete a collection.

    Returns the number of chunks deleted.
    Also deletes the associated vocabulary file.
    """
    vdb = get_vector_db()

    if not hasattr(vdb, "delete_collection"):
        raise HTTPException(
            status_code=501,
            detail="Vector DB plugin does not support deleting collections",
        )

    result = vdb.delete_collection(name)

    # Some plugins return count, others return None
    deleted_count = result if isinstance(result, int) else 0

    # Also delete associated vocabulary file
    _delete_vocabulary(name)

    return {"deleted": True, "collection": name, "chunks_deleted": deleted_count}


def _delete_vocabulary(collection: str) -> None:
    """Delete vocabulary file associated with a collection."""
    from fitz_ai.core.paths import FitzPaths

    vocab_path = FitzPaths.vocabulary(collection)
    if vocab_path.exists():
        try:
            vocab_path.unlink()
        except Exception as e:
            logger.debug(f"Failed to delete vocabulary for {collection}: {e}")
