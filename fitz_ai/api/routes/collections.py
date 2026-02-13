# fitz_ai/api/routes/collections.py
"""Collection management endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from fitz_ai.api.dependencies import get_service
from fitz_ai.api.error_handlers import handle_api_errors
from fitz_ai.api.models.schemas import CollectionInfo, CollectionStats

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("", response_model=list[CollectionInfo])
@handle_api_errors
async def list_collections() -> list[CollectionInfo]:
    """List all available collections."""
    service = get_service()
    collections = service.list_collections()

    return [CollectionInfo(name=c.name, item_count=c.chunk_count) for c in collections]


@router.get("/{name}", response_model=CollectionStats)
@handle_api_errors
async def get_collection(name: str) -> CollectionStats:
    """
    Get statistics for a specific collection.
    """
    service = get_service()
    info = service.get_collection(name)

    return CollectionStats(
        name=info.name,
        item_count=info.chunk_count,
        metadata=info.metadata,
    )


@router.delete("/{name}")
@handle_api_errors
async def delete_collection(name: str) -> dict:
    """
    Delete a collection.

    Returns whether the collection was deleted.
    """
    service = get_service()
    deleted = service.delete_collection(name)

    return {"deleted": deleted, "collection": name}
