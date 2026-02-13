# fitz_ai/api/routes/ingest.py
"""Document point/indexing endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from fitz_ai.api.dependencies import get_service
from fitz_ai.api.error_handlers import handle_api_errors
from fitz_ai.api.models.schemas import PointRequest, PointResponse

router = APIRouter(tags=["ingest"])


@router.post("/point", response_model=PointResponse)
@handle_api_errors
async def point(request: PointRequest) -> PointResponse:
    """
    Point at a folder for progressive querying.

    Queries work immediately via agentic search. Background indexing
    runs silently — queries get progressively faster over time.
    """
    service = get_service()

    manifest = service.point(
        source=request.source,
        collection=request.collection,
    )

    file_count = len(manifest.entries())

    return PointResponse(
        files=file_count,
        collection=request.collection,
    )
