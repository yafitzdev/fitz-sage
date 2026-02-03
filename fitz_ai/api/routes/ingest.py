# fitz_ai/api/routes/ingest.py
"""Document ingestion endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from fitz_ai.api.dependencies import get_service
from fitz_ai.api.error_handlers import handle_api_errors
from fitz_ai.api.models.schemas import IngestRequest, IngestResponse

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
@handle_api_errors
async def ingest(request: IngestRequest) -> IngestResponse:
    """
    Ingest documents into the knowledge base.

    Provide a path to a file or directory. Documents will be chunked,
    embedded, and stored in the vector database.
    """
    service = get_service()

    result = service.ingest(
        source=request.source,
        collection=request.collection,
        clear_existing=request.clear_existing,
    )

    return IngestResponse(
        documents=result.documents_processed,
        chunks=result.chunks_created,
        collection=result.collection,
    )
