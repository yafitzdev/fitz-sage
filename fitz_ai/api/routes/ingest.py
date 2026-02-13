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

    Provide a path to a file or directory. Documents will be parsed,
    indexed for code symbols and document sections, and stored in
    the database.
    """
    service = get_service()

    result = service.ingest(
        source=request.source,
        collection=request.collection,
        force=request.force,
    )

    return IngestResponse(
        documents=result.documents_processed,
        sections=result.sections_created,
        symbols=result.symbols_created,
        collection=result.collection,
    )
