# fitz_ai/api/routes/query.py
"""Query and chat endpoints."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter

from fitz_ai.api.dependencies import get_fitz_instance
from fitz_ai.api.error_handlers import handle_api_errors
from fitz_ai.api.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)
from fitz_ai.retrieval.rewriter.types import ConversationContext, ConversationMessage

router = APIRouter(tags=["query"])


def _to_conversation_context(history: List[ChatMessage]) -> ConversationContext | None:
    """Convert API history to ConversationContext for query rewriting."""
    if not history:
        return None
    messages = [ConversationMessage(role=msg.role, content=msg.content) for msg in history]
    return ConversationContext(history=messages)


@router.post("/query", response_model=QueryResponse)
@handle_api_errors
async def query(request: QueryRequest) -> QueryResponse:
    """
    Query the knowledge base.

    Submit a question and receive an answer with sources.
    Optionally include conversation_history for query rewriting
    (resolves pronouns like "their" to referenced entities).
    """
    f = get_fitz_instance(request.collection)
    context = _to_conversation_context(request.conversation_history)
    answer = f.ask(request.question, top_k=request.top_k, conversation_context=context)

    sources = [
        SourceInfo(
            source_id=p.source_id,
            excerpt=p.excerpt,
            metadata=p.metadata,
        )
        for p in answer.provenance
    ]

    return QueryResponse(
        text=answer.text,
        mode=answer.mode.value if answer.mode else None,
        sources=sources,
    )


@router.post("/chat", response_model=ChatResponse)
@handle_api_errors
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Multi-turn chat with the knowledge base.

    Send a message along with conversation history. The server is stateless;
    the client is responsible for maintaining and sending the history.

    Query rewriting automatically resolves pronouns and references using
    the conversation history (e.g., "their products" → "TechCorp's products").
    """
    f = get_fitz_instance(request.collection)
    context = _to_conversation_context(request.history)
    answer = f.ask(request.message, top_k=request.top_k, conversation_context=context)

    sources = [
        SourceInfo(
            source_id=p.source_id,
            excerpt=p.excerpt,
            metadata=p.metadata,
        )
        for p in answer.provenance
    ]

    return ChatResponse(
        text=answer.text,
        mode=answer.mode.value if answer.mode else None,
        sources=sources,
    )
