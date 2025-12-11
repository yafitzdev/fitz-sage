# src/fitz_rag/sourcer/prompt_builder.py
"""
Prompt builder utilities for fitz-rag.

This module converts:
    - TRF JSON
    - RetrievalContext
    - List[SourceConfig]
    - Prompt text

into a unified user message to send to the LLM.
"""

from __future__ import annotations

import json
from typing import Dict, List

from fitz_rag.core import RetrievedChunk
from fitz_rag.sourcer.rag_base import RetrievalContext, SourceConfig
from fitz_rag.config import get_config

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import PROMPT

_cfg = get_config()
logger = get_logger(__name__)


class PromptBuilderError(Exception):
    """Raised when prompt construction fails."""


def _format_chunks(label: str, chunks: List[RetrievedChunk]) -> str:
    logger.debug(f"{PROMPT} Formatting {len(chunks)} chunks for label='{label}'")

    try:
        if not chunks:
            return f"{label}: <no chunks>"

        out = [f"{label} (top {len(chunks)}):"]
        for idx, c in enumerate(chunks, 1):
            src = c.metadata.get("file") or c.metadata.get("source") or ""
            out.append(f"[{idx}] score={c.score:.4f} | {src}")
            out.append(c.text.strip())
            out.append("")

        return "\n".join(out).strip()

    except Exception as e:
        logger.error(f"{PROMPT} Failed formatting chunks for '{label}'")
        raise PromptBuilderError(f"Failed to format chunks for label '{label}': {e}") from e


def build_rag_block(ctx: RetrievalContext, sources: List[SourceConfig]) -> str:
    logger.debug(f"{PROMPT} Building RAG block for query='{ctx.query}'")

    try:
        parts = [f"Retrieval query: {ctx.query}", ""]

        for src in sources:
            label = src.label or src.name.upper()
            chunks = ctx.artefacts.get(src.name, [])
            parts.append(_format_chunks(label, chunks))
            parts.append("")

        return "\n".join(parts).strip()

    except Exception as e:
        logger.error(f"{PROMPT} Failed building RAG block")
        raise PromptBuilderError(f"Failed to build RAG block: {e}") from e


def build_user_prompt(
    trf: Dict,
    ctx: RetrievalContext,
    prompt_text: str,
    sources: List[SourceConfig],
    max_trf_json_chars: int | None = None,
) -> str:

    logger.debug(f"{PROMPT} Building final user prompt")

    try:
        if max_trf_json_chars is None:
            max_trf_json_chars = _cfg.get("retriever", {}).get("max_trf_json_chars", None)

        # TRF JSON formatting
        try:
            trf_json = json.dumps(trf, indent=2)
        except Exception as je:
            logger.error(f"{PROMPT} Invalid TRF JSON structure")
            raise PromptBuilderError(f"Invalid TRF JSON structure: {je}") from je

        if max_trf_json_chars and len(trf_json) > max_trf_json_chars:
            trf_json = trf_json[:max_trf_json_chars] + "\n...[TRF JSON truncated]..."

        # Build retrieval block
        rag_block = build_rag_block(ctx, sources)

        return (
            f"=== TRF JSON ===\n{trf_json}\n\n"
            f"=== RAG CONTEXT ===\n{rag_block}\n\n"
            f"=== TASK ===\n{prompt_text}\n"
        )

    except Exception as e:
        logger.error(f"{PROMPT} Failed building user prompt")
        raise PromptBuilderError(f"Failed to build user prompt: {e}") from e
