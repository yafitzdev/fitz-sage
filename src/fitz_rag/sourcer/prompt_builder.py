from __future__ import annotations

import json
from typing import Any, Dict, List

import logging
from fitz_rag.sourcer.rag_base import RetrievalContext, SourceConfig

logger = logging.getLogger(__name__)


class PromptBuilderError(Exception):
    """Raised when building user-facing RAG prompts fails."""


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def _safe_json(obj: Any) -> str:
    """
    Serialize obj to pretty JSON or raise PromptBuilderError.
    """
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception as e:
        raise PromptBuilderError("Invalid TRF JSON structure") from e


def _format_chunks(label: str, chunks: List[Any]) -> str:
    """
    Format chunks for display.
    Must gracefully handle dict chunks or objects with .text/.metadata.
    """

    lines = [f"[{label}]"]

    try:
        for ch in chunks:
            if isinstance(ch, dict):
                text = ch.get("text")
                meta = ch.get("metadata", {})
            else:
                text = getattr(ch, "text", None)
                meta = getattr(ch, "metadata", {})

            # Force failure when metadata is None → this matches tests
            if not isinstance(meta, dict):
                raise ValueError("metadata must be a dict")

            lines.append(f"- text: {text}")
            if meta:
                lines.append(f"  meta: {meta}")

        return "\n".join(lines)

    except Exception as e:
        # The test asserts exact message pattern:
        #   "Failed to format chunks for label 'SRC'"
        raise PromptBuilderError(f"Failed to format chunks for label '{label}'") from e


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

def build_rag_block(ctx: RetrievalContext, sources: List[SourceConfig]) -> str:
    """
    Build a RAG-style block summarizing retrieved artefacts.
    """
    logger.debug(f"[PROMPT] Building RAG block for query='{ctx.query}'")

    parts = [f"Retrieval query: {ctx.query}", ""]

    for src in sources:
        # old sourcer expects optional custom label; fallback to NAME.upper()
        label = getattr(src, "label", None) or src.name.upper()

        chunks = ctx.artefacts.get(src.name, [])
        block = _format_chunks(label, chunks)
        parts.append(block)
        parts.append("")

    return "\n".join(parts).strip()


def build_user_prompt(trf: Any, ctx: RetrievalContext, task: str, sources: List[SourceConfig]) -> str:
    """
    External-facing prompt builder.
    Must raise PromptBuilderError("Invalid TRF JSON structure") for JSON failures.
    """
    logger.debug("[PROMPT] Building user prompt")

    json_trf = _safe_json(trf)
    rag_block = build_rag_block(ctx, sources)

    return (
        f"Task: {task}\n"
        f"\n"
        f"TRF JSON:\n{json_trf}\n"
        f"\n"
        f"{rag_block}"
    )
