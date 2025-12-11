"""
Context builder for fitz_rag.

This module handles:
- merging retrieved chunks
- grouping by document
- deduplication
- packing the final context window for RAG prompts

This is NOT responsible for chunk creation (that belongs to fitz_ingest).
"""

from __future__ import annotations

from typing import List, Dict, Any

from fitz_rag.exceptions.pipeline import PipelineError


# ---------------------------------------------------------
# Internal helper
# ---------------------------------------------------------
def _normalize_text(text: str) -> str:
    try:
        return (
            text.replace("\r", "")
            .replace("\t", " ")
            .strip()
        )
    except Exception as e:
        raise PipelineError("Failed normalizing text") from e


# ---------------------------------------------------------
# 1) Deduplication
# ---------------------------------------------------------
def dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        seen = set()
        output = []

        for ch in chunks:
            text_norm = _normalize_text(ch.get("text", ""))
            if text_norm in seen:
                continue
            seen.add(text_norm)
            ch["text"] = text_norm
            output.append(ch)

        return output
    except Exception as e:
        raise PipelineError("Failed deduplication of chunks") from e


# ---------------------------------------------------------
# 2) Grouping by file / document
# ---------------------------------------------------------
def group_by_document(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    try:
        groups = {}
        for ch in chunks:
            f = ch.get("file", "unknown")
            groups.setdefault(f, []).append(ch)
        return groups
    except Exception as e:
        raise PipelineError("Failed grouping chunks by document") from e


# ---------------------------------------------------------
# 3) Merge adjacent chunks
# ---------------------------------------------------------
def merge_adjacent_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        if not chunks:
            return []

        merged = []
        buffer_text = chunks[0]["text"]
        buffer_meta = dict(chunks[0])

        for prev, curr in zip(chunks, chunks[1:]):
            same_doc = curr.get("file") == prev.get("file")

            if same_doc:
                buffer_text += "\n" + curr["text"]
            else:
                merged.append({
                    "text": buffer_text,
                    **{k: v for k, v in buffer_meta.items() if k != "text"},
                })
                buffer_text = curr["text"]
                buffer_meta = curr

        merged.append({
            "text": buffer_text,
            **{k: v for k, v in buffer_meta.items() if k != "text"},
        })

        return merged
    except Exception as e:
        raise PipelineError("Failed merging adjacent chunks") from e


# ---------------------------------------------------------
# 4) Pack context window
# ---------------------------------------------------------
def pack_context_window(
    chunks: List[Dict[str, Any]],
    max_chars: int = 6000,
) -> List[Dict[str, Any]]:
    try:
        packed = []
        total = 0

        for ch in chunks:
            txt = ch["text"]
            if total + len(txt) > max_chars:
                break
            packed.append(ch)
            total += len(txt)

        return packed
    except Exception as e:
        raise PipelineError("Failed packing context window") from e


# ---------------------------------------------------------
# 5) Main API
# ---------------------------------------------------------
def build_context(
    retrieved_chunks: List[Dict[str, Any]],
    max_chars: int = 6000,
) -> str:
    try:
        clean_chunks = dedupe_chunks(retrieved_chunks)
        groups = group_by_document(clean_chunks)

        merged_per_doc = []
        for _, chs in groups.items():
            merged = merge_adjacent_chunks(chs)
            merged_per_doc.extend(merged)

        merged_list = merged_per_doc
        packed = pack_context_window(merged_list, max_chars=max_chars)

        final_sections = []
        for ch in packed:
            file = ch.get("file", "unknown")
            text = ch["text"]

            section = (
                f"### Source: {file}\n"
                f"{text}\n"
            )
            final_sections.append(section)

        return "\n".join(final_sections)
    except Exception as e:
        raise PipelineError("Failed to build context window") from e
