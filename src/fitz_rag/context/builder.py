from __future__ import annotations

from typing import Any, Dict, List, Mapping

from fitz_rag.exceptions.pipeline import PipelineError

ChunkDict = Dict[str, Any]


def _normalize_text(text: str) -> str:
    """
    Normalize text for deduplication:
    - strip leading/trailing whitespace
    - collapse internal whitespace to single spaces
    """
    if text is None:
        return ""
    # Split on any whitespace and re-join with single spaces
    return " ".join(str(text).split())


def _to_chunk_dict(chunk_like: Any) -> ChunkDict:
    """
    Universal normalization layer for 'chunks'.

    Accepts:
    - dicts with keys like text, metadata, id, score, file
    - legacy Chunk objects with .text/.content/.metadata/.id/.score
    - simple test objects with .content and .metadata

    Returns a canonical dict:

        {
            "id": str | None,
            "text": str,
            "metadata": dict,
            "score": float | None,
        }

    Also normalizes 'file' from either metadata["file"] or top-level "file".
    """
    # If it's already a dict, start from a shallow copy
    if isinstance(chunk_like, dict):
        data = dict(chunk_like)
        text = (
            data.get("text")
            if data.get("text") is not None
            else data.get("content")
        )
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}

        # If there's a top-level 'file', ensure it is reflected in metadata
        file_val = data.get("file") or metadata.get("file")
        if file_val is not None:
            metadata = dict(metadata)
            metadata.setdefault("file", file_val)

        return {
            "id": data.get("id"),
            "text": str(text) if text is not None else "",
            "metadata": metadata,
            "score": data.get("score"),
        }

    # Fallback: treat as an object with attributes
    text = getattr(chunk_like, "text", None)
    if text is None:
        text = getattr(chunk_like, "content", "")

    metadata = getattr(chunk_like, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    cid = getattr(chunk_like, "id", None)
    score = getattr(chunk_like, "score", None)

    # Optional 'file' attribute
    file_val = getattr(chunk_like, "file", None) or metadata.get("file")
    if file_val is not None:
        metadata = dict(metadata)
        metadata.setdefault("file", file_val)

    return {
        "id": cid,
        "text": str(text),
        "metadata": metadata,
        "score": score,
    }


def dedupe_chunks(chunks: List[Any]) -> List[ChunkDict]:
    """
    Deduplicate based on normalized text.
    The first occurrence is kept.
    Returns canonical dict chunks.
    """
    try:
        seen = set()
        output: List[ChunkDict] = []

        for ch in chunks:
            c = _to_chunk_dict(ch)
            text_norm = _normalize_text(c["text"])
            if text_norm in seen:
                continue

            seen.add(text_norm)

            # Create a new dict with normalized text
            c_out = dict(c)
            c_out["text"] = text_norm
            output.append(c_out)

        return output

    except Exception as e:
        raise PipelineError("Failed deduplication of chunks") from e


def group_by_document(chunks: List[Any]) -> Dict[str, List[ChunkDict]]:
    """
    Group chunks by their 'file' origin.

    Priority:
    - metadata["file"]
    - top-level "file"
    - "unknown"
    """
    try:
        groups: Dict[str, List[ChunkDict]] = {}

        for ch in chunks:
            c = _to_chunk_dict(ch)
            meta = c.get("metadata", {}) or {}
            file_val = meta.get("file") or getattr(ch, "file", None)

            if file_val is None and isinstance(ch, dict):
                file_val = ch.get("file")

            if file_val is None:
                file_val = "unknown"

            # Ensure metadata["file"] is set for downstream consumers
            meta = dict(meta)
            meta.setdefault("file", file_val)
            c["metadata"] = meta

            groups.setdefault(str(file_val), []).append(c)

        return groups

    except Exception as e:
        raise PipelineError("Failed grouping chunks by document") from e


def merge_adjacent_chunks(chunks: List[Any]) -> List[ChunkDict]:
    """
    Merge adjacent chunks that belong to the same document (same 'file').
    """
    try:
        if not chunks:
            return []

        normed = [_to_chunk_dict(ch) for ch in chunks]

        merged: List[ChunkDict] = []

        def file_of(c: ChunkDict) -> str:
            meta = c.get("metadata", {}) or {}
            return str(meta.get("file", "unknown"))

        buffer = dict(normed[0])
        buffer_text = buffer.get("text", "")
        buffer_file = file_of(buffer)

        for curr in normed[1:]:
            curr_file = file_of(curr)

            if curr_file == buffer_file:
                # same doc → append text
                buffer_text += "\n" + curr.get("text", "")
            else:
                # flush buffer
                buffer["text"] = buffer_text
                merged.append(buffer)

                # start new buffer
                buffer = dict(curr)
                buffer_text = buffer.get("text", "")
                buffer_file = curr_file

        # flush last buffer
        buffer["text"] = buffer_text
        merged.append(buffer)

        return merged

    except Exception as e:
        raise PipelineError("Failed merging adjacent chunks") from e


def pack_context_window(
    chunks: List[Any],
    max_chars: int = 6000,
) -> List[ChunkDict]:
    """
    Return the largest prefix of chunks whose total text does not exceed max_chars.
    Returns canonical dict chunks.
    """
    try:
        packed: List[ChunkDict] = []
        total = 0

        for ch in chunks:
            c = _to_chunk_dict(ch)
            t = c.get("text", "")
            if not isinstance(t, str):
                t = str(t)

            if total + len(t) > max_chars:
                break

            packed.append(c)
            total += len(t)

        return packed

    except Exception as e:
        raise PipelineError("Failed packing context window") from e


def build_context(
    retrieved_chunks: List[Any],
    max_chars: int = 6000,
) -> str:
    """
    High-level context builder used by RAG pipeline.

    Steps:
    - dedupe
    - group by document
    - merge adjacent within each doc
    - pack into max_chars
    - render as markdown sections
    """
    try:
        clean_chunks = dedupe_chunks(retrieved_chunks)
        groups = group_by_document(clean_chunks)

        merged_per_doc: List[ChunkDict] = []
        for _, chs in groups.items():
            merged_per_doc.extend(merge_adjacent_chunks(chs))

        packed = pack_context_window(merged_per_doc, max_chars=max_chars)

        final_sections: List[str] = []
        for ch in packed:
            c = _to_chunk_dict(ch)
            meta = c.get("metadata", {}) or {}
            file_val = meta.get("file") or "unknown"
            text = c.get("text", "")

            section = (
                f"### Source: {file_val}\n"
                f"{text}\n"
            )
            final_sections.append(section)

        return "\n".join(final_sections)

    except Exception as e:
        raise PipelineError("Failed to build context window") from e
