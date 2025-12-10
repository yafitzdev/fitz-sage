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


# ---------------------------------------------------------
# Internal helper
# ---------------------------------------------------------
def _normalize_text(text: str) -> str:
    """
    Normalize chunk text to reduce duplicates.
    - strip whitespace
    - remove dangerous spacing
    """
    return (
        text.replace("\r", "")
        .replace("\t", " ")
        .strip()
    )


# ---------------------------------------------------------
# 1) Deduplication
# ---------------------------------------------------------
def dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate chunks (same text).
    """
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


# ---------------------------------------------------------
# 2) Grouping by file / document
# ---------------------------------------------------------
def group_by_document(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Groups retrieved chunks by the 'file' metadata.
    """
    groups = {}

    for ch in chunks:
        f = ch.get("file", "unknown")
        groups.setdefault(f, []).append(ch)

    # Sort each group by whatever ordering RAG returned (usually relevance)
    return groups


# ---------------------------------------------------------
# 3) Merge adjacent chunks (simple concatenation)
# ---------------------------------------------------------
def merge_adjacent_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If multiple chunks come from the same doc and appear consecutively,
    combine them into one larger text block.

    This is intentionally simple — sophisticated merging can be added later.
    """

    if not chunks:
        return []

    merged = []
    buffer_text = chunks[0]["text"]
    buffer_meta = dict(chunks[0])

    for prev, curr in zip(chunks, chunks[1:]):
        same_doc = curr.get("file") == prev.get("file")

        if same_doc:
            # just append raw text
            buffer_text += "\n" + curr["text"]
        else:
            # flush buffer
            merged.append({
                "text": buffer_text,
                **{k: v for k, v in buffer_meta.items() if k != "text"},
            })
            buffer_text = curr["text"]
            buffer_meta = curr

    # last flush
    merged.append({
        "text": buffer_text,
        **{k: v for k, v in buffer_meta.items() if k != "text"},
    })

    return merged


# ---------------------------------------------------------
# 4) Pack context window to a max character size
# ---------------------------------------------------------
def pack_context_window(
    chunks: List[Dict[str, Any]],
    max_chars: int = 6000,
) -> List[Dict[str, Any]]:
    """
    Keep adding merged chunks until max_chars is reached.
    """
    packed = []
    total = 0

    for ch in chunks:
        txt = ch["text"]
        if total + len(txt) > max_chars:
            break
        packed.append(ch)
        total += len(txt)

    return packed


# ---------------------------------------------------------
# 5) Main API
# ---------------------------------------------------------
def build_context(
    retrieved_chunks: List[Dict[str, Any]],
    max_chars: int = 6000,
) -> str:
    """
    Full context-building pipeline.

    Steps:
        1. normalize + dedupe
        2. group by document
        3. merge adjacent (within each doc group)
        4. flatten
        5. pack into the final max_chars window
        6. format nicely for the prompt
    """

    # 1) Cleanup & dedupe
    clean_chunks = dedupe_chunks(retrieved_chunks)

    # 2) Group them
    groups = group_by_document(clean_chunks)

    merged_per_doc = []

    # 3) Merge inside each document group
    for file, chs in groups.items():
        merged = merge_adjacent_chunks(chs)
        merged_per_doc.extend(merged)

    # 4) Sort merged list by relevance order (already preserved)
    merged_list = merged_per_doc

    # 5) Pack the window
    packed = pack_context_window(merged_list, max_chars=max_chars)

    # 6) Build final formatted context
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
