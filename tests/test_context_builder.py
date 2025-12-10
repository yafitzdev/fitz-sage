"""
Tests for fitz_rag.context.builder
"""

from __future__ import annotations

from fitz_rag.context.builder import (
    _normalize_text,
    dedupe_chunks,
    group_by_document,
    merge_adjacent_chunks,
    pack_context_window,
    build_context,
)


# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------
def test_normalize_text():
    text = "  Hello\tWorld \r\n"
    out = _normalize_text(text)
    assert out == "Hello World"


# ---------------------------------------------------------
# Deduplication
# ---------------------------------------------------------
def test_dedupe_chunks():
    chunks = [
        {"text": "A", "file": "f1"},
        {"text": "A ", "file": "f1"},  # normalized duplicate
        {"text": "B", "file": "f1"},
    ]

    out = dedupe_chunks(chunks)

    assert len(out) == 2
    assert out[0]["text"] == "A"
    assert out[1]["text"] == "B"


# ---------------------------------------------------------
# Grouping by document
# ---------------------------------------------------------
def test_group_by_document():
    chunks = [
        {"text": "A", "file": "doc1"},
        {"text": "B", "file": "doc1"},
        {"text": "C", "file": "doc2"},
    ]

    grouped = group_by_document(chunks)

    assert set(grouped.keys()) == {"doc1", "doc2"}
    assert len(grouped["doc1"]) == 2
    assert len(grouped["doc2"]) == 1


# ---------------------------------------------------------
# Merging adjacent chunks
# ---------------------------------------------------------
def test_merge_adjacent_chunks():
    chunks = [
        {"text": "A1", "file": "doc1"},
        {"text": "A2", "file": "doc1"},
        {"text": "B1", "file": "doc2"},
        {"text": "B2", "file": "doc2"},
    ]

    out = merge_adjacent_chunks(chunks)

    assert len(out) == 2
    assert "A1" in out[0]["text"]
    assert "A2" in out[0]["text"]
    assert "B1" in out[1]["text"]
    assert "B2" in out[1]["text"]


# ---------------------------------------------------------
# Packing window
# ---------------------------------------------------------
def test_pack_context_window():
    chunks = [
        {"text": "A" * 10, "file": "f1"},
        {"text": "B" * 10, "file": "f1"},
        {"text": "C" * 10, "file": "f1"},
    ]

    out = pack_context_window(chunks, max_chars=25)

    # A(10) + B(10) = 20 < 25 → pick 2 chunks
    assert len(out) == 2


# ---------------------------------------------------------
# Full context build
# ---------------------------------------------------------
def test_build_context():
    chunks = [
        {"text": "A1", "file": "doc1"},
        {"text": "A2", "file": "doc1"},
        {"text": "B1", "file": "doc2"},
    ]

    ctx = build_context(chunks, max_chars=1000)

    # Should contain 2 sections: doc1 + doc2
    assert "### Source: doc1" in ctx
    assert "### Source: doc2" in ctx

    # doc1 merged
    assert "A1" in ctx and "A2" in ctx

    # doc2 included
    assert "B1" in ctx
