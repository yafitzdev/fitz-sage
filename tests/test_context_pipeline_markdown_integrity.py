# tests/test_context_pipeline_markdown_integrity.py

import pytest
from fitz_rag.context.pipeline import ContextPipeline

def test_context_pipeline_markdown_integrity():
    chunks = [
        {"text": "A" * 500, "file": "doc1"},
        {"text": "B" * 500, "file": "doc2"},
    ]

    ctx = ContextPipeline(max_chars=200).build(chunks)

    # Must contain valid markdown headers
    assert "### Source: doc1" in ctx
    assert "### Source: doc2" not in ctx  # second doc excluded by pack

    # No accidental duplicate headers or structural corruption
    assert "### ###" not in ctx
    assert ctx.strip().startswith("### Source: doc1")
