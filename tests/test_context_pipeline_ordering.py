# tests/test_context_pipeline_ordering.py

import pytest
from rag.context.pipeline import ContextPipeline

def test_context_pipeline_preserves_document_order():
    chunks = [
        {"text": "A1", "file": "doc1"},
        {"text": "B1", "file": "doc2"},
        {"text": "C1", "file": "doc3"},
        {"text": "A2", "file": "doc1"},
    ]

    ctx = ContextPipeline(max_chars=500).build(chunks)

    # ordering must be doc1 → doc2 → doc3
    pos_doc1 = ctx.index("### Source: doc1")
    pos_doc2 = ctx.index("### Source: doc2")
    pos_doc3 = ctx.index("### Source: doc3")

    assert pos_doc1 < pos_doc2 < pos_doc3
