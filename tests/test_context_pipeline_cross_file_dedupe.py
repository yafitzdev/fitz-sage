# tests/test_context_pipeline_cross_file_dedupe.py

import pytest
from fitz_rag.context.pipeline import ContextPipeline

def test_context_pipeline_cross_file_dedupe():
    chunks = [
        {"text": "Hello WORLD", "file": "doc1"},
        {"text": "  hello   world ", "file": "doc2"},  # duplicate after normalization
        {"text": "HELLO WORLD ", "file": "doc3"},       # also duplicate
    ]

    ctx = ContextPipeline(max_chars=500).build(chunks)

    # Expect only ONE occurrence of text after dedupe
    assert ctx.count("Hello WORLD") == 1
    assert ctx.count("hello world") == 1 or ctx.count("Hello world") == 1
