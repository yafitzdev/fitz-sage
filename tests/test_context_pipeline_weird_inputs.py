# tests/test_context_pipeline_weird_inputs.py

import pytest
from fitz_rag.context.pipeline import ContextPipeline

class Obj:
    def __init__(self, text, file):
        self.text = text
        self.metadata = {"file": file}

def test_context_pipeline_weird_inputs():
    chunks = [
        Obj("alpha", "x.txt"),
        Obj("beta", "y.txt"),
    ]

    ctx = ContextPipeline(max_chars=200).build(chunks)

    assert "alpha" in ctx
    assert "beta" in ctx
    assert "### Source: x.txt" in ctx
    assert "### Source: y.txt" in ctx
