# tests/unit/test_context_pipeline_consolidated.py
"""Consolidated Context Pipeline tests - all context pipeline behavior scenarios."""

import pytest

from fitz_ai.engines.fitz_rag.pipeline.pipeline import ContextPipeline


# Parametrized test for context pipeline scenarios
@pytest.mark.parametrize(
    "scenario_name,chunks,max_chars,assertion_fn",
    [
        # Cross-file dedupe: no cross-file deduplication happens
        (
            "cross_file_dedupe",
            [
                {"content": "Hello WORLD", "file": "doc1"},
                {"content": "  hello   world ", "file": "doc2"},
                {"content": "HELLO WORLD ", "file": "doc3"},
            ],
            500,
            lambda out: (
                len(out) == 3
                and [c.doc_id for c in out] == ["unknown", "unknown", "unknown"]
                and [c.content for c in out] == ["Hello WORLD", "  hello   world ", "HELLO WORLD "]
            ),
        ),
        # Markdown integrity: pipeline emits non-empty context even with truncation
        (
            "markdown_integrity",
            [
                {"content": "A" * 500, "file": "doc1"},
                {"content": "B" * 500, "file": "doc2"},
            ],
            200,
            lambda out: out and any(c.content for c in out),
        ),
        # Ordering: preserves document order
        (
            "ordering",
            [
                {"content": "A1", "file": "doc1"},
                {"content": "B1", "file": "doc2"},
                {"content": "C1", "file": "doc3"},
                {"content": "A2", "file": "doc1"},
            ],
            500,
            lambda out: (
                out
                and "\n".join(c.content for c in out).find("A1")
                < "\n".join(c.content for c in out).find("B1")
                < "\n".join(c.content for c in out).find("C1")
            ),
        ),
        # Pack boundary: packing may drop later chunks when max_chars is small
        (
            "pack_boundary",
            [
                {"content": "A" * 50, "file": "doc1"},
                {"content": "B" * 50, "file": "doc1"},
                {"content": "C" * 50, "file": "doc2"},
            ],
            80,
            lambda out: out and "A" * 20 in "".join(c.content for c in out),
        ),
        # Unknown group: groups chunks with no file as "unknown"
        (
            "unknown_group",
            [
                {"content": "A", "metadata": {}},  # no file
                {"content": "B", "file": None},  # explicit None
                {"content": "C"},  # nothing at all
            ],
            200,
            lambda out: out and all(c.doc_id == "unknown" for c in out),
        ),
    ],
)
def test_context_pipeline_scenarios(scenario_name, chunks, max_chars, assertion_fn):
    """Test context pipeline scenarios using parametrization."""
    pipeline = ContextPipeline(max_chars=max_chars)
    out = pipeline.process(chunks)

    assert assertion_fn(out), f"Scenario '{scenario_name}' failed"


# Separate test for weird inputs (custom objects)
def test_context_pipeline_weird_inputs():
    """Test context pipeline with non-dict inputs."""

    class Obj:
        def __init__(self, text: str, file: str):
            self.text = text
            self.file = file

    chunks = [
        Obj("alpha", "x.txt"),
        Obj("beta", "y.txt"),
    ]

    out = ContextPipeline(max_chars=200).process(chunks)

    # Current pipeline converts unknown objects into an empty unknown chunk.
    assert len(out) == 1
    assert out[0].doc_id == "unknown"
    assert out[0].content == ""
