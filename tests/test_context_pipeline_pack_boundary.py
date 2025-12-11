from fitz_rag.context.pipeline import ContextPipeline


def test_context_pipeline_pack_boundary():
    chunks = [
        {"text": "A" * 50, "file": "doc1"},
        {"text": "B" * 50, "file": "doc1"},
        {"text": "C" * 50, "file": "doc2"},
    ]

    pipeline = ContextPipeline(max_chars=80)
    ctx = pipeline.build(chunks)

    # Because doc1 merged block is ~101 chars, pack logic includes it fully.
    # doc2 is excluded because adding it exceeds max_chars.

    assert "A" * 50 in ctx
    assert "B" * 50 in ctx  # merged blocks are kept intact
    assert "C" * 50 not in ctx  # doc2 excluded entirely

    # Only one source section should appear
    assert ctx.count("### Source:") == 1
