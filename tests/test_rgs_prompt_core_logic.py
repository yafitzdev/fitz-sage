from __future__ import annotations

from fitz_rag.generation.rgs import RGS, RGSConfig


def test_rgs_prompt_structure():
    rgs = RGS(RGSConfig(max_chunks=2, enable_citations=True))

    chunks = [
        {"id": "1", "text": "Hello", "metadata": {"file": "doc1"}},
        {"id": "2", "text": "World", "metadata": {"file": "doc2"}},
        {"id": "3", "text": "Ignored", "metadata": {}},
    ]

    prompt = rgs.build_prompt("What?", chunks)

    # System prompt contains grounding & citation info
    assert "retrieval-grounded assistant" in prompt.system.lower()
    assert "[s1]" or "[S1]"  # depending on casing
    assert "only using the provided" in prompt.system.lower()

    user = prompt.user.lower()

    # Should contain exactly 2 snippets because max_chunks=2
    assert user.count("[s1]") == 1
    assert user.count("[s2]") == 1

    # Should embed the question
    assert "what?" in user
