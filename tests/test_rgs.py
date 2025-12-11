import pytest

from fitz_rag.generation.rgs import (
    RGS,
    RGSConfig,
    RGSPrompt,
    RGSAnswer,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def build_fake_chunks(n=3):
    return [
        {
            "id": f"chunk_{i}",
            "text": f"Content of chunk {i}",
            "metadata": {"file": f"doc_{i}.txt", "page": i},
        }
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_rgs_prompt_structure_basic():
    rgs = RGS()
    chunks = build_fake_chunks(2)
    query = "What is the answer?"

    prompt = rgs.build_prompt(query, chunks)
    assert isinstance(prompt, RGSPrompt)

    assert "retrieval-grounded assistant" in prompt.system
    assert "ONLY using the provided context" in prompt.system

    # RGS citations always appear in user block
    assert "[S1]" in prompt.user
    assert "[S2]" in prompt.user

    assert "User question:" in prompt.user
    assert query in prompt.user

    assert "Content of chunk 1" in prompt.user
    assert "Content of chunk 2" in prompt.user


def test_rgs_respects_max_chunks():
    cfg = RGSConfig(max_chunks=1)
    rgs = RGS(cfg)
    chunks = build_fake_chunks(3)

    prompt = rgs.build_prompt("Test query", chunks)

    assert "[S1]" in prompt.user
    assert "[S2]" not in prompt.user
    assert "[S3]" not in prompt.user


def test_rgs_metadata_formatting():
    rgs = RGS()
    chunks = build_fake_chunks(1)

    prompt = rgs.build_prompt("Query", chunks)

    assert "(metadata:" in prompt.user
    assert "file='doc_1.txt'" in prompt.user
    assert "page=1" in prompt.user


def test_rgs_answer_wrapper():
    chunks = build_fake_chunks(2)
    rgs = RGS()
    raw_answer = "This is the final answer."

    wrapped = rgs.build_answer(raw_answer, chunks)

    assert isinstance(wrapped, RGSAnswer)
    assert wrapped.answer == raw_answer
    assert len(wrapped.sources) == 2

    assert wrapped.sources[0].metadata["file"] == "doc_1.txt"
    assert wrapped.sources[1].metadata["file"] == "doc_2.txt"


def test_rgs_no_chunks_behavior():
    rgs = RGS()
    prompt = rgs.build_prompt("What is this?", chunks=[])

    assert "No context snippets are available" in prompt.user
    assert "What is this?" in prompt.user

    # Updated for new message wording
    assert "cannot answer based on missing context" in prompt.user.lower()


def test_rgs_disable_citations():
    cfg = RGSConfig(enable_citations=False)
    rgs = RGS(cfg)
    chunks = build_fake_chunks(1)

    prompt = rgs.build_prompt("Query", chunks)

    assert "cite" not in prompt.system.lower()
    # Labels still appear
    assert "[S1]" in prompt.user


def test_rgs_strict_grounding():
    cfg = RGSConfig(strict_grounding=True)
    rgs = RGS(cfg)
    chunks = build_fake_chunks(1)

    prompt = rgs.build_prompt("Query", chunks)

    assert "MUST say \"I don't know" in prompt.system
    assert "Do NOT invent facts" in prompt.system
