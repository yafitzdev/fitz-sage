# tests/unit/test_rgs_consolidated.py
"""Consolidated RGS tests - all RGS configuration and behavior scenarios."""

import pytest
from fitz_ai.engines.fitz_rag.generation.retrieval_guided.synthesis import (
    RGS,
    RGSConfig,
)


# Parametrized test for RGS configuration scenarios
@pytest.mark.parametrize(
    "scenario_name,config,chunks,query,assertion_fn",
    [
        # Chunk ID fallback: generates chunk IDs when missing
        (
            "chunk_id_fallback",
            RGSConfig(),
            [
                {"content": "alpha", "metadata": {}},
                {"content": "beta", "metadata": {}},
            ],
            "ok",
            lambda ans, prompt: (
                ans.sources[0].source_id == "chunk_1"
                and ans.sources[1].source_id == "chunk_2"
            ),
        ),
        # Max chunks limit: respects max_chunks setting
        (
            "chunk_limit",
            RGSConfig(max_chunks=2),
            [
                {"id": "1", "content": "A", "metadata": {}},
                {"id": "2", "content": "B", "metadata": {}},
                {"id": "3", "content": "C", "metadata": {}},
            ],
            "Q?",
            lambda ans, prompt: (
                "A" in prompt.user and "B" in prompt.user and "C" not in prompt.user
            ),
        ),
        # Exclude query: query not in context when include_query_in_context=False
        (
            "exclude_query",
            RGSConfig(include_query_in_context=False),
            [{"id": "1", "content": "hello world", "metadata": {}}],
            "my question",
            lambda ans, prompt: (
                "my question" not in prompt.user
                and "Answer the question using ONLY the context above." in prompt.user
            ),
        ),
        # No citations: citation instruction removed when enable_citations=False
        (
            "no_citations",
            RGSConfig(enable_citations=False),
            [
                {"id": "1", "content": "alpha", "metadata": {"file": "doc1"}},
                {"id": "2", "content": "beta", "metadata": {"file": "doc2"}},
            ],
            "What?",
            lambda ans, prompt: (
                "Use citations" not in prompt.system
                and "alpha" in prompt.user
                and "beta" in prompt.user
            ),
        ),
        # Metadata format: content and source labels in prompt
        (
            "metadata_format",
            RGSConfig(),
            [
                {
                    "id": "1",
                    "content": "alpha",
                    "metadata": {"file": "doc1", "a": 1, "b": 2, "c": 3},
                },
            ],
            "Q?",
            lambda ans, prompt: (
                "alpha" in prompt.user and "[S1]" in prompt.user and "Q?" in prompt.user
            ),
        ),
        # Strict grounding: instruction present when strict_grounding=True
        (
            "strict_grounding",
            RGSConfig(strict_grounding=True),
            [{"id": "1", "content": "alpha", "metadata": {}}],
            "Q?",
            lambda ans, prompt: (
                "I don't know based on the provided information." in prompt.system
            ),
        ),
    ],
)
def test_rgs_scenarios(scenario_name, config, chunks, query, assertion_fn):
    """Test RGS configuration scenarios using parametrization."""
    rgs = RGS(config)

    # Build prompt/answer depending on what we're testing
    if scenario_name == "chunk_id_fallback":
        ans = rgs.build_answer(query, chunks)
        prompt = None
    else:
        prompt = rgs.build_prompt(query, chunks)
        ans = None

    # Run scenario-specific assertion
    assert assertion_fn(ans, prompt), f"Scenario '{scenario_name}' failed"


# Separate test for metadata handling
def test_rgs_metadata_truncation():
    """Test that RGS handles chunks with metadata."""
    rgs = RGS(RGSConfig())

    chunk = {
        "id": "x",
        "content": "hello",
        "metadata": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
    }

    prompt = rgs.build_prompt("q", [chunk])

    # Content should be in the prompt
    assert "hello" in prompt.user
    # Query should be present
    assert "q" in prompt.user


# Separate test for max chunks limit (different assertion style)
def test_rgs_max_chunks_limit_detailed():
    """Test max_chunks=1 with detailed assertions."""
    rgs = RGS(RGSConfig(max_chunks=1))

    chunks = [
        {"id": "1", "content": "alpha", "metadata": {}},
        {"id": "2", "content": "beta", "metadata": {}},
    ]

    prompt = rgs.build_prompt("?", chunks)

    assert "alpha" in prompt.user
    assert "beta" not in prompt.user


# Test for prompt core logic
def test_rgs_prompt_structure():
    """Test RGS prompt structure and core logic."""
    rgs = RGS(RGSConfig(max_chunks=2, enable_citations=True))

    chunks = [
        {"id": "1", "content": "Hello", "metadata": {"file": "doc1"}},
        {"id": "2", "content": "World", "metadata": {"file": "doc2"}},
        {"id": "3", "content": "Ignored", "metadata": {}},
    ]

    prompt = rgs.build_prompt("What?", chunks)

    assert "You are a retrieval-grounded assistant." in prompt.system
    assert "You are given the following context snippets:" in prompt.user
    assert "[S1]" in prompt.user
    assert "[S2]" in prompt.user
    assert "Hello" in prompt.user
    assert "World" in prompt.user
    assert "Ignored" not in prompt.user


# Test for prompt slots customization
def test_rgs_prompt_slots_override_defaults():
    """Test that prompt config can override default prompts."""
    from fitz_ai.engines.fitz_rag.generation.prompting import PromptConfig

    cfg = RGSConfig(
        prompt_config=PromptConfig(
            system_base="SYSTEM BASE OVERRIDE",
            user_instructions="Do the thing in bullet points.",
        )
    )
    rgs = RGS(cfg)

    chunks = [{"id": "1", "content": "alpha", "metadata": {"file": "doc1"}}]
    prompt = rgs.build_prompt("Q?", chunks)

    assert "SYSTEM BASE OVERRIDE" in prompt.system
    assert "You are a retrieval-grounded assistant." not in prompt.system

    assert "Do the thing in bullet points." in prompt.user
