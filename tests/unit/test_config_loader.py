# tests/unit/test_config_loader.py
"""Test config loading."""

import pytest
from fitz_ai.config.loader import load_engine_config


def test_load_config_from_defaults():
    """Test loading config from default.yaml."""
    config = load_engine_config("fitz_rag")

    # Verify it's a Pydantic model
    assert hasattr(config, "chat")
    assert hasattr(config, "embedding")
    assert hasattr(config, "vector_db")
    assert hasattr(config, "collection")

    # Verify string plugin specs
    assert isinstance(config.chat, str)
    assert isinstance(config.embedding, str)

    # Verify None for disabled features
    assert config.vision is None or isinstance(config.vision, str)

    # Verify flattened generation settings
    assert hasattr(config, "enable_citations")
    assert hasattr(config, "strict_grounding")
    assert hasattr(config, "max_chunks")


def test_config_required_field():
    """Test that collection field is required."""
    from fitz_ai.engines.fitz_rag.config.schema import FitzRagConfig
    from pydantic import ValidationError

    # Missing required 'collection' field
    with pytest.raises(ValidationError):
        FitzRagConfig(chat="cohere", embedding="cohere")


def test_config_validation():
    """Test config validation (Pydantic)."""
    from fitz_ai.engines.fitz_rag.config.schema import FitzRagConfig
    from pydantic import ValidationError

    # Invalid top_k (must be >= 1)
    with pytest.raises(ValidationError):
        FitzRagConfig(chat="cohere", embedding="cohere", collection="test", top_k=0)

    # Invalid chunk_size (must be >= 50)
    with pytest.raises(ValidationError):
        FitzRagConfig(
            chat="cohere", embedding="cohere", collection="test", chunk_size=10
        )


def test_config_none_for_disabled():
    """Test that None properly disables optional features."""
    from fitz_ai.engines.fitz_rag.config.schema import FitzRagConfig

    config = FitzRagConfig(
        chat="cohere",
        embedding="cohere",
        collection="test",
        rerank=None,  # Explicitly disabled
        vision=None,  # Explicitly disabled
    )

    assert config.rerank is None
    assert config.vision is None
