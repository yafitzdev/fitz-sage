# tests/unit/test_config_loader.py
"""Test config loading."""

from unittest.mock import patch

import pytest

from fitz_ai.config.loader import load_engine_config


def test_load_config_from_defaults():
    """Test loading config from default.yaml (isolated from user config)."""
    # Mock _load_user_config to return None, ensuring we only test defaults
    with patch("fitz_ai.config.loader._load_user_config", return_value=None):
        config = load_engine_config("fitz_krag")

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

    # Verify generation settings
    assert hasattr(config, "enable_citations")
    assert hasattr(config, "strict_grounding")
    assert hasattr(config, "top_addresses")


def test_config_required_field():
    """Test that collection field is required."""
    from pydantic import ValidationError

    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig

    # Missing required 'collection' field
    with pytest.raises(ValidationError):
        FitzKragConfig(chat="cohere", embedding="cohere")


def test_config_validation():
    """Test config validation (Pydantic)."""
    from pydantic import ValidationError

    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig

    # Invalid top_addresses (must be >= 1)
    with pytest.raises(ValidationError):
        FitzKragConfig(chat="cohere", embedding="cohere", collection="test", top_addresses=0)

    # Invalid max_context_tokens (must be >= 100)
    with pytest.raises(ValidationError):
        FitzKragConfig(chat="cohere", embedding="cohere", collection="test", max_context_tokens=10)


def test_config_none_for_disabled():
    """Test that None properly disables optional features."""
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig

    config = FitzKragConfig(
        chat="cohere",
        embedding="cohere",
        collection="test",
        rerank=None,  # Explicitly disabled
        vision=None,  # Explicitly disabled
    )

    assert config.rerank is None
    assert config.vision is None
