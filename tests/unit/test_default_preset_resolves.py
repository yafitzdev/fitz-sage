# tests/test_default_preset_resolves.py
"""Test that default config loads and resolves correctly."""

from fitz_ai.config import load_engine_config
from fitz_ai.engines.fitz_rag.config import FitzRagConfig


def test_default_preset_resolves_to_runtime_config():
    """Test that default config loads and has all required fields."""
    cfg = load_engine_config("fitz_rag")

    assert isinstance(cfg, FitzRagConfig)

    # Required plugins (string specs)
    assert cfg.chat
    assert cfg.embedding
    assert cfg.vector_db
    assert cfg.collection
