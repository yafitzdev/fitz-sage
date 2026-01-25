# tests/test_default_config_loads_and_validates_minimally.py
"""
Test that the default config can be loaded and validated.
"""

from unittest.mock import patch

from fitz_ai.config import load_engine_config
from fitz_ai.engines.fitz_rag.config import FitzRagConfig


def test_default_config_loads_and_validates_base_schema():
    """
    Verifies the CURRENT architectural contract:

    - default.yaml can be loaded
    - resolved runtime config validates against FitzRagConfig
    - all required fields are present
    """
    # Mock _load_user_config to avoid user config interference
    with patch("fitz_ai.config.loader._load_user_config", return_value=None):
        cfg = load_engine_config("fitz_rag")

    assert isinstance(cfg, FitzRagConfig)

    # Core plugins (string specs)
    assert cfg.chat
    assert cfg.embedding
    assert cfg.vector_db

    # Collection is required
    assert cfg.collection
