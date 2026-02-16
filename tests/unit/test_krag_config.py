# tests/unit/test_krag_config.py
"""Tests for FitzKragConfig schema and defaults."""


import pytest
import yaml

from fitz_ai.engines.fitz_krag.config import FitzKragConfig, get_default_config_path


class TestFitzKragConfig:
    def test_minimal_config(self):
        config = FitzKragConfig(collection="test")
        assert config.collection == "test"
        assert config.chat == "ollama"
        assert config.embedding == "ollama"

    def test_defaults(self):
        config = FitzKragConfig(collection="test")
        assert config.code_languages == ["python", "typescript", "java", "go"]
        assert config.summary_batch_size == 15
        assert config.top_addresses == 10
        assert config.top_read == 5
        assert config.keyword_weight == 0.4
        assert config.semantic_weight == 0.6
        assert config.fallback_to_chunks is True
        assert config.enable_citations is True
        assert config.strict_grounding is True
        assert config.max_context_tokens == 8000

    def test_custom_values(self):
        config = FitzKragConfig(
            collection="my_project",
            chat="anthropic/claude-sonnet-4",
            embedding="openai/text-embedding-3-small",
            top_addresses=20,
            keyword_weight=0.3,
            semantic_weight=0.7,
        )
        assert config.chat == "anthropic/claude-sonnet-4"
        assert config.top_addresses == 20
        assert config.keyword_weight == 0.3

    def test_collection_required(self):
        with pytest.raises(Exception):
            FitzKragConfig()  # type: ignore[call-arg]

    def test_validation_top_addresses(self):
        with pytest.raises(Exception):
            FitzKragConfig(collection="test", top_addresses=0)

    def test_validation_weights(self):
        with pytest.raises(Exception):
            FitzKragConfig(collection="test", keyword_weight=1.5)

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            FitzKragConfig(collection="test", nonexistent_field=True)


class TestDefaultYaml:
    def test_default_config_path_exists(self):
        path = get_default_config_path()
        assert path.exists()
        assert path.name == "default.yaml"

    def test_default_yaml_loads(self):
        path = get_default_config_path()
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        assert "fitz_krag" in raw
        assert raw["fitz_krag"]["chat"] == "ollama"
        assert raw["fitz_krag"]["collection"] == "default"

    def test_default_yaml_creates_valid_config(self):
        path = get_default_config_path()
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        config = FitzKragConfig(**raw["fitz_krag"])
        assert config.collection == "default"
