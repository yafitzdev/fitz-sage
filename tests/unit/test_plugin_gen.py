# tests/test_plugin_gen.py
"""
Tests for the plugin generator.
"""

from __future__ import annotations

import pytest

from fitz_ai.plugin_gen.types import (
    GenerationResult,
    PluginType,
    ValidationLevel,
    ValidationResult,
)
from fitz_ai.plugin_gen.validators import PluginValidator, format_validation_error

# =============================================================================
# PluginType Tests
# =============================================================================


class TestPluginType:
    """Tests for PluginType enum."""

    def test_is_yaml(self):
        """Test is_yaml property."""
        assert PluginType.LLM_CHAT.is_yaml is True
        assert PluginType.LLM_EMBEDDING.is_yaml is True
        assert PluginType.LLM_RERANK.is_yaml is True
        assert PluginType.VECTOR_DB.is_yaml is True
        assert PluginType.RETRIEVAL.is_yaml is True
        assert PluginType.CHUNKER.is_yaml is False
        assert PluginType.READER.is_yaml is False
        assert PluginType.CONSTRAINT.is_yaml is False

    def test_is_python(self):
        """Test is_python property."""
        assert PluginType.CHUNKER.is_python is True
        assert PluginType.READER.is_python is True
        assert PluginType.CONSTRAINT.is_python is True
        assert PluginType.LLM_CHAT.is_python is False

    def test_file_extension(self):
        """Test file_extension property."""
        assert PluginType.LLM_CHAT.file_extension == ".yaml"
        assert PluginType.CHUNKER.file_extension == ".py"

    def test_display_name(self):
        """Test display_name property."""
        assert PluginType.LLM_CHAT.display_name == "LLM Chat Provider"
        assert PluginType.CHUNKER.display_name == "Document Chunker"

    def test_description(self):
        """Test description property."""
        assert "chat llm" in PluginType.LLM_CHAT.description.lower()
        assert "chunking" in PluginType.CHUNKER.description.lower()


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_passed(self):
        """Test passed factory method."""
        result = ValidationResult.passed(ValidationLevel.SYNTAX)
        assert result.success is True
        assert result.level == ValidationLevel.SYNTAX
        assert result.error is None

    def test_failed(self):
        """Test failed factory method."""
        result = ValidationResult.failed(
            ValidationLevel.SCHEMA,
            "Missing required field",
            line=10,
        )
        assert result.success is False
        assert result.level == ValidationLevel.SCHEMA
        assert result.error == "Missing required field"
        assert result.details["line"] == 10


# =============================================================================
# GenerationResult Tests
# =============================================================================


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_first_error_from_validations(self):
        """Test first_error property with validation errors."""
        result = GenerationResult(
            success=False,
            plugin_type=PluginType.LLM_CHAT,
            validations=[
                ValidationResult.passed(ValidationLevel.SYNTAX),
                ValidationResult.failed(ValidationLevel.SCHEMA, "Bad schema"),
            ],
        )
        assert result.first_error == "Bad schema"

    def test_first_error_from_errors(self):
        """Test first_error property with errors list."""
        result = GenerationResult(
            success=False,
            plugin_type=PluginType.LLM_CHAT,
            errors=["LLM error"],
        )
        assert result.first_error == "LLM error"


# =============================================================================
# Validator Tests
# =============================================================================


class TestPluginValidator:
    """Tests for PluginValidator."""

    @pytest.fixture
    def validator(self):
        return PluginValidator()

    # YAML Validation Tests
    # -------------------------------------------------------------------------

    def test_yaml_syntax_valid(self, validator):
        """Test valid YAML syntax."""
        code = """
plugin_name: test
plugin_type: chat
version: "1.0"
provider:
  name: test
  base_url: https://api.test.com
"""
        results = validator._validate_yaml(code, PluginType.LLM_CHAT, None)
        assert results[0].level == ValidationLevel.SYNTAX
        assert results[0].success is True

    def test_yaml_syntax_invalid(self, validator):
        """Test invalid YAML syntax."""
        code = """
plugin_name: test
  bad_indent: value
"""
        results = validator._validate_yaml(code, PluginType.LLM_CHAT, None)
        assert results[0].level == ValidationLevel.SYNTAX
        assert results[0].success is False
        assert "syntax" in results[0].error.lower()

    def test_yaml_not_mapping(self, validator):
        """Test YAML that isn't a mapping."""
        code = "- item1\n- item2"
        results = validator._validate_yaml(code, PluginType.LLM_CHAT, None)
        assert results[0].success is False
        assert "mapping" in results[0].error.lower()

    # Python Validation Tests
    # -------------------------------------------------------------------------

    def test_python_syntax_valid(self, validator):
        """Test valid Python syntax."""
        code = """
from dataclasses import dataclass

@dataclass
class TestPlugin:
    plugin_name: str = "test"
"""
        results = validator._validate_python(code, PluginType.CHUNKER, None)
        assert results[0].level == ValidationLevel.SYNTAX
        assert results[0].success is True

    def test_python_syntax_invalid(self, validator):
        """Test invalid Python syntax."""
        code = """
def test(
    missing_paren
"""
        results = validator._validate_python(code, PluginType.CHUNKER, None)
        assert results[0].level == ValidationLevel.SYNTAX
        assert results[0].success is False


# =============================================================================
# Error Formatting Tests
# =============================================================================


class TestFormatValidationError:
    """Tests for format_validation_error function."""

    def test_format_basic_error(self):
        """Test basic error formatting."""
        result = ValidationResult.failed(ValidationLevel.SCHEMA, "Missing field: name")
        formatted = format_validation_error(result, "plugin_name: test")
        assert "CHECK FAILED: schema" in formatted
        assert "Missing field: name" in formatted

    def test_format_with_line_number(self):
        """Test formatting with line number."""
        result = ValidationResult.failed(
            ValidationLevel.SCHEMA,
            "Invalid value",
            line=5,
        )
        code = "\n".join([f"line {i}" for i in range(10)])
        formatted = format_validation_error(result, code)
        assert "RELEVANT CODE:" in formatted

    def test_format_with_suggestion(self):
        """Test formatting generates suggestions."""
        result = ValidationResult.failed(
            ValidationLevel.SCHEMA,
            "Missing required field: auth.env_vars",
        )
        formatted = format_validation_error(result, "")
        assert "SUGGESTION:" in formatted


# =============================================================================
# Context Builder Tests
# =============================================================================


class TestContextBuilder:
    """Tests for context builder functions."""

    def test_load_example_plugin(self):
        """Test loading example plugins."""
        from fitz_ai.plugin_gen.context import load_example_plugin

        # Should load an example for LLM_CHAT
        example = load_example_plugin(PluginType.LLM_CHAT)
        if example:  # May not exist in test environment
            assert "plugin_name" in example

    def test_build_generation_prompt(self):
        """Test building generation prompt."""
        from fitz_ai.plugin_gen.context import build_generation_prompt

        prompt = build_generation_prompt(PluginType.LLM_CHAT, "anthropic")
        assert "anthropic" in prompt.lower()
        assert "LLM Chat Provider" in prompt
        assert "Required fields" in prompt or "## Requirements" in prompt

    def test_build_retry_prompt(self):
        """Test building retry prompt."""
        from fitz_ai.plugin_gen.context import build_retry_prompt

        prompt = build_retry_prompt(
            PluginType.LLM_CHAT,
            "anthropic",
            "old code here",
            "Missing field: auth",
        )
        assert "failed validation" in prompt.lower()
        assert "old code here" in prompt
        assert "Missing field: auth" in prompt

    def test_extract_code_from_response(self):
        """Test code extraction from LLM response."""
        from fitz_ai.plugin_gen.context import extract_code_from_response

        # With markdown code block
        response = """Here's the plugin:

```yaml
plugin_name: test
plugin_type: chat
```

That should work!"""
        code = extract_code_from_response(response, PluginType.LLM_CHAT)
        assert "plugin_name: test" in code
        assert "Here's the plugin" not in code

        # Without code block
        response = "plugin_name: test\nplugin_type: chat"
        code = extract_code_from_response(response, PluginType.LLM_CHAT)
        assert "plugin_name: test" in code


# =============================================================================
# Test Inputs Tests
# =============================================================================


class TestTestInputs:
    """Tests for test input functions."""

    def test_get_test_input_chunker(self):
        """Test getting chunker test input."""
        from fitz_ai.plugin_gen.test_inputs import get_test_input

        inputs = get_test_input(PluginType.CHUNKER)
        assert inputs is not None
        assert "text" in inputs
        assert "meta" in inputs
        assert len(inputs["text"]) > 0

    def test_get_test_input_constraint(self):
        """Test getting constraint test input."""
        from fitz_ai.plugin_gen.test_inputs import get_test_input

        inputs = get_test_input(PluginType.CONSTRAINT)
        assert inputs is not None
        assert "query" in inputs
        assert "chunks" in inputs

    def test_get_test_input_reader(self):
        """Test getting reader test input (should be None)."""
        from fitz_ai.plugin_gen.test_inputs import get_test_input

        inputs = get_test_input(PluginType.READER)
        assert inputs is None  # Reader requires actual files

    def test_get_expected_behavior(self):
        """Test getting expected behavior descriptions."""
        from fitz_ai.plugin_gen.test_inputs import get_expected_behavior

        behavior = get_expected_behavior(PluginType.CHUNKER)
        assert "chunk_text" in behavior
        assert "List[Chunk]" in behavior or "list" in behavior.lower()


# =============================================================================
# Integration Tests (marked as slow)
# =============================================================================


@pytest.mark.slow
class TestPluginGeneratorIntegration:
    """Integration tests for the plugin generator."""

    @pytest.mark.skip(reason="Requires LLM API key")
    def test_generate_chat_plugin(self):
        """Test generating a chat plugin end-to-end."""
        from fitz_ai.plugin_gen import PluginGenerator, PluginType

        generator = PluginGenerator()
        result = generator.generate(PluginType.LLM_CHAT, "test provider")

        # We can't guarantee success without mocking, but we can check structure
        assert result.attempts >= 1
        assert len(result.validations) > 0
