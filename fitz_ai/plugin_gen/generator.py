# fitz_ai/plugin_gen/generator.py
"""
Plugin generator using LLM.

Orchestrates the generate → validate → retry loop to produce
working plugins from natural language descriptions.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Optional

from fitz_ai.plugin_gen.context import (
    build_generation_prompt,
    build_retry_prompt,
    extract_code_from_response,
    load_example_plugin,
)
from fitz_ai.plugin_gen.library_context import (
    get_library_context_for_query,
)
from fitz_ai.plugin_gen.types import GenerationResult, PluginType
from fitz_ai.plugin_gen.validators import PluginValidator, format_validation_error

logger = logging.getLogger(__name__)

# Maximum retry attempts
MAX_RETRIES = 3


class PluginGenerator:
    """
    Generates plugins using LLM with automatic validation and retry.

    The generator:
    1. Builds a prompt with examples and schema info
    2. Calls the LLM to generate code
    3. Validates the generated code at multiple levels
    4. If validation fails, feeds error back to LLM and retries
    5. Saves successful plugins to the correct directory

    Example:
        generator = PluginGenerator()
        result = generator.generate(PluginType.LLM_CHAT, "anthropic")

        if result.success:
            print(f"Created plugin at: {result.path}")
        else:
            print(f"Failed: {result.first_error}")
    """

    def __init__(
        self,
        chat_plugin: str = "openai",
        tier: str = "smart",
    ):
        """
        Initialize the generator.

        Args:
            chat_plugin: Chat LLM plugin to use for generation
            tier: Model tier to use ("smart", "fast", "balanced")
        """
        self.chat_plugin = chat_plugin
        self.tier = tier
        self.validator = PluginValidator()
        self._llm_client = None

    def _get_llm(self):
        """Lazy load the LLM client."""
        if self._llm_client is None:
            from fitz_ai.llm.registry import get_llm_plugin

            self._llm_client = get_llm_plugin(
                plugin_name=self.chat_plugin,
                plugin_type="chat",
                tier=self.tier,
            )
        return self._llm_client

    def generate(
        self,
        plugin_type: PluginType,
        description: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> GenerationResult:
        """
        Generate a plugin from a description.

        Args:
            plugin_type: Type of plugin to generate
            description: User's description (e.g., "anthropic", "sentence chunker")
            progress_callback: Optional callback for progress updates

        Returns:
            GenerationResult with success status, path, and validation results
        """
        result = GenerationResult(
            success=False,
            plugin_type=plugin_type,
        )

        def progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        # Load example plugin
        progress("Loading example plugins...")
        example_code = load_example_plugin(plugin_type)

        # Check for external library mentions
        library_context = None
        if plugin_type.is_python:
            progress("Checking for package dependencies...")
            lookup_result = get_library_context_for_query(description)

            # Handle lookup failures (epistemic honesty)
            if not lookup_result.success:
                progress(f"Cannot proceed: {lookup_result.error}")
                result.errors.append(lookup_result.error)
                return result

            library_context = lookup_result.context
            if library_context:
                progress(f"Fetched docs for '{library_context.name}' package")

        # Build initial prompt
        prompt = build_generation_prompt(plugin_type, description, example_code, library_context)

        code: Optional[str] = None
        last_error: Optional[str] = None

        for attempt in range(1, MAX_RETRIES + 1):
            result.attempts = attempt
            progress(f"Generating plugin (attempt {attempt}/{MAX_RETRIES})...")

            try:
                # Generate code
                if attempt == 1:
                    response = self._call_llm(prompt)
                else:
                    # Retry with error feedback
                    retry_prompt = build_retry_prompt(
                        plugin_type, description, code or "", last_error or ""
                    )
                    response = self._call_llm(retry_prompt)

                # Extract code from response
                code = extract_code_from_response(response, plugin_type)
                result.code = code

                # Validate
                progress("Validating generated code...")
                validations = self.validator.validate(
                    code,
                    plugin_type,
                    uses_external_library=library_context is not None,
                )
                result.validations = validations

                # Check if all validations passed
                all_passed = all(v.success for v in validations)

                if all_passed:
                    progress("All validations passed!")
                    result.success = True

                    # Extract plugin name from generated code
                    plugin_name = self._extract_plugin_name(code, plugin_type)
                    result.plugin_name = plugin_name

                    # Save plugin
                    progress("Saving plugin...")
                    result.path = self._save_plugin(code, plugin_type, plugin_name)

                    return result

                # Find first failure for retry
                for v in validations:
                    if not v.success:
                        last_error = format_validation_error(v, code)
                        progress(f"Validation failed: {v.level.value} - {v.error}")
                        break

            except Exception as e:
                last_error = str(e)
                result.errors.append(last_error)
                progress(f"Error: {e}")

        # All retries exhausted
        result.errors.append(f"Failed after {MAX_RETRIES} attempts")
        return result

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt."""
        llm = self._get_llm()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert plugin developer. Generate complete, working "
                    "plugins that pass all validation checks. Output ONLY the code, "
                    "no explanations or markdown formatting unless explicitly requested."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        response = llm.chat(messages)
        return response

    def _extract_plugin_name(self, code: str, plugin_type: PluginType) -> str:
        """Extract the plugin name from generated code."""
        # Try plugin_name field (YAML and Python)
        match = re.search(
            r'plugin_name[:\s]*[=:]?\s*["\']?([a-z][a-z0-9_]*)["\']?', code, re.IGNORECASE
        )
        if match:
            return match.group(1).lower()

        # Try name field (YAML)
        match = re.search(
            r'^name[:\s]+["\']?([a-z][a-z0-9_]*)["\']?', code, re.MULTILINE | re.IGNORECASE
        )
        if match:
            return match.group(1).lower()

        return "custom_plugin"

    def _save_plugin(
        self,
        code: str,
        plugin_type: PluginType,
        plugin_name: str,
    ) -> Path:
        """Save the generated plugin to the correct location."""
        # Get output directory and ensure it exists
        output_dir = plugin_type.get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", plugin_name)
        filename = f"{safe_name}{plugin_type.file_extension}"
        output_path = output_dir / filename

        # Write file
        output_path.write_text(code, encoding="utf-8")

        logger.info(f"Saved plugin to: {output_path}")
        return output_path


__all__ = ["PluginGenerator", "MAX_RETRIES"]
