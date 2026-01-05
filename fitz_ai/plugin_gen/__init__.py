# fitz_ai/plugin_gen/__init__.py
"""
LLM-Powered Plugin Generator.

Generates fully working plugins using LLM with automatic validation
and retry on errors.

Usage:
    from fitz_ai.plugin_gen import PluginGenerator, PluginType

    generator = PluginGenerator()
    result = generator.generate(PluginType.LLM_CHAT, "anthropic")

    if result.success:
        print(f"Created plugin at: {result.path}")
    else:
        print(f"Failed: {result.errors}")
"""

from fitz_ai.plugin_gen.generator import PluginGenerator
from fitz_ai.plugin_gen.types import (
    GenerationResult,
    PluginType,
    ReviewDecision,
    ReviewResult,
    ValidationLevel,
    ValidationResult,
)

__all__ = [
    "PluginGenerator",
    "PluginType",
    "ReviewDecision",
    "ReviewResult",
    "ValidationLevel",
    "ValidationResult",
    "GenerationResult",
]
