# fitz_ai/llm/factory.py
"""
Chat factory for per-task tier selection.

Enables different LLM tiers for different micro-tasks within a single module.
Tier assignments are developer decisions, not user-configurable.

Usage:
    from fitz_ai.llm.factory import ChatFactory, get_chat_factory

    # Create factory once
    chat_factory = get_chat_factory("cohere")

    # Use different tiers per task
    fast_chat = chat_factory("fast")       # Simple tasks
    balanced_chat = chat_factory("balanced")  # Complex tasks
    smart_chat = chat_factory("smart")     # User-facing responses

Tier Guidelines:
    - fast: Background tasks, enrichment, simple classification
    - balanced: SQL generation, structured output, bulk evaluation
    - smart: User-facing responses, complex reasoning
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from fitz_ai.llm.config import create_chat_provider
from fitz_ai.llm.providers.base import ChatProvider

ModelTier = Literal["fast", "balanced", "smart"]
ChatFactory = Callable[[ModelTier], ChatProvider]


def get_chat_factory(spec: str, config: dict[str, Any] | None = None) -> ChatFactory:
    """
    Create a factory that returns cached chat clients per tier.

    Args:
        spec: Provider spec (e.g., "cohere", "openai/gpt-4o").
        config: Optional config with auth/base_url settings.

    Returns:
        Factory function: (tier) -> ChatProvider

    Example:
        factory = get_chat_factory("cohere")
        chat = factory("fast")  # Returns cached fast-tier client
        chat.chat([{"role": "user", "content": "Hello"}])
    """
    # Cache clients per tier (lazy initialization)
    cache: dict[ModelTier, ChatProvider] = {}

    def factory(tier: ModelTier = "fast") -> ChatProvider:
        if tier not in cache:
            cache[tier] = create_chat_provider(spec, config, tier)
        return cache[tier]

    return factory


__all__ = [
    "ModelTier",
    "ChatFactory",
    "get_chat_factory",
]
