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

from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    pass

ModelTier = Literal["fast", "balanced", "smart"]
ChatFactory = Callable[[ModelTier], Any]


def get_chat_factory(plugin_name: str, **kwargs: Any) -> ChatFactory:
    """
    Create a factory that returns cached chat clients per tier.

    Args:
        plugin_name: Name of the chat plugin (e.g., "cohere", "openai").
        **kwargs: Additional kwargs passed to plugin initialization.

    Returns:
        Factory function: (tier) -> chat_client

    Example:
        factory = get_chat_factory("cohere")
        chat = factory("fast")  # Returns cached fast-tier client
        chat.chat([{"role": "user", "content": "Hello"}])
    """
    from fitz_ai.llm.registry import get_llm_plugin

    # Cache clients per tier (lazy initialization)
    cache: dict[ModelTier, Any] = {}

    def factory(tier: ModelTier = "fast") -> Any:
        if tier not in cache:
            cache[tier] = get_llm_plugin(
                plugin_type="chat",
                plugin_name=plugin_name,
                tier=tier,
                **kwargs,
            )
        return cache[tier]

    return factory


__all__ = [
    "ModelTier",
    "ChatFactory",
    "get_chat_factory",
]
