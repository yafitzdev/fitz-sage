# fitz_ai/llm/config.py
"""
Configuration parser for LLM providers.

Parses provider/model strings and instantiates correct provider + auth combinations.
"""

from __future__ import annotations

import os
from typing import Any

from fitz_ai.llm.auth import ApiKeyAuth, AuthProvider, M2MAuth
from fitz_ai.llm.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    ModelTier,
    RerankProvider,
    VisionProvider,
)

# Provider name → environment variable mapping
ENV_VAR_MAP: dict[str, str | None] = {
    "cohere": "COHERE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "ollama": None,  # No auth required
}

# Provider name → default header format
HEADER_FORMAT_MAP: dict[str, str] = {
    "cohere": "bearer",
    "openai": "bearer",
    "anthropic": "x-api-key",
    "azure_openai": "bearer",
}


def parse_provider_string(spec: str) -> tuple[str, str | None]:
    """
    Parse a provider/model string into provider name and model.

    Args:
        spec: Provider spec like "cohere" or "cohere/command-a-03-2025"

    Returns:
        Tuple of (provider_name, model_name or None)

    Examples:
        >>> parse_provider_string("cohere")
        ('cohere', None)
        >>> parse_provider_string("cohere/command-a-03-2025")
        ('cohere', 'command-a-03-2025')
        >>> parse_provider_string("openai/gpt-4o")
        ('openai', 'gpt-4o')
    """
    if "/" in spec:
        provider, model = spec.split("/", 1)
        return provider.strip(), model.strip()
    return spec.strip(), None


def resolve_auth(provider: str, config: dict[str, Any] | None = None) -> AuthProvider | None:
    """
    Resolve authentication for a provider.

    Args:
        provider: Provider name (cohere, openai, etc.)
        config: Optional config dict with auth settings

    Returns:
        AuthProvider instance, or None for providers that don't need auth (ollama)

    Config format for simple API key:
        {} or None - uses default env var for provider

    Config format for M2M OAuth2:
        {
            "auth": {
                "type": "m2m",
                "token_url": "https://auth.corp.com/oauth/token",
                "client_id": "${CLIENT_ID}",
                "client_secret": "${CLIENT_SECRET}",
                "scope": "optional-scope"  # optional
            },
            "cert_path": "/etc/ssl/corp-ca.crt"  # optional
        }
    """
    config = config or {}
    auth_config = config.get("auth", {})

    # Check for M2M auth
    if auth_config.get("type") == "m2m":
        return M2MAuth(
            token_url=auth_config["token_url"],
            client_id=auth_config["client_id"],
            client_secret=auth_config["client_secret"],
            cert_path=config.get("cert_path"),
            scope=auth_config.get("scope"),
        )

    # Default: API key auth
    env_var = ENV_VAR_MAP.get(provider)
    if env_var is None:
        # Provider doesn't need auth (e.g., ollama)
        return None

    header_format = HEADER_FORMAT_MAP.get(provider, "bearer")
    return ApiKeyAuth(env_var, header_format=header_format)  # type: ignore[arg-type]


def _get_provider_kwargs(config: dict[str, Any] | None) -> dict[str, Any]:
    """Extract provider kwargs from config."""
    if not config:
        return {}

    kwargs: dict[str, Any] = {}

    # Pass through base_url if specified
    if "base_url" in config:
        kwargs["base_url"] = config["base_url"]

    return kwargs


def create_chat_provider(
    spec: str,
    config: dict[str, Any] | None = None,
    tier: ModelTier = "smart",
) -> ChatProvider:
    """
    Create a chat provider from a spec string.

    Args:
        spec: Provider spec like "cohere" or "cohere/command-a-03-2025"
        config: Optional config dict with auth/base_url settings
        tier: Model tier (smart, balanced, fast)

    Returns:
        ChatProvider instance

    Examples:
        >>> chat = create_chat_provider("cohere")
        >>> chat = create_chat_provider("cohere/command-a-03-2025")
        >>> chat = create_chat_provider("openai/gpt-4o", {"base_url": "https://proxy.example.com"})
    """
    provider, model = parse_provider_string(spec)
    auth = resolve_auth(provider, config)
    kwargs = _get_provider_kwargs(config)

    if model:
        kwargs["model"] = model

    if provider == "cohere":
        from fitz_ai.llm.providers.cohere import CohereChat

        return CohereChat(auth, tier=tier, **kwargs)  # type: ignore[arg-type]

    elif provider == "openai" or provider == "azure_openai":
        from fitz_ai.llm.providers.openai import OpenAIChat

        return OpenAIChat(auth, tier=tier, **kwargs)  # type: ignore[arg-type]

    elif provider == "anthropic":
        from fitz_ai.llm.providers.anthropic import AnthropicChat

        return AnthropicChat(auth, tier=tier, **kwargs)  # type: ignore[arg-type]

    elif provider == "ollama":
        from fitz_ai.llm.providers.ollama import OllamaChat

        return OllamaChat(tier=tier, **kwargs)

    else:
        raise ValueError(f"Unknown chat provider: {provider}")


def create_embedding_provider(
    spec: str,
    config: dict[str, Any] | None = None,
) -> EmbeddingProvider:
    """
    Create an embedding provider from a spec string.

    Args:
        spec: Provider spec like "cohere" or "cohere/embed-multilingual-v3.0"
        config: Optional config dict with auth/base_url/dimensions settings

    Returns:
        EmbeddingProvider instance
    """
    provider, model = parse_provider_string(spec)
    auth = resolve_auth(provider, config)
    kwargs = _get_provider_kwargs(config)

    if model:
        kwargs["model"] = model

    # Pass through dimensions if specified
    config = config or {}
    if "dimensions" in config:
        kwargs["dimensions"] = config["dimensions"]

    if provider == "cohere":
        from fitz_ai.llm.providers.cohere import CohereEmbedding

        # Cohere uses input_type parameter
        if "input_type" in config:
            kwargs["input_type"] = config["input_type"]

        return CohereEmbedding(auth, **kwargs)  # type: ignore[arg-type]

    elif provider == "openai" or provider == "azure_openai":
        from fitz_ai.llm.providers.openai import OpenAIEmbedding

        return OpenAIEmbedding(auth, **kwargs)  # type: ignore[arg-type]

    elif provider == "ollama":
        from fitz_ai.llm.providers.ollama import OllamaEmbedding

        return OllamaEmbedding(**kwargs)

    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def create_rerank_provider(
    spec: str | None,
    config: dict[str, Any] | None = None,
) -> RerankProvider | None:
    """
    Create a rerank provider from a spec string.

    Args:
        spec: Provider spec like "cohere" or "cohere/rerank-v3.5", or None
        config: Optional config dict with auth settings

    Returns:
        RerankProvider instance, or None if spec is None
    """
    if spec is None:
        return None

    provider, model = parse_provider_string(spec)
    auth = resolve_auth(provider, config)
    kwargs = _get_provider_kwargs(config)

    if model:
        kwargs["model"] = model

    if provider == "cohere":
        from fitz_ai.llm.providers.cohere import CohereRerank

        return CohereRerank(auth, **kwargs)  # type: ignore[arg-type]

    else:
        raise ValueError(f"Unknown rerank provider: {provider}. Only 'cohere' is supported.")


def create_vision_provider(
    spec: str | None,
    config: dict[str, Any] | None = None,
) -> VisionProvider | None:
    """
    Create a vision provider from a spec string.

    Args:
        spec: Provider spec like "openai/gpt-4o" or "anthropic/claude-sonnet-4", or None
        config: Optional config dict with auth/base_url settings

    Returns:
        VisionProvider instance, or None if spec is None
    """
    if spec is None:
        return None

    provider, model = parse_provider_string(spec)
    auth = resolve_auth(provider, config)
    kwargs = _get_provider_kwargs(config)

    if model:
        kwargs["model"] = model

    if provider == "openai" or provider == "azure_openai":
        from fitz_ai.llm.providers.openai import OpenAIVision

        return OpenAIVision(auth, **kwargs)  # type: ignore[arg-type]

    elif provider == "anthropic":
        from fitz_ai.llm.providers.anthropic import AnthropicVision

        return AnthropicVision(auth, **kwargs)  # type: ignore[arg-type]

    else:
        raise ValueError(f"Unknown vision provider: {provider}. Supported: 'openai', 'anthropic'")


__all__ = [
    "parse_provider_string",
    "resolve_auth",
    "create_chat_provider",
    "create_embedding_provider",
    "create_rerank_provider",
    "create_vision_provider",
    "ENV_VAR_MAP",
]
