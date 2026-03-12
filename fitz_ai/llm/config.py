# fitz_ai/llm/config.py
"""
Configuration parser for LLM providers.

Parses provider/model strings and instantiates correct provider + auth combinations.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from fitz_ai.llm.auth import ApiKeyAuth, AuthProvider, CompositeAuth, M2MAuth
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
    "lmstudio": None,  # No auth required (local)
    "enterprise": None,  # Auth configured explicitly via auth block
}

# LM Studio default models by tier
LMSTUDIO_CHAT_MODELS: dict[str, str] = {
    "smart": "qwen3-coder-30b",
    "balanced": "qwen3.5-9b",
    "fast": "qwen3.5-4b",
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


def _validate_enterprise_config(auth_config: dict[str, Any]) -> None:
    """Validate enterprise auth config with actionable error messages."""
    required = ["token_url", "client_id", "client_secret", "llm_api_key_env"]
    missing = [f for f in required if not auth_config.get(f)]

    if missing:
        raise ValueError(
            f"Enterprise auth config missing required fields: {', '.join(missing)}\n\n"
            f"Required configuration:\n"
            f"  auth:\n"
            f"    type: enterprise\n"
            f"    token_url: <OAuth2 token endpoint>\n"
            f"    client_id: ${{CLIENT_ID}}  # or literal value\n"
            f"    client_secret: ${{CLIENT_SECRET}}\n"
            f"    llm_api_key_env: <env var name for LLM API key>\n\n"
            f"Optional fields:\n"
            f"    scope: <OAuth2 scope>\n"
            f"    llm_api_key_header: X-Api-Key  # default\n"
            f"    client_cert_path: <mTLS cert path>\n"
            f"    client_key_path: <mTLS key path>\n"
            f"    client_key_password: ${{KEY_PASSWORD}}"
        )

    # Validate API key env var exists at startup (fail fast)
    api_key_env = auth_config["llm_api_key_env"]
    if not os.environ.get(api_key_env):
        raise ValueError(
            f"Enterprise auth requires environment variable {api_key_env} to be set.\n"
            f"This variable should contain the LLM API key for the underlying provider."
        )


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

    Config format for enterprise auth (M2M + API key):
        {
            "auth": {
                "type": "enterprise",
                "token_url": "https://auth.corp.com/oauth/token",
                "client_id": "${CLIENT_ID}",
                "client_secret": "${CLIENT_SECRET}",
                "scope": "optional-scope",  # optional
                "llm_api_key_env": "BMW_LLM_API_KEY",
                "llm_api_key_header": "X-Api-Key",  # optional, default X-Api-Key
                "client_cert_path": "/path/to/client.crt",  # optional, for mTLS
                "client_key_path": "/path/to/client.key",  # optional, for mTLS
                "client_key_password": "${KEY_PASSWORD}"  # optional, for encrypted key
            },
            "cert_path": "/etc/ssl/corp-ca.crt"  # optional, for CA verification
        }
    """
    config = config or {}
    auth_config = config.get("auth", {})

    # Check for enterprise auth (M2M + API key)
    if auth_config.get("type") == "enterprise":
        _validate_enterprise_config(auth_config)

        # Create M2MAuth for bearer token (Authorization header)
        m2m = M2MAuth(
            token_url=auth_config["token_url"],
            client_id=auth_config["client_id"],
            client_secret=auth_config["client_secret"],
            cert_path=config.get("cert_path"),
            scope=auth_config.get("scope"),
            client_cert_path=auth_config.get("client_cert_path"),
            client_key_path=auth_config.get("client_key_path"),
            client_key_password=auth_config.get("client_key_password"),
        )

        # Create ApiKeyAuth for LLM API key (X-Api-Key header by default)
        api_key_env = auth_config["llm_api_key_env"]
        header_name = auth_config.get("llm_api_key_header", "X-Api-Key")
        header_format: Literal["bearer", "x-api-key", "basic"] = (
            "x-api-key" if header_name.lower() == "x-api-key" else "bearer"
        )
        api_key = ApiKeyAuth(api_key_env, header_format=header_format)

        return CompositeAuth(m2m, api_key)

    # Check for M2M auth
    if auth_config.get("type") == "m2m":
        return M2MAuth(
            token_url=auth_config["token_url"],
            client_id=auth_config["client_id"],
            client_secret=auth_config["client_secret"],
            cert_path=config.get("cert_path"),
            scope=auth_config.get("scope"),
        )

    # Enterprise provider requires explicit auth config
    if provider == "enterprise":
        raise ValueError(
            "Enterprise provider requires an 'auth' block in config.\n"
            "Example:\n"
            "  chat_kwargs:\n"
            "    auth:\n"
            "      type: m2m\n"
            "      token_url: https://auth.corp.internal/oauth/token\n"
            "      client_id: ${CLIENT_ID}\n"
            "      client_secret: ${CLIENT_SECRET}\n"
            "    base_url: https://llm.corp.internal/v1\n"
            "    model: openai/gpt-4o"
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

    # Pass through models dict for tier-based model selection (Ollama, etc.)
    if "models" in config:
        kwargs["models"] = config["models"]

    # Pass through model for single model override (embeddings, etc.)
    if "model" in config:
        kwargs["model"] = config["model"]

    # Pass through context window override (e.g. num_ctx for Ollama)
    if "num_ctx" in config:
        kwargs["num_ctx"] = config["num_ctx"]

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

    if provider == "enterprise":
        from fitz_ai.llm.providers.enterprise import EnterpriseChat

        # Enterprise requires base_url and model
        base_url = kwargs.pop("base_url", None)
        model = kwargs.pop("model", None)
        if not base_url:
            raise ValueError(
                "Enterprise provider requires 'base_url' in config.\n"
                "Example:\n"
                "  chat_kwargs:\n"
                "    base_url: https://llm.corp.internal/v1\n"
                "    model: openai/gpt-4o"
            )
        if not model:
            raise ValueError(
                "Enterprise provider requires 'model' in config.\n"
                "Example:\n"
                "  chat_kwargs:\n"
                "    base_url: https://llm.corp.internal/v1\n"
                "    model: openai/gpt-4o"
            )
        return EnterpriseChat(auth, base_url=base_url, model=model, **kwargs)  # type: ignore[arg-type]

    elif provider == "cohere":
        from fitz_ai.llm.providers.cohere import CohereChat

        return CohereChat(auth, tier=tier, **kwargs)  # type: ignore[arg-type]

    elif provider == "openai" or provider == "azure_openai":
        from fitz_ai.llm.providers.openai import OpenAIChat

        return OpenAIChat(auth, tier=tier, **kwargs)  # type: ignore[arg-type]

    elif provider == "anthropic":
        from fitz_ai.llm.providers.anthropic import AnthropicChat

        return AnthropicChat(auth, tier=tier, **kwargs)  # type: ignore[arg-type]

    elif provider == "lmstudio":
        from fitz_ai.llm.providers.enterprise import EnterpriseChat

        base_url = kwargs.pop("base_url", "http://localhost:1234/v1")
        model = kwargs.pop("model", None)
        if not model:
            model = LMSTUDIO_CHAT_MODELS.get(tier)
        if not model:
            raise ValueError(
                "LM Studio requires a model name.\n"
                "Example:\n"
                "  chat: lmstudio/qwen3-coder-30b\n"
                "Or set model in chat_kwargs:\n"
                "  chat_kwargs:\n"
                "    model: qwen3-coder-30b"
            )
        return EnterpriseChat(None, base_url=base_url, model=model, **kwargs)

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

    if provider == "enterprise":
        from fitz_ai.llm.providers.enterprise import EnterpriseEmbedding

        # Enterprise requires base_url and model
        base_url = kwargs.pop("base_url", None)
        model = kwargs.pop("model", None)
        if not base_url:
            raise ValueError(
                "Enterprise provider requires 'base_url' in config.\n"
                "Example:\n"
                "  embedding_kwargs:\n"
                "    base_url: https://llm.corp.internal/v1\n"
                "    model: openai/text-embedding-3-small"
            )
        if not model:
            raise ValueError(
                "Enterprise provider requires 'model' in config.\n"
                "Example:\n"
                "  embedding_kwargs:\n"
                "    base_url: https://llm.corp.internal/v1\n"
                "    model: openai/text-embedding-3-small"
            )
        return EnterpriseEmbedding(auth, base_url=base_url, model=model, **kwargs)  # type: ignore[arg-type]

    elif provider == "cohere":
        from fitz_ai.llm.providers.cohere import CohereEmbedding

        # Cohere uses input_type parameter
        if "input_type" in config:
            kwargs["input_type"] = config["input_type"]

        return CohereEmbedding(auth, **kwargs)  # type: ignore[arg-type]

    elif provider == "openai" or provider == "azure_openai":
        from fitz_ai.llm.providers.openai import OpenAIEmbedding

        return OpenAIEmbedding(auth, **kwargs)  # type: ignore[arg-type]

    elif provider == "lmstudio":
        from fitz_ai.llm.providers.enterprise import EnterpriseEmbedding

        base_url = kwargs.pop("base_url", "http://localhost:1234/v1")
        model = kwargs.pop("model", None)
        if not model:
            raise ValueError(
                "LM Studio requires a model name.\n"
                "Example:\n"
                "  embedding: lmstudio/text-embedding-nomic-embed-text-v1.5\n"
                "Or set model in embedding_kwargs:\n"
                "  embedding_kwargs:\n"
                "    model: text-embedding-nomic-embed-text-v1.5"
            )
        return EnterpriseEmbedding(None, base_url=base_url, model=model, **kwargs)

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

    elif provider == "ollama":
        from fitz_ai.llm.providers.ollama import OllamaRerank

        return OllamaRerank(**kwargs)

    else:
        raise ValueError(f"Unknown rerank provider: {provider}. Supported: 'cohere', 'ollama'.")


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

    elif provider == "ollama":
        from fitz_ai.llm.providers.ollama import OllamaVision

        return OllamaVision(**kwargs)

    else:
        raise ValueError(
            f"Unknown vision provider: {provider}. Supported: 'openai', 'anthropic', 'ollama'"
        )


__all__ = [
    "parse_provider_string",
    "resolve_auth",
    "create_chat_provider",
    "create_embedding_provider",
    "create_rerank_provider",
    "create_vision_provider",
    "ENV_VAR_MAP",
]
