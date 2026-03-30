# fitz_sage/llm/providers/__init__.py
"""
LLM provider implementations.

Each provider wraps an official SDK or HTTP client for a specific LLM service.
All providers are optional - install the SDK you need.
"""

from fitz_sage.llm.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    ModelTier,
    RerankProvider,
    RerankResult,
    StreamingChatProvider,
    VisionProvider,
)

__all__ = [
    # Protocols
    "ChatProvider",
    "StreamingChatProvider",
    "EmbeddingProvider",
    "RerankProvider",
    "VisionProvider",
    # Types
    "ModelTier",
    "RerankResult",
]

# Optional: Cohere (requires cohere package)
try:
    from fitz_sage.llm.providers.cohere import (  # noqa: F401
        CohereChat,
        CohereEmbedding,
        CohereRerank,
    )

    __all__.extend(["CohereChat", "CohereEmbedding", "CohereRerank"])
except ImportError:
    pass

# Optional: Ollama (requires ollama package)
try:
    from fitz_sage.llm.providers.ollama import (  # noqa: F401
        OllamaChat,
        OllamaEmbedding,
        OllamaVision,
    )

    __all__.extend(["OllamaChat", "OllamaEmbedding", "OllamaVision"])
except ImportError:
    pass

# Optional: OpenAI (requires openai package)
try:
    from fitz_sage.llm.providers.openai import (  # noqa: F401
        OpenAIChat,
        OpenAIEmbedding,
        OpenAIVision,
    )

    __all__.extend(["OpenAIChat", "OpenAIEmbedding", "OpenAIVision"])
except ImportError:
    pass

# Optional: Anthropic (requires anthropic package)
try:
    from fitz_sage.llm.providers.anthropic import (  # noqa: F401
        AnthropicChat,
        AnthropicVision,
    )

    __all__.extend(["AnthropicChat", "AnthropicVision"])
except ImportError:
    pass
