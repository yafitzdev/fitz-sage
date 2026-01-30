# fitz_ai/llm/providers/__init__.py
"""
LLM provider implementations.

Each provider wraps an official SDK or HTTP client for a specific LLM service.
"""

from fitz_ai.llm.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    ModelTier,
    RerankProvider,
    RerankResult,
    StreamingChatProvider,
    VisionProvider,
)
from fitz_ai.llm.providers.cohere import CohereChat, CohereEmbedding, CohereRerank
from fitz_ai.llm.providers.ollama import OllamaChat, OllamaEmbedding

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
    # Cohere
    "CohereChat",
    "CohereEmbedding",
    "CohereRerank",
    # Ollama
    "OllamaChat",
    "OllamaEmbedding",
]

# Optional: OpenAI (requires openai package)
try:
    from fitz_ai.llm.providers.openai import OpenAIChat, OpenAIEmbedding, OpenAIVision

    __all__.extend(["OpenAIChat", "OpenAIEmbedding", "OpenAIVision"])
except ImportError:
    pass

# Optional: Anthropic (requires anthropic package)
try:
    from fitz_ai.llm.providers.anthropic import AnthropicChat, AnthropicVision

    __all__.extend(["AnthropicChat", "AnthropicVision"])
except ImportError:
    pass
