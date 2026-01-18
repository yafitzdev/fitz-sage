# fitz_ai/llm/transforms.py
"""
Message transformation strategies for different LLM provider formats.

Each provider has its own message format. These transforms convert from
the standard Fitz format (OpenAI-compatible) to provider-specific formats.

Standard Fitz format:
[
    {"role": "system", "content": "You are helpful..."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
]
"""

from __future__ import annotations

from typing import Any, Protocol

from fitz_ai.llm.types import Message, TransformResult


class MessageTransformer(Protocol):
    """Protocol for message transformation."""

    def transform(self, messages: list[Message]) -> TransformResult:
        """Transform messages into provider-specific format.

        Returns a dict that will be merged into the request payload.
        """
        ...


# =============================================================================
# OpenAI-compatible format
# =============================================================================


class OpenAIChatTransform:
    """
    OpenAI-compatible message format.

    Most APIs use this format - messages are passed directly as an array.
    Used by: OpenAI, Azure OpenAI, Ollama, many OpenAI-compatible APIs.

    Output:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            ...
        ]
    }
    """

    def transform(self, messages: list[Message]) -> TransformResult:
        return {"messages": messages}


# =============================================================================
# Cohere v2 format (current API)
# =============================================================================


class CohereChatTransform:
    """
    Cohere v2 Chat API format.

    Cohere v2 API uses OpenAI-compatible message format with messages array.
    System messages use role "system", user messages use "user",
    assistant messages use "assistant".

    Documentation: https://docs.cohere.com/reference/chat

    Output:
    {
        "messages": [
            {"role": "system", "content": "System prompt here"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant response"}
        ]
    }
    """

    def transform(self, messages: list[Message]) -> TransformResult:
        # Cohere v2 API accepts OpenAI-compatible format directly
        return {"messages": messages}


# =============================================================================
# Anthropic format
# =============================================================================


class AnthropicChatTransform:
    """
    Anthropic Messages API format.

    Splits messages into:
    - system: A single string (not in messages array)
    - messages: User/assistant turns only (must alternate)

    Output:
    {
        "system": "System prompt here",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"}
        ]
    }
    """

    def transform(self, messages: list[Message]) -> TransformResult:
        system_content = ""
        user_messages: list[dict[str, str]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_content = content
            elif role in ("user", "assistant"):
                user_messages.append({"role": role, "content": content})

        result: TransformResult = {"messages": user_messages}

        if system_content:
            result["system"] = system_content

        return result


# =============================================================================
# Google Gemini format
# =============================================================================


class GeminiChatTransform:
    """
    Google Gemini API format.

    Uses 'contents' with 'parts' structure:
    - system_instruction: System message (optional)
    - contents: Array of user/model turns with parts

    Output:
    {
        "system_instruction": {"parts": [{"text": "System prompt"}]},
        "contents": [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi!"}]}
        ]
    }
    """

    # Role mapping from standard to Gemini
    ROLE_MAP = {
        "user": "user",
        "assistant": "model",
    }

    def transform(self, messages: list[Message]) -> TransformResult:
        system_instruction: dict[str, Any] | None = None
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
            elif role in self.ROLE_MAP:
                contents.append({"role": self.ROLE_MAP[role], "parts": [{"text": content}]})

        result: TransformResult = {"contents": contents}

        if system_instruction:
            result["system_instruction"] = system_instruction

        return result


# =============================================================================
# Ollama format
# =============================================================================


class OllamaChatTransform:
    """
    Ollama Chat API format.

    Very similar to OpenAI, but uses /api/chat endpoint.
    Messages are passed directly with standard roles.

    Output:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """

    def transform(self, messages: list[Message]) -> TransformResult:
        # Ollama accepts OpenAI-compatible format
        return {"messages": messages}


# =============================================================================
# Vision Transforms (handle image + text input)
# =============================================================================


class CohereVisionTransform:
    """
    Cohere Command A Vision API format.

    Cohere uses OpenAI-compatible format with "image_url" content type.
    Model: command-a-vision-07-2025

    Input messages should have format:
    [
        {"role": "user", "content": "Describe this image", "image_base64": "..."}
    ]

    Output:
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                ]
            }
        ]
    }
    """

    def transform(self, messages: list[Message]) -> TransformResult:
        transformed: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            image_base64 = msg.get("image_base64")
            image_url = msg.get("image_url")

            if image_base64 or image_url:
                # Vision message with image (OpenAI-compatible format)
                content_parts: list[dict[str, Any]] = [{"type": "text", "text": content}]
                if image_base64:
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        }
                    )
                elif image_url:
                    content_parts.append({"type": "image_url", "image_url": {"url": image_url}})
                transformed.append({"role": role, "content": content_parts})
            else:
                transformed.append({"role": role, "content": content})

        return {"messages": transformed}


class OpenAIVisionTransform:
    """
    OpenAI Vision API format (GPT-4o, GPT-4o-mini).

    Images are passed as content array with type "image_url" containing
    base64-encoded data or URL.

    Input messages should have format:
    [
        {"role": "user", "content": "Describe this image", "image_base64": "..."}
    ]

    Output:
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                ]
            }
        ]
    }
    """

    def transform(self, messages: list[Message]) -> TransformResult:
        transformed: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            image_base64 = msg.get("image_base64")
            image_url = msg.get("image_url")

            if image_base64 or image_url:
                # Vision message with image
                content_parts: list[dict[str, Any]] = [{"type": "text", "text": content}]
                if image_base64:
                    # Detect image type from base64 header or default to png
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        }
                    )
                elif image_url:
                    content_parts.append({"type": "image_url", "image_url": {"url": image_url}})
                transformed.append({"role": role, "content": content_parts})
            else:
                # Regular text message
                transformed.append({"role": role, "content": content})

        return {"messages": transformed}


class AnthropicVisionTransform:
    """
    Anthropic Vision API format (Claude 3).

    Images are passed as content array with type "image" containing
    base64-encoded data.

    Input messages should have format:
    [
        {"role": "user", "content": "Describe this image", "image_base64": "..."}
    ]

    Output:
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
                    {"type": "text", "text": "Describe this image"}
                ]
            }
        ]
    }
    """

    def transform(self, messages: list[Message]) -> TransformResult:
        system_content = ""
        user_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            image_base64 = msg.get("image_base64")

            if role == "system":
                system_content = content
            elif role in ("user", "assistant"):
                if image_base64 and role == "user":
                    # Vision message with image
                    content_parts: list[dict[str, Any]] = [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64,
                            },
                        },
                        {"type": "text", "text": content},
                    ]
                    user_messages.append({"role": role, "content": content_parts})
                else:
                    user_messages.append({"role": role, "content": content})

        result: TransformResult = {"messages": user_messages}
        if system_content:
            result["system"] = system_content

        return result


class OllamaVisionTransform:
    """
    Ollama Vision API format (LLaVA, llama3.2-vision).

    Images are passed as a separate "images" array with base64 data.

    Input messages should have format:
    [
        {"role": "user", "content": "Describe this image", "image_base64": "..."}
    ]

    Output:
    {
        "messages": [
            {"role": "user", "content": "Describe this image", "images": ["base64..."]}
        ]
    }
    """

    def transform(self, messages: list[Message]) -> TransformResult:
        transformed: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            image_base64 = msg.get("image_base64")

            if image_base64:
                # Ollama expects images as array of base64 strings
                transformed.append({"role": role, "content": content, "images": [image_base64]})
            else:
                transformed.append({"role": role, "content": content})

        return {"messages": transformed}


# =============================================================================
# Transform Registry
# =============================================================================


TRANSFORM_REGISTRY: dict[str, type[MessageTransformer]] = {
    # Chat transforms
    "openai_chat": OpenAIChatTransform,
    "cohere_chat": CohereChatTransform,
    "anthropic_chat": AnthropicChatTransform,
    "gemini_chat": GeminiChatTransform,
    "ollama_chat": OllamaChatTransform,
    # Vision transforms
    "openai_vision": OpenAIVisionTransform,
    "cohere_vision": CohereVisionTransform,
    "anthropic_vision": AnthropicVisionTransform,
    "ollama_vision": OllamaVisionTransform,
}


def get_transformer(name: str) -> MessageTransformer:
    """Get a message transformer by name.

    Args:
        name: Transform name (e.g., "openai_chat", "cohere_chat")

    Returns:
        Instantiated transformer

    Raises:
        ValueError: If transform name is not recognized
    """
    if name not in TRANSFORM_REGISTRY:
        available = sorted(TRANSFORM_REGISTRY.keys())
        raise ValueError(f"Unknown message transform: {name!r}. Available: {available}")

    return TRANSFORM_REGISTRY[name]()
