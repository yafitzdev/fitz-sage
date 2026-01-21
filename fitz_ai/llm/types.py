# fitz_ai/llm/types.py
"""
Typed models for LLM message handling.

Provides strong typing for the standard Fitz message format (OpenAI-compatible)
and common return structures from message transformations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

# =============================================================================
# Input Message Types (Standard Fitz/OpenAI-compatible format)
# =============================================================================


class Message(TypedDict, total=False):
    """
    Standard Fitz message format (OpenAI-compatible).

    Required fields:
        role: Message role ("system", "user", or "assistant")
        content: Message text content

    Optional fields (for vision):
        image_base64: Base64-encoded image data
        image_url: URL to an image
    """

    role: Literal["system", "user", "assistant"]
    content: str
    image_base64: str
    image_url: str


# Alias for list of messages (most common usage)
MessageList = list[Message]


# =============================================================================
# Transform Result Types
# =============================================================================


class OpenAIPayload(TypedDict, total=False):
    """OpenAI-compatible message payload."""

    messages: list[dict[str, Any]]


class AnthropicPayload(TypedDict, total=False):
    """Anthropic-specific payload with separate system prompt."""

    messages: list[dict[str, Any]]
    system: str


class GeminiPayload(TypedDict, total=False):
    """Gemini-specific payload with contents and system_instruction."""

    contents: list[dict[str, Any]]
    system_instruction: dict[str, Any]


# Union of all possible transform results
TransformResult = OpenAIPayload | AnthropicPayload | GeminiPayload | dict[str, Any]


# =============================================================================
# Content Part Types (for multi-part messages)
# =============================================================================


class TextPart(TypedDict):
    """Text content part."""

    type: Literal["text"]
    text: str


class ImageUrlPart(TypedDict):
    """Image URL content part (OpenAI/Cohere style)."""

    type: Literal["image_url"]
    image_url: dict[str, str]  # {"url": "..."}


class ImageSourcePart(TypedDict):
    """Image source content part (Anthropic style)."""

    type: Literal["image"]
    source: dict[str, str]  # {"type": "base64", "media_type": "...", "data": "..."}


ContentPart = TextPart | ImageUrlPart | ImageSourcePart


# =============================================================================
# Schema Types
# =============================================================================


@dataclass
class FieldInfo:
    """Metadata for a schema field."""

    type: Literal["string", "integer", "float", "boolean", "list", "object"]
    required: bool = False
    default: Any = None
    options: list[Any] | None = None
    description: str = ""
    example: Any = None


@dataclass
class SchemaDefinition:
    """Complete schema definition for a plugin."""

    fields: dict[str, FieldInfo] = field(default_factory=dict)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Message types
    "Message",
    "MessageList",
    # Transform results
    "TransformResult",
    "OpenAIPayload",
    "AnthropicPayload",
    "GeminiPayload",
    # Content parts
    "ContentPart",
    "TextPart",
    "ImageUrlPart",
    "ImageSourcePart",
    # Schema types
    "FieldInfo",
    "SchemaDefinition",
]
