# fitz_ai/llm/providers/anthropic.py
"""
Anthropic provider wrappers using the official SDK.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from fitz_ai.llm.auth import AuthProvider
from fitz_ai.llm.providers.base import ModelTier

logger = logging.getLogger(__name__)

# Default models by tier
CHAT_MODELS: dict[ModelTier, str] = {
    "smart": "claude-sonnet-4-20250514",
    "balanced": "claude-haiku-4-20250514",
    "fast": "claude-haiku-4-20250514",
}

VISION_MODEL = "claude-sonnet-4-20250514"


def _extract_system_message(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Extract system message from messages list.

    Anthropic requires system message as separate parameter.

    Returns:
        Tuple of (system_message, remaining_messages)
    """
    system_msg = None
    filtered = []

    for msg in messages:
        if msg.get("role") == "system":
            system_msg = msg.get("content", "")
        else:
            filtered.append(msg)

    return system_msg, filtered


class AnthropicChat:
    """
    Anthropic chat provider using the official SDK.

    Args:
        auth: Authentication provider.
        model: Model name override.
        tier: Model tier (smart, balanced, fast).
        **kwargs: Additional default kwargs for chat calls.
    """

    def __init__(
        self,
        auth: AuthProvider,
        model: str | None = None,
        tier: ModelTier = "smart",
        models: dict[ModelTier, str] | None = None,
        **kwargs: Any,
    ) -> None:
        import anthropic

        headers = auth.get_headers()
        api_key = headers.get("Authorization", "").replace("Bearer ", "")
        if not api_key:
            api_key = headers.get("X-Api-Key", "")

        request_kwargs = auth.get_request_kwargs()

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if "verify" in request_kwargs:
            import httpx

            client_kwargs["http_client"] = httpx.Client(verify=request_kwargs["verify"])

        self._client = anthropic.Anthropic(**client_kwargs)
        # Use provided models dict, falling back to defaults
        tier_models = models or CHAT_MODELS
        self._model = model or tier_models.get(tier) or CHAT_MODELS[tier]
        self._defaults = kwargs

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Generate a chat completion."""
        params = {**self._defaults, **kwargs}

        system_msg, filtered_messages = _extract_system_message(messages)

        call_kwargs: dict[str, Any] = {
            "model": params.pop("model", self._model),
            "messages": filtered_messages,
            "max_tokens": params.pop("max_tokens", 4096),
        }
        if system_msg:
            call_kwargs["system"] = system_msg

        call_kwargs.update(params)

        response = self._client.messages.create(**call_kwargs)

        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
        return ""

    def chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        """Generate a streaming chat completion."""
        params = {**self._defaults, **kwargs}

        system_msg, filtered_messages = _extract_system_message(messages)

        call_kwargs: dict[str, Any] = {
            "model": params.pop("model", self._model),
            "messages": filtered_messages,
            "max_tokens": params.pop("max_tokens", 4096),
        }
        if system_msg:
            call_kwargs["system"] = system_msg

        call_kwargs.update(params)

        with self._client.messages.stream(**call_kwargs) as stream:
            for text in stream.text_stream:
                yield text


class AnthropicVision:
    """
    Anthropic vision provider using the official SDK.

    Args:
        auth: Authentication provider.
        model: Model name override.
        **kwargs: Additional default kwargs.
    """

    def __init__(
        self,
        auth: AuthProvider,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        import anthropic

        headers = auth.get_headers()
        api_key = headers.get("Authorization", "").replace("Bearer ", "")
        if not api_key:
            api_key = headers.get("X-Api-Key", "")

        request_kwargs = auth.get_request_kwargs()

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if "verify" in request_kwargs:
            import httpx

            client_kwargs["http_client"] = httpx.Client(verify=request_kwargs["verify"])

        self._client = anthropic.Anthropic(**client_kwargs)
        self._model = model or VISION_MODEL
        self._defaults = kwargs

    def describe_image(self, image_base64: str, prompt: str | None = None) -> str:
        """Describe an image using the vision model."""
        actual_prompt = prompt or (
            "Describe this figure/chart/diagram in detail. Include any data values, "
            "labels, axes, trends, and key insights visible in the image."
        )

        # Detect image type from base64 header
        media_type = "image/png"
        if image_base64.startswith("/9j/"):
            media_type = "image/jpeg"
        elif image_base64.startswith("iVBOR"):
            media_type = "image/png"
        elif image_base64.startswith("R0lGOD"):
            media_type = "image/gif"
        elif image_base64.startswith("UklGR"):
            media_type = "image/webp"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64,
                        },
                    },
                    {"type": "text", "text": actual_prompt},
                ],
            }
        ]

        params = {**self._defaults}
        response = self._client.messages.create(
            model=params.pop("model", self._model),
            messages=messages,
            max_tokens=params.pop("max_tokens", 4096),
            **params,
        )

        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
        return ""


__all__ = [
    "AnthropicChat",
    "AnthropicVision",
    "CHAT_MODELS",
    "VISION_MODEL",
]
