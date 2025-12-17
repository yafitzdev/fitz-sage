# fitz/llm/chat/plugins/cohere.py
"""
Cohere chat plugin using direct HTTP requests.

Uses Cohere v1 Chat API which expects:
- preamble: system message
- message: user message
- chat_history: conversation history
"""

from __future__ import annotations

import os
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


class CohereChatClient:
    """
    Cohere chat plugin using direct HTTP requests.

    Required:
        - COHERE_API_KEY environment variable OR api_key parameter

    Optional:
        - model: Chat model (default: command-r-plus)
        - temperature: Sampling temperature (default: 0.2)
    """

    plugin_name = "cohere"
    plugin_type = "chat"

    def __init__(
            self,
            api_key: str | None = None,
            model: str = "command-r-plus",
            temperature: float = 0.2,
            base_url: str = "https://api.cohere.ai/v1",
    ) -> None:
        if httpx is None:
            raise RuntimeError(
                "httpx is required for Cohere plugin. "
                "Install with: pip install httpx"
            )

        # Get API key
        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "COHERE_API_KEY is not set. "
                "Set it as an environment variable or pass api_key parameter."
            )
        self._api_key = key

        self.model = model
        self.temperature = temperature
        self.base_url = base_url

        # Create HTTP client
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,  # Longer timeout for LLM generation
        )

    def chat(self, messages: list[dict[str, Any]]) -> str:
        """
        Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
                     Supports: system, user, assistant roles

        Returns:
            The assistant's response text

        Raises:
            RuntimeError: If the API request fails
        """
        # Convert standard messages format to Cohere format
        # Cohere v1 uses:
        #   - preamble: system prompt
        #   - message: current user message
        #   - chat_history: previous messages

        preamble = ""
        chat_history = []
        current_message = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                preamble = content
            elif role == "user":
                # Last user message becomes the "message" parameter
                # Previous user messages go to chat_history
                if current_message:
                    chat_history.append({
                        "role": "USER",
                        "message": current_message
                    })
                current_message = content
            elif role == "assistant":
                chat_history.append({
                    "role": "CHATBOT",
                    "message": content
                })

        # Build request payload
        payload: dict[str, Any] = {
            "model": self.model,
            "message": current_message,
            "temperature": self.temperature,
        }

        if preamble:
            payload["preamble"] = preamble

        if chat_history:
            payload["chat_history"] = chat_history

        try:
            response = self._client.post("/chat", json=payload)
            response.raise_for_status()

            data = response.json()

            # Cohere response has "text" field
            if "text" in data:
                return data["text"]

            # Fallback for unexpected format
            return str(data)

        except httpx.HTTPStatusError as exc:
            error_detail = ""
            try:
                error_data = exc.response.json()
                error_detail = f": {error_data.get('message', '')}"
            except Exception:
                pass

            raise RuntimeError(
                f"Cohere API request failed with status {exc.response.status_code}{error_detail}"
            ) from exc

        except Exception as exc:
            raise RuntimeError(f"Failed to complete chat: {exc}") from exc

    def __del__(self):
        """Clean up HTTP client on deletion."""
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception:
                pass