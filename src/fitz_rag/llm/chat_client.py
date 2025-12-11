from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any
import os

from fitz_rag.exceptions.llm import LLMError, LLMResponseError

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


class ChatClient(Protocol):
    """Base interface for all LLM chat clients."""

    def chat(self, system_prompt: str, user_content: str) -> str:
        ...


@dataclass
class CohereChatClient:
    """
    Cohere Chat API wrapper.
    """

    api_key: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.3

    def __post_init__(self):
        if cohere is None:
            raise RuntimeError("Install cohere: `pip install cohere`")

        key = self.api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY not set")

        self.model = (
            self.model
            or os.getenv("COHERE_CHAT_MODEL")
            or "command-r7b-12-2024"
        )

        try:
            self._client = cohere.ClientV2(api_key=key)
        except Exception as e:
            raise LLMError("Failed to initialize Cohere Chat client") from e

    def _build_messages(self, system_prompt: str, user_content: str):
        msg = []
        if system_prompt:
            msg.append({"role": "system", "content": system_prompt})
        msg.append({"role": "user", "content": user_content})
        return msg

    def chat(self, system_prompt: str, user_content: str) -> str:
        """Executes a chat call with structured exception handling."""
        messages = self._build_messages(system_prompt, user_content)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        try:
            response = self._client.chat(**kwargs)
        except Exception as e:
            raise LLMError("Chat request failed") from e

        # Parse segments
        try:
            parts = []
            for seg in response.message.content:
                if getattr(seg, "type", None) == "text":
                    parts.append(seg.text)
                else:
                    text = getattr(seg, "text", None)
                    if isinstance(text, str):
                        parts.append(text)

            return "".join(parts).strip()
        except Exception as e:
            raise LLMResponseError("Malformed LLM response") from e


@dataclass
class DummyChatClient:
    prefix: str = "[DUMMY-ANSWER] "
    last_user_content: Optional[str] = None

    def chat(self, system_prompt: str, user_content: str) -> str:
        self.last_user_content = user_content
        return f"{self.prefix}OK"
