"""
fitz_rag.generation.rgs

Retrieval-Guided Synthesis (RGS) utilities for fitz_rag.

RGS is the "answer synthesis" layer of a RAG pipeline:
- takes a user query
- takes retrieved chunks (documents)
- builds a strict, retrieval-grounded prompt
- helps the caller run the LLM in a controlled way

This module is intentionally provider-agnostic:
- it does NOT call any specific LLM API
- it only builds prompts and wraps answers
- you decide how to send the prompts to your LLM adapter

Typical usage
-------------

    from fitz_rag.generation.rgs import RGS, RGSConfig

    rgs = RGS(config=RGSConfig())

    # retriever_result is a list of chunk objects or dicts
    prompt = rgs.build_prompt(query, retriever_result)

    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]

    # Example with your LLM adapter (pseudo-code):
    # raw_answer = llm.chat(messages)

    # Then wrap into structured answer if you like:
    # answer = rgs.build_answer(raw_answer, retriever_result)

You can integrate RGS into any pipeline step that already has:
- the user query (string)
- the retrieved chunks (list)
- an LLM adapter capable of chat/completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Protocol, Union


# ---------------------------------------------------------------------------
# Chunk abstraction
# ---------------------------------------------------------------------------

class SupportsRGSChunk(Protocol):
    """
    Minimal protocol for a chunk-like object to be used by RGS.

    Any object that has at least the following attributes will work:

        - .id: str          unique identifier for the chunk
        - .content: str     the text content of the chunk
        - .metadata: dict   arbitrary metadata (source file, page, etc.)

    You do NOT have to explicitly inherit from this protocol.
    It's only used for type checking / documentation.
    """
    id: str
    content: str
    metadata: Dict[str, Any]


ChunkInput = Union[SupportsRGSChunk, Mapping[str, Any]]


# ---------------------------------------------------------------------------
# Config & result models
# ---------------------------------------------------------------------------

@dataclass
class RGSConfig:
    """
    Configuration options for Retrieval-Guided Synthesis.
    """

    # Whether to ask the model to provide citations like [S1], [S2] in the answer
    enable_citations: bool = True

    # Whether to emphasize "do not hallucinate" behaviour in the instructions
    strict_grounding: bool = True

    # Optional style hint for the answer, e.g. "short", "detailed", "bullet points"
    answer_style: Optional[str] = None

    # Maximum number of chunks to include in the prompt (to avoid over-long context)
    max_chunks: Optional[int] = 8

    # Optional max length instruction for the model (does not enforce actual token limit)
    max_answer_chars: Optional[int] = None

    # Whether to include the original user query in the context block
    include_query_in_context: bool = True

    # Label used to refer to each chunk in the prompt (e.g. "S" -> [S1], [S2])
    source_label_prefix: str = "S"


@dataclass
class RGSSourceRef:
    """
    Reference to a source chunk used for answering.
    """

    source_id: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RGSAnswer:
    """
    Structured view of an LLM answer plus the sources used.

    Note: RGS does NOT parse citations from the answer text.
    It simply attaches the same chunks that were used in the prompt.
    """

    answer: str
    sources: List[RGSSourceRef]


@dataclass
class RGSPrompt:
    """
    Represents a system + user prompt pair that you can pass to your LLM adapter.
    """

    system: str
    user: str


# ---------------------------------------------------------------------------
# Core RGS implementation
# ---------------------------------------------------------------------------

class RGS:
    """
    Retrieval-Guided Synthesis helper.

    This class is stateless except for its configuration.
    """

    def __init__(self, config: Optional[RGSConfig] = None) -> None:
        self.config: RGSConfig = config or RGSConfig()

    # -------------------------------
    # Public API
    # -------------------------------

    def build_prompt(
        self,
        query: str,
        chunks: Sequence[ChunkInput],
    ) -> RGSPrompt:
        """
        Build a system+user prompt pair for the given query and retrieved chunks.

        Parameters
        ----------
        query:
            The user question / instruction.
        chunks:
            A sequence of chunk-like objects or dicts. Each must provide:
            - id
            - content
            - metadata (optional, can be empty)

        Returns
        -------
        RGSPrompt
            The system and user prompt texts.
        """
        limited_chunks = self._limit_chunks(chunks)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, limited_chunks)

        return RGSPrompt(system=system_prompt, user=user_prompt)

    def build_answer(
        self,
        raw_answer: str,
        chunks: Sequence[ChunkInput],
    ) -> RGSAnswer:
        """
        Wrap a raw LLM answer into a structured RGSAnswer object.

        This function doesn't do any NLP / parsing of the answer; it simply
        associates the answer with the chunks that were used to build the prompt.

        Parameters
        ----------
        raw_answer:
            The answer text returned by the LLM.
        chunks:
            The same chunk sequence you passed into build_prompt().

        Returns
        -------
        RGSAnswer
        """
        limited_chunks = self._limit_chunks(chunks)

        sources: List[RGSSourceRef] = []
        for idx, chunk in enumerate(limited_chunks, start=1):
            cid = self._get_chunk_id(chunk, idx)
            meta = self._get_chunk_metadata(chunk)
            sources.append(RGSSourceRef(source_id=cid, index=idx, metadata=meta))

        return RGSAnswer(answer=raw_answer, sources=sources)

    # -------------------------------
    # Internal helpers
    # -------------------------------

    def _limit_chunks(self, chunks: Sequence[ChunkInput]) -> List[ChunkInput]:
        if self.config.max_chunks is None:
            return list(chunks)
        return list(chunks[: self.config.max_chunks])

    def _build_system_prompt(self) -> str:
        """
        Compose the system instructions for the LLM.
        """
        parts: List[str] = [
            "You are a retrieval-grounded assistant.",
            "You must answer the user's question ONLY using the provided context snippets.",
        ]

        if self.config.strict_grounding:
            parts.append(
                "If the answer is not contained in the context, you MUST say "
                "\"I don't know based on the provided information.\" "
                "Do NOT invent facts or guess."
            )

        if self.config.enable_citations:
            prefix = self.config.source_label_prefix
            parts.append(
                f"When you use a context snippet, cite it using the notation "
                f"[{prefix}N], where N is the snippet number (e.g., [{prefix}1], [{prefix}2])."
            )

        if self.config.answer_style:
            parts.append(f"Preferred answer style: {self.config.answer_style}.")

        if self.config.max_answer_chars:
            parts.append(
                f"Try to keep your answer within approximately "
                f"{self.config.max_answer_chars} characters."
            )

        return "\n".join(parts)

    def _build_user_prompt(
        self,
        query: str,
        chunks: Sequence[ChunkInput],
    ) -> str:
        """
        Compose the user-facing part of the prompt, including the context.
        """
        if not chunks:
            # Degenerate case: no context
            return (
                "No context snippets are available.\n\n"
                f"User question:\n{query}\n\n"
                "Explain that you don't have enough information to answer."
            )

        context_lines: List[str] = []
        prefix = self.config.source_label_prefix

        for idx, chunk in enumerate(chunks, start=1):
            label = f"{prefix}{idx}"
            content = self._get_chunk_content(chunk)
            metadata = self._get_chunk_metadata(chunk)

            header = f"[{label}]"
            if metadata:
                # Keep metadata compact to avoid blowing up the prompt
                meta_str = self._format_metadata(metadata)
                header += f" (metadata: {meta_str})"

            context_lines.append(header)
            context_lines.append(content.strip())
            context_lines.append("")  # blank line between snippets

        context_block = "\n".join(context_lines).rstrip()

        user_parts: List[str] = [
            "You are given the following context snippets:",
            "",
            context_block,
            "",
        ]

        if self.config.include_query_in_context:
            user_parts.append("User question:")
            user_parts.append(query.strip())
        else:
            user_parts.append(
                "Now answer the user's question based ONLY on the above context."
            )

        return "\n".join(user_parts)

    @staticmethod
    def _get_chunk_id(chunk: ChunkInput, fallback_index: int) -> str:
        # Supports both object attributes and dict-like access
        if hasattr(chunk, "id"):
            cid = getattr(chunk, "id")
            if isinstance(cid, str):
                return cid

        if isinstance(chunk, Mapping) and "id" in chunk and isinstance(chunk["id"], str):
            return chunk["id"]

        # Fallback: synthetic ID
        return f"chunk_{fallback_index}"

    @staticmethod
    def _get_chunk_content(chunk: ChunkInput) -> str:
        if hasattr(chunk, "content"):
            value = getattr(chunk, "content")
            if isinstance(value, str):
                return value

        if isinstance(chunk, Mapping) and "content" in chunk and isinstance(chunk["content"], str):
            return chunk["content"]

        raise ValueError(
            "Chunk is missing 'content' field; expected an attribute or key 'content' of type str."
        )

    @staticmethod
    def _get_chunk_metadata(chunk: ChunkInput) -> Dict[str, Any]:
        if hasattr(chunk, "metadata"):
            value = getattr(chunk, "metadata")
            if isinstance(value, Mapping):
                return dict(value)

        if isinstance(chunk, Mapping) and "metadata" in chunk and isinstance(chunk["metadata"], Mapping):
            return dict(chunk["metadata"])

        return {}

    @staticmethod
    def _format_metadata(metadata: Mapping[str, Any], max_items: int = 3) -> str:
        """
        Render a compact metadata string like: 'file=foo.txt, page=3, score=0.87'.
        """
        items = list(metadata.items())[:max_items]
        parts: List[str] = []
        for key, value in items:
            parts.append(f"{key}={value!r}")
        if len(metadata) > max_items:
            parts.append("...")
        return ", ".join(parts)
