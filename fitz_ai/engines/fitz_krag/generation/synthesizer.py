# fitz_ai/engines/fitz_krag/generation/synthesizer.py
"""
Code synthesizer — generates grounded answers from code context.

Adapts the RGS pattern for address-based provenance. Answers include
[S1], [S2] citation markers mapping to file:line source blocks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fitz_ai.core import Answer, GenerationError, Provenance
from fitz_ai.core.answer_mode import AnswerMode
from fitz_ai.engines.fitz_krag.types import ReadResult
from fitz_ai.governance.instructions import get_mode_instruction

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_ai.llm.providers.base import ChatProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_GROUNDED = """\
You are a code-aware assistant. Answer questions using ONLY the provided source \
code and documentation context. Cite sources using [S1], [S2], etc. markers that \
correspond to the numbered source blocks.

Rules:
- Only use information from the provided context
- Cite specific sources with [S1], [S2] markers
- If the context doesn't contain enough information, say so
- Reference specific files and line numbers when helpful
- Be precise about what the code does, not what it might do"""

SYSTEM_PROMPT_OPEN = """\
You are a code-aware assistant. Answer questions using the provided source code \
and documentation context as primary references. Cite sources using [S1], [S2], \
etc. markers. You may supplement with general knowledge when the context is \
insufficient, but clearly indicate when doing so."""


class CodeSynthesizer:
    """Generates grounded answers from code context."""

    def __init__(self, chat: "ChatProvider", config: "FitzKragConfig"):
        self._chat = chat
        self._config = config

    def generate(
        self,
        query: str,
        context: str,
        results: list[ReadResult],
        answer_mode: AnswerMode = AnswerMode.TRUSTWORTHY,
    ) -> Answer:
        """
        Generate an answer grounded in the provided code context.

        Args:
            query: User's question
            context: Assembled context string from ContextAssembler
            results: ReadResults used to build provenance
            answer_mode: Epistemic posture controlling answer framing

        Returns:
            Answer with text, provenance, and metadata
        """
        # ABSTAIN: skip LLM call entirely
        if answer_mode == AnswerMode.ABSTAIN:
            provenance = self._build_provenance(results)
            return Answer(
                text="The available information does not allow a definitive answer.",
                provenance=provenance,
                mode=answer_mode,
                metadata={
                    "engine": "fitz_krag",
                    "query": query,
                    "sources_count": len(results),
                    "answer_mode": answer_mode.value,
                },
            )

        system_prompt = (
            SYSTEM_PROMPT_GROUNDED if self._config.strict_grounding else SYSTEM_PROMPT_OPEN
        )

        # Prepend mode instruction
        mode_instruction = get_mode_instruction(answer_mode)
        system_prompt = f"{mode_instruction}\n\n{system_prompt}"

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        try:
            raw_answer = self._chat.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise GenerationError(f"LLM generation failed: {e}") from e

        provenance = self._build_provenance(results)

        return Answer(
            text=raw_answer,
            provenance=provenance,
            mode=answer_mode,
            metadata={
                "engine": "fitz_krag",
                "query": query,
                "sources_count": len(results),
                "answer_mode": answer_mode.value,
            },
        )

    def _build_provenance(self, results: list[ReadResult]) -> list[Provenance]:
        """Build provenance list from read results."""
        provenance = []
        for r in results:
            # Skip context-type expansions (imports, class headers) from provenance
            if r.metadata.get("context_type"):
                continue

            metadata = {
                "kind": r.address.kind.value,
                "location": r.address.location,
                "file_path": r.file_path,
            }
            if r.line_range:
                metadata["line_range"] = list(r.line_range)
            metadata.update({k: v for k, v in r.address.metadata.items() if k != "context_type"})

            provenance.append(
                Provenance(
                    source_id=r.address.source_id,
                    excerpt=r.content[:200] if r.content else "",
                    metadata=metadata,
                )
            )
        return provenance
