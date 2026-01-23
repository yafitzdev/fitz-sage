# fitz_ai/retrieval/detection/modules/rewriter.py
"""Rewriter detection module."""

from __future__ import annotations

from typing import Any

from fitz_ai.retrieval.detection.protocol import DetectionCategory, DetectionResult

from .base import DetectionModule


class RewriterModule(DetectionModule):
    """Detects queries that need rewriting (context, decomposition)."""

    @property
    def category(self) -> DetectionCategory:
        return DetectionCategory.REWRITER

    @property
    def json_key(self) -> str:
        return "rewriter"

    def prompt_fragment(self) -> str:
        return '''"rewriter": {
    "needs_context": true/false,
    "is_compound": true/false,
    "decomposed_queries": []
  }
  // needs_context: contains "it", "this", "that", "they", "the same" without clear referent in query
  // is_compound: multiple distinct questions, "and also", semicolons separating topics'''

    def parse_result(self, data: dict[str, Any]) -> DetectionResult[None]:
        needs_context = data.get("needs_context", False)
        is_compound = data.get("is_compound", False)

        if not needs_context and not is_compound:
            return self.not_detected()

        return DetectionResult(
            detected=True,
            category=self.category,
            confidence=0.9,
            intent=None,
            matches=[],
            metadata={
                "needs_context": needs_context,
                "is_compound": is_compound,
            },
            transformations=data.get("decomposed_queries", []),
        )
