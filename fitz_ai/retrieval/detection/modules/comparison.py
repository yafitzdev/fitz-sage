# fitz_ai/retrieval/detection/modules/comparison.py
"""Comparison detection module."""

from __future__ import annotations

from typing import Any

from fitz_ai.retrieval.detection.protocol import DetectionCategory, DetectionResult

from .base import DetectionModule


class ComparisonModule(DetectionModule):
    """Detects comparison queries (A vs B, differences, etc.)."""

    @property
    def category(self) -> DetectionCategory:
        return DetectionCategory.COMPARISON

    @property
    def json_key(self) -> str:
        return "comparison"

    def prompt_fragment(self) -> str:
        return '''"comparison": {
    "detected": true/false,
    "entities": [],
    "comparison_queries": []
  }
  // comparison: "vs", "versus", "compare", "difference between", "differ", "which is better", "how does X compare to Y", "X or Y"'''

    def parse_result(self, data: dict[str, Any]) -> DetectionResult[None]:
        if not data.get("detected", False):
            return self.not_detected()

        return DetectionResult(
            detected=True,
            category=self.category,
            confidence=0.9,
            intent=None,
            matches=[],
            metadata={
                "entities": data.get("entities", []),
            },
            transformations=data.get("comparison_queries", []),
        )
