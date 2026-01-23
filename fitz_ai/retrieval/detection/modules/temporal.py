# fitz_ai/retrieval/detection/modules/temporal.py
"""Temporal detection module."""

from __future__ import annotations

from enum import Enum
from typing import Any

from fitz_ai.retrieval.detection.protocol import DetectionCategory, DetectionResult

from .base import DetectionModule


class TemporalIntent(Enum):
    """Temporal query intent types."""

    COMPARISON = "COMPARISON"
    TREND = "TREND"
    POINT_IN_TIME = "POINT_IN_TIME"
    RANGE = "RANGE"
    SEQUENCE = "SEQUENCE"


class TemporalModule(DetectionModule):
    """Detects temporal/time-based queries."""

    @property
    def category(self) -> DetectionCategory:
        return DetectionCategory.TEMPORAL

    @property
    def json_key(self) -> str:
        return "temporal"

    def prompt_fragment(self) -> str:
        return '''"temporal": {
    "detected": true/false,
    "intent": "COMPARISON" | "TREND" | "POINT_IN_TIME" | "RANGE" | "SEQUENCE" | null,
    "references": [],
    "time_focused_queries": []
  }
  // temporal: time periods, versions, dates, quarters, "between X and Y", "since", "before", "last week/month/year"
  // intent: COMPARISON="between X and Y", TREND="over time/history", POINT_IN_TIME="as of/in Q1", RANGE="from X to Y", SEQUENCE="first/then/after"'''

    def parse_result(self, data: dict[str, Any]) -> DetectionResult[TemporalIntent]:
        if not data.get("detected", False):
            return self.not_detected()

        # Parse intent
        intent_str = data.get("intent")
        intent = None
        if intent_str:
            try:
                intent = TemporalIntent(intent_str)
            except ValueError:
                pass

        return DetectionResult(
            detected=True,
            category=self.category,
            confidence=0.9,
            intent=intent,
            matches=[],
            metadata={
                "references": data.get("references", []),
            },
            transformations=data.get("time_focused_queries", []),
        )
