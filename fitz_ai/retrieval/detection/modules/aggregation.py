# fitz_ai/retrieval/detection/modules/aggregation.py
"""Aggregation detection module."""

from __future__ import annotations

from enum import Enum
from typing import Any

from fitz_ai.retrieval.detection.protocol import DetectionCategory, DetectionResult

from .base import DetectionModule


class AggregationType(Enum):
    """Aggregation query types."""

    LIST = "LIST"
    COUNT = "COUNT"
    UNIQUE = "UNIQUE"


class AggregationModule(DetectionModule):
    """Detects aggregation queries (list all, count, enumerate)."""

    @property
    def category(self) -> DetectionCategory:
        return DetectionCategory.AGGREGATION

    @property
    def json_key(self) -> str:
        return "aggregation"

    def prompt_fragment(self) -> str:
        return '''"aggregation": {
    "detected": true/false,
    "type": "LIST" | "COUNT" | "UNIQUE" | null,
    "target": null,
    "fetch_multiplier": 1
  }
  // aggregation: "list all", "how many", "count", "enumerate", "what are all the", "show me all"
  // type: LIST="list/show/enumerate", COUNT="how many/count", UNIQUE="unique/distinct"'''

    def parse_result(self, data: dict[str, Any]) -> DetectionResult[AggregationType]:
        if not data.get("detected", False):
            return self.not_detected()

        # Parse type
        type_str = data.get("type")
        agg_type = None
        if type_str:
            try:
                agg_type = AggregationType(type_str)
            except ValueError:
                pass

        return DetectionResult(
            detected=True,
            category=self.category,
            confidence=0.9,
            intent=agg_type,
            matches=[],
            metadata={
                "target": data.get("target"),
                "fetch_multiplier": data.get("fetch_multiplier", 3),
            },
            transformations=[],
        )
