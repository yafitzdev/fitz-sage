# fitz_ai/retrieval/detection/modules/freshness.py
"""Freshness detection module."""

from __future__ import annotations

from typing import Any

from fitz_ai.retrieval.detection.protocol import DetectionCategory, DetectionResult

from .base import DetectionModule


class FreshnessModule(DetectionModule):
    """Detects freshness signals (recency, authority boosting)."""

    @property
    def category(self) -> DetectionCategory:
        return DetectionCategory.FRESHNESS

    @property
    def json_key(self) -> str:
        return "freshness"

    def prompt_fragment(self) -> str:
        return """"freshness": {
    "boost_recency": true/false,
    "boost_authority": true/false
  }
  // boost_recency: "latest", "recent", "new", "current", "updated", "newest"
  // boost_authority: "official", "recommended", "best practice", "standard", "proper way"
"""

    def parse_result(self, data: dict[str, Any]) -> DetectionResult[None]:
        boost_recency = data.get("boost_recency", False)
        boost_authority = data.get("boost_authority", False)

        if not boost_recency and not boost_authority:
            return self.not_detected()

        return DetectionResult(
            detected=True,
            category=self.category,
            confidence=0.9,
            intent=None,
            matches=[],
            metadata={
                "boost_recency": boost_recency,
                "boost_authority": boost_authority,
            },
            transformations=[],
        )
