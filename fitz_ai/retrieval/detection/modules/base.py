# fitz_ai/retrieval/detection/modules/base.py
"""
Base class for detection modules.

Each module defines:
- What category it detects
- Its prompt fragment (combined with others into one LLM call)
- How to parse LLM response into DetectionResult

Similar to enrichment bus modules, but all modules share ONE LLM call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fitz_ai.retrieval.detection.protocol import DetectionCategory, DetectionResult


class DetectionModule(ABC):
    """
    Base class for detection modules.

    Each module contributes a fragment to the combined classification prompt.
    All modules are evaluated in a single LLM call for efficiency.

    To add a new detection category:
    1. Create a module file in modules/
    2. Inherit from DetectionModule
    3. Implement category, json_key, prompt_fragment(), parse_result()
    4. Register in modules/__init__.py
    """

    @property
    @abstractmethod
    def category(self) -> DetectionCategory:
        """The detection category this module handles."""
        ...

    @property
    @abstractmethod
    def json_key(self) -> str:
        """Key in the JSON response for this module's output."""
        ...

    @abstractmethod
    def prompt_fragment(self) -> str:
        """
        Prompt fragment describing what to detect.

        This is combined with other modules into one prompt.
        Should describe the JSON structure expected for this key.
        """
        ...

    @abstractmethod
    def parse_result(self, data: dict[str, Any]) -> DetectionResult[Any]:
        """
        Parse LLM response data into DetectionResult.

        Args:
            data: The dict from LLM response for this module's json_key

        Returns:
            DetectionResult with detection info
        """
        ...

    def not_detected(self) -> DetectionResult[Any]:
        """Return a not-detected result for this category."""
        return DetectionResult.not_detected(self.category)
