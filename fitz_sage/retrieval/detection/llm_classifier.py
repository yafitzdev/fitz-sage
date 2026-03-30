# fitz_sage/retrieval/detection/llm_classifier.py
"""
LLM-based query classification using detection modules.

Combines all module prompt fragments into a single LLM call,
then distributes results to each module for parsing.

Similar to the enrichment bus, but for query-time classification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fitz_sage.llm.factory import ChatFactory, ModelTier
from fitz_sage.logging.logger import get_logger

if TYPE_CHECKING:
    from .modules.base import DetectionModule
    from .protocol import DetectionCategory, DetectionResult

logger = get_logger(__name__)

PROMPT_HEADER = """Classify this search query. Return JSON only.

Query: "{query}"

Return this exact structure:
{{
  {module_fragments}
}}

Only set detected=true when the query CLEARLY matches the criteria. Default to detected=false unless there is explicit evidence."""


@dataclass
class LLMClassifier:
    """
    Module-based LLM classifier.

    Combines all module prompt fragments into one LLM call,
    then distributes results to each module for parsing.
    """

    chat_factory: ChatFactory
    modules: list["DetectionModule"] = field(default_factory=list)

    # Tier for classification (developer decision)
    TIER_CLASSIFY: ModelTier = "fast"

    def __post_init__(self):
        """Load default modules if none provided."""
        if not self.modules:
            from .modules import DEFAULT_MODULES

            self.modules = list(DEFAULT_MODULES)

    def classify(
        self,
        query: str,
        limit_to: "set[DetectionCategory] | None" = None,
    ) -> dict["DetectionCategory", "DetectionResult[Any]"]:
        """
        Classify query using all modules (or a filtered subset) in one LLM call.

        Args:
            query: User's query string
            limit_to: If provided, only run modules whose category is in this set.
                If the filtered list is empty, returns empty results without calling LLM.

        Returns:
            Dict mapping DetectionCategory to DetectionResult
        """
        active_modules = self.modules
        if limit_to is not None:
            active_modules = [m for m in self.modules if m.category in limit_to]
            if not active_modules:
                return self._empty_results()

        prompt = self._build_prompt(query, active_modules)

        try:
            chat = self.chat_factory(self.TIER_CLASSIFY)
            response = chat.chat([{"role": "user", "content": prompt}])
            raw_results = self._parse_response(response)
            return self._distribute_to_modules(raw_results, active_modules)
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return self._empty_results(active_modules)

    def _build_prompt(self, query: str, modules: "list[DetectionModule] | None" = None) -> str:
        """Build combined prompt from the given modules (defaults to self.modules)."""
        active = modules if modules is not None else self.modules
        fragments = [m.prompt_fragment() for m in active]
        combined = ",\n  ".join(fragments)
        return PROMPT_HEADER.format(query=query, module_fragments=combined)

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = response.strip()

        # Try to find JSON in code block
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        continue

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break

        logger.warning(f"Failed to parse LLM response as JSON: {text[:100]}...")
        return {}

    def _distribute_to_modules(
        self,
        raw_results: dict[str, Any],
        modules: "list[DetectionModule] | None" = None,
    ) -> dict["DetectionCategory", "DetectionResult[Any]"]:
        """Distribute parsed results to each module (defaults to self.modules)."""
        active = modules if modules is not None else self.modules
        return distribute_to_modules(raw_results, active)

    def _empty_results(
        self, modules: "list[DetectionModule] | None" = None
    ) -> dict["DetectionCategory", "DetectionResult[Any]"]:
        """Return not-detected results for the given modules (defaults to self.modules)."""
        active = modules if modules is not None else self.modules
        return {module.category: module.not_detected() for module in active}


def distribute_to_modules(
    raw_results: dict[str, Any],
    modules: "list[DetectionModule]",
) -> dict["DetectionCategory", "DetectionResult[Any]"]:
    """Distribute parsed JSON results to detection modules.

    Used by both LLMClassifier and QueryBatcher.
    """
    results = {}
    for module in modules:
        module_data = raw_results.get(module.json_key, {})
        if not isinstance(module_data, dict):
            module_data = {}
        results[module.category] = module.parse_result(module_data)
    return results
