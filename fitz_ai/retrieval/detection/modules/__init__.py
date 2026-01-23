# fitz_ai/retrieval/detection/modules/__init__.py
"""
Detection modules for LLM-based query classification.

Each module contributes a prompt fragment and parsing logic.
All modules are combined into a single LLM call.

To add a new detection category:
1. Create a new module file (e.g., my_category.py)
2. Inherit from DetectionModule
3. Implement category, json_key, prompt_fragment(), parse_result()
4. Add to DEFAULT_MODULES below
"""

from .aggregation import AggregationModule, AggregationType
from .base import DetectionModule
from .comparison import ComparisonModule
from .freshness import FreshnessModule
from .rewriter import RewriterModule
from .temporal import TemporalIntent, TemporalModule

# Default modules used by DetectionOrchestrator
DEFAULT_MODULES: list[DetectionModule] = [
    TemporalModule(),
    AggregationModule(),
    ComparisonModule(),
    FreshnessModule(),
    RewriterModule(),
]

__all__ = [
    # Base
    "DetectionModule",
    "DEFAULT_MODULES",
    # Modules
    "AggregationModule",
    "ComparisonModule",
    "FreshnessModule",
    "RewriterModule",
    "TemporalModule",
    # Enums
    "AggregationType",
    "TemporalIntent",
]
