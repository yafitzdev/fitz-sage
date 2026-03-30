"""Query - paradigm-agnostic query representation. See docs/API_REFERENCE.md for examples."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .constraints import Constraints


@dataclass
class Query:
    """Query representation: text + optional constraints + engine-specific metadata."""

    text: str
    constraints: Optional[Constraints] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize query text."""
        if self.text is None:
            self.text = ""
        if isinstance(self.text, str) and not self.text.strip():
            raise ValueError("Query text cannot be empty")
