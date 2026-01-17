# fitz_ai/core/answer.py
"""Answer - paradigm-agnostic answer representation. See docs/api_reference.md for examples."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .provenance import Provenance

if TYPE_CHECKING:
    from fitz_ai.core.answer_mode import AnswerMode


@dataclass
class Answer:
    """Answer representation: text + source provenance + epistemic mode + metadata."""

    text: str
    provenance: List[Provenance] = field(default_factory=list)
    mode: Optional["AnswerMode"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate answer after initialization."""
        if self.text is None:
            raise ValueError("Answer text cannot be None (use empty string for no answer)")
