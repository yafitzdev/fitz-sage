# fitz_ai/retrieval/rewriter/types.py
"""Type definitions for query rewriting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class RewriteType(Enum):
    """Type of rewrite performed."""

    NONE = "none"  # No rewrite needed
    CONVERSATIONAL = "conversational"  # Resolved pronouns/references
    CLARITY = "clarity"  # Fixed typos, simplified
    RETRIEVAL = "retrieval"  # Optimized for retrieval
    DECOMPOSITION = "decomposition"  # Compound query split into parts
    COMBINED = "combined"  # Multiple rewrites applied


@dataclass
class ConversationMessage:
    """Single message in conversation history."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class ConversationContext:
    """Conversation context for query rewriting."""

    history: List[ConversationMessage] = field(default_factory=list)
    max_turns: int = 5  # Limit history to last N turns

    def is_empty(self) -> bool:
        """Check if history is empty."""
        return len(self.history) == 0

    def recent_history(self) -> List[ConversationMessage]:
        """Get last N turns of history."""
        # 2 messages per turn (user + assistant)
        return self.history[-self.max_turns * 2 :]

    def format_for_prompt(self) -> str:
        """Format history for inclusion in prompt."""
        if self.is_empty():
            return ""

        lines = []
        for msg in self.recent_history():
            prefix = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)


@dataclass
class RewriteResult:
    """Result of query rewriting."""

    original_query: str
    rewritten_query: str
    rewrite_type: RewriteType
    confidence: float  # 0.0 to 1.0
    is_ambiguous: bool = False
    disambiguated_queries: List[str] = field(default_factory=list)
    is_compound: bool = False
    decomposed_queries: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def was_rewritten(self) -> bool:
        """Check if query was actually rewritten."""
        return (
            self.rewrite_type != RewriteType.NONE
            and self.rewritten_query != self.original_query
        )

    @property
    def all_query_variations(self) -> List[str]:
        """Get all query variations to search with."""
        variations = [self.original_query]

        if self.was_rewritten:
            variations.append(self.rewritten_query)

        # Add decomposed queries for compound queries (multiple topics)
        if self.is_compound and self.decomposed_queries:
            variations.extend(self.decomposed_queries)

        # Add disambiguated queries for ambiguous queries (multiple meanings)
        if self.is_ambiguous and self.disambiguated_queries:
            variations.extend(self.disambiguated_queries)

        return variations
