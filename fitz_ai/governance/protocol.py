# fitz_ai/governance/protocol.py
"""
EvidenceItem Protocol - Generic interface for evidence that constraints can evaluate.

Both Chunk (core) and ReadResult (KRAG) satisfy this protocol without adapter code.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EvidenceItem(Protocol):
    """Any retrieved evidence that constraints can evaluate."""

    content: str
    metadata: dict[str, Any]


__all__ = ["EvidenceItem"]
