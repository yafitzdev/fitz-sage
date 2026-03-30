# fitz_sage/core/answer_mode.py
"""
Answer Mode - Epistemic framing for answers.

AnswerMode controls how certain the answer should sound,
not what it says. It is determined by constraint signals
after retrieval, before synthesis.

Modes:
- TRUSTWORTHY: Answer clearly and directly based on the evidence
- DISPUTED: Explicitly state sources disagree
- ABSTAIN: State that evidence is insufficient
"""

from enum import Enum


class AnswerMode(str, Enum):
    """
    Epistemic posture for answer generation.

    Selected based on constraint signals, not LLM reasoning.
    """

    TRUSTWORTHY = "trustworthy"
    """Evidence supports answering. Answer clearly and directly."""

    DISPUTED = "disputed"
    """Sources explicitly disagree; summarize the disagreement."""

    ABSTAIN = "abstain"
    """Evidence is insufficient; do not attempt a definitive answer."""
