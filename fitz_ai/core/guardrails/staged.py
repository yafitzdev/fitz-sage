# fitz_ai/core/guardrails/staged.py
"""
Staged Constraint Pipeline - Hierarchical constraint evaluation.

Replaces flat sequential execution with a staged pipeline where:
- Stage 1 (Relevance) gates Stage 3 (Consistency)
- Context flows forward between stages
- Short-circuiting prevents wasted LLM calls

The output is still list[ConstraintResult], fully compatible with
AnswerGovernor.decide(). The governor does not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from fitz_ai.core.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from .base import ConstraintPlugin, ConstraintResult

logger = get_logger(__name__)

# Stage name constants
STAGE_RELEVANCE = "relevance"
STAGE_SUFFICIENCY = "sufficiency"
STAGE_CONSISTENCY = "consistency"

# Constraint name -> stage mapping
_STAGE_MAP: dict[str, str] = {
    "insufficient_evidence": STAGE_RELEVANCE,
    "specific_info_type": STAGE_RELEVANCE,
    "causal_attribution": STAGE_SUFFICIENCY,
    "answer_verification": STAGE_SUFFICIENCY,
    "conflict_aware": STAGE_CONSISTENCY,
    "deterministic_conflict": STAGE_CONSISTENCY,
}


@dataclass
class StageContext:
    """
    Accumulated context passed between constraint stages.

    Each stage reads prior context and contributes its own findings
    for downstream stages to use.
    """

    # Stage 1: Relevance
    relevance_confirmed: bool = True
    max_similarity: float = 1.0
    relevance_signal: str | None = None

    # Stage 2: Sufficiency
    sufficiency_checked: bool = False
    has_sufficient_evidence: bool = True
    evidence_gaps: list[str] = field(default_factory=list)

    # Stage 3: Consistency
    consistency_checked: bool = False


@dataclass
class ConstraintStage:
    """
    A named stage containing one or more constraints.

    Stages execute in order. Each stage receives accumulated context
    from prior stages and can contribute to it.
    """

    name: str
    constraints: list[ConstraintPlugin]
    short_circuit_signals: frozenset[str] = field(default_factory=frozenset)


class StagedConstraintPipeline:
    """
    Hierarchical constraint evaluation pipeline.

    Stages run in order. Each stage:
    1. Receives StageContext from prior stages
    2. Runs its constraints
    3. Updates StageContext for downstream stages
    4. Optionally short-circuits (skips remaining stages)

    Output: list[ConstraintResult] compatible with AnswerGovernor.decide().
    """

    def __init__(self, stages: list[ConstraintStage]) -> None:
        self.stages = stages

    def run(self, query: str, chunks: Sequence[Chunk]) -> list[ConstraintResult]:
        """Execute staged pipeline, return all constraint results."""
        all_results: list[ConstraintResult] = []
        context = StageContext()

        for stage in self.stages:
            if self._should_skip(stage, context):
                logger.debug(
                    f"{PIPELINE} StagedPipeline: skipping stage '{stage.name}' "
                    f"(relevance_signal={context.relevance_signal})"
                )
                continue

            stage_results = self._run_stage(query, chunks, stage)
            all_results.extend(stage_results)
            self._update_context(stage, stage_results, context)

            if self._check_short_circuit(stage, stage_results):
                logger.info(
                    f"{PIPELINE} StagedPipeline: stage '{stage.name}' triggered short-circuit"
                )
                break

        denied_count = sum(1 for r in all_results if not r.allow_decisive_answer)
        logger.debug(
            f"{PIPELINE} StagedPipeline complete: {len(all_results)} results, "
            f"{denied_count} denied"
        )
        return all_results

    def _should_skip(self, stage: ConstraintStage, context: StageContext) -> bool:
        """Check if stage should be skipped based on prior context."""
        if stage.name == STAGE_CONSISTENCY and context.relevance_signal == "abstain":
            return True
        if stage.name == STAGE_SUFFICIENCY and context.relevance_signal == "abstain":
            return True
        return False

    def _run_stage(
        self,
        query: str,
        chunks: Sequence[Chunk],
        stage: ConstraintStage,
    ) -> list[ConstraintResult]:
        """Run all constraints in a stage with error handling."""
        results: list[ConstraintResult] = []
        logger.debug(
            f"{PIPELINE} StagedPipeline: running stage '{stage.name}' "
            f"({len(stage.constraints)} constraint(s))"
        )

        for constraint in stage.constraints:
            try:
                result = constraint.apply(query, chunks)

                if not result.allow_decisive_answer:
                    metadata = dict(result.metadata)
                    metadata["constraint_name"] = constraint.name
                    metadata["stage"] = stage.name
                    result = ConstraintResult(
                        allow_decisive_answer=result.allow_decisive_answer,
                        reason=result.reason,
                        signal=result.signal,
                        metadata=metadata,
                    )
                    logger.info(
                        f"{PIPELINE} Constraint '{constraint.name}' "
                        f"(stage={stage.name}) denied: {result.reason}"
                    )
                else:
                    logger.debug(
                        f"{PIPELINE} Constraint '{constraint.name}' "
                        f"(stage={stage.name}) passed"
                    )

                results.append(result)

            except Exception as e:
                logger.warning(
                    f"{PIPELINE} Constraint '{constraint.name}' "
                    f"in stage '{stage.name}' raised: {e}"
                )
                continue

        return results

    def _update_context(
        self,
        stage: ConstraintStage,
        results: list[ConstraintResult],
        context: StageContext,
    ) -> None:
        """Update StageContext based on stage results."""
        if stage.name == STAGE_RELEVANCE:
            for r in results:
                if not r.allow_decisive_answer:
                    context.relevance_confirmed = False
                    context.relevance_signal = r.signal
                    context.max_similarity = r.metadata.get("max_similarity", 0.0)
                    break

        elif stage.name == STAGE_SUFFICIENCY:
            context.sufficiency_checked = True
            for r in results:
                if not r.allow_decisive_answer:
                    context.has_sufficient_evidence = False
                    if r.reason:
                        context.evidence_gaps.append(r.reason)

        elif stage.name == STAGE_CONSISTENCY:
            context.consistency_checked = True

    def _check_short_circuit(
        self,
        stage: ConstraintStage,
        results: list[ConstraintResult],
    ) -> bool:
        """Check if this stage's results should short-circuit remaining stages."""
        if not stage.short_circuit_signals:
            return False
        return any(r.signal in stage.short_circuit_signals for r in results)


def _build_staged_pipeline(
    constraints: Sequence[ConstraintPlugin],
) -> StagedConstraintPipeline:
    """
    Group constraints into stages based on their name.

    Unknown constraints default to the sufficiency stage (safe).
    GovernanceAnalyzer bypasses staging entirely (runs as single stage).
    """
    relevance: list[ConstraintPlugin] = []
    sufficiency: list[ConstraintPlugin] = []
    consistency: list[ConstraintPlugin] = []

    for c in constraints:
        name = c.name

        if name == "governance_analyzer":
            return StagedConstraintPipeline(
                stages=[ConstraintStage(name="unified", constraints=[c])]
            )

        stage_name = _STAGE_MAP.get(name, STAGE_SUFFICIENCY)
        if stage_name == STAGE_RELEVANCE:
            relevance.append(c)
        elif stage_name == STAGE_CONSISTENCY:
            consistency.append(c)
        else:
            sufficiency.append(c)

    stages: list[ConstraintStage] = []
    if relevance:
        stages.append(
            ConstraintStage(
                name=STAGE_RELEVANCE,
                constraints=relevance,
                short_circuit_signals=frozenset({"abstain"}),
            )
        )
    if sufficiency:
        stages.append(
            ConstraintStage(
                name=STAGE_SUFFICIENCY,
                constraints=sufficiency,
            )
        )
    if consistency:
        stages.append(
            ConstraintStage(
                name=STAGE_CONSISTENCY,
                constraints=consistency,
            )
        )

    return StagedConstraintPipeline(stages=stages)


def run_staged_constraints(
    query: str,
    chunks: Sequence[Chunk],
    constraints: Sequence[ConstraintPlugin],
) -> list[ConstraintResult]:
    """
    Drop-in replacement for run_constraints() using staged execution.

    Automatically groups constraints into stages based on their name.
    Returns list[ConstraintResult] compatible with AnswerGovernor.decide().
    """
    if not constraints:
        return []

    pipeline = _build_staged_pipeline(constraints)
    return pipeline.run(query, chunks)


__all__ = [
    "StageContext",
    "ConstraintStage",
    "StagedConstraintPipeline",
    "run_staged_constraints",
]
