"""Transparent progress regularization of existing ECSA experiment candidates.

This selector never invents an EIG proxy: ScienceKernel information gain is
supplied by the coordinator and remains lexicographically primary whenever
strictly positive. Progress resolves uninformative ties and bootstrap choices.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .contracts import GroundAction
from .experiments import ContractExperiment
from .learning_progress import LearningProgressLedger, ProgressAssessment


@dataclass(frozen=True)
class SelectionBreakdown:
    experiment_id: str
    information_gain_bits: float
    uncertainty: float
    progress_estimate: float
    stagnation_penalty: float
    cost: float
    total_score: float


class ProgressAwareSelection:
    def __init__(self, ledger: LearningProgressLedger):
        if not isinstance(ledger, LearningProgressLedger):
            raise TypeError("LearningProgressLedger required")
        self.ledger = ledger

    def select(
        self,
        experiments: tuple[ContractExperiment, ...],
        *,
        eig: dict[str, float],
        context_signature: tuple[str, ...],
    ) -> tuple[ContractExperiment, SelectionBreakdown]:
        if not isinstance(experiments, tuple) or not experiments or not all(
            isinstance(exp, ContractExperiment) for exp in experiments
        ):
            raise ValueError("nonempty typed candidate tuple required")
        if not isinstance(eig, dict):
            raise TypeError("EIG scores must be a mapping")
        if any(
            not isinstance(k, str) or isinstance(v, bool)
            or not isinstance(v, (float, int)) or not isfinite(v) or v < 0
            for k, v in eig.items()
        ):
            raise ValueError("EIG scores must be finite and nonnegative")

        evaluations: list[tuple[ContractExperiment, SelectionBreakdown]] = []
        for exp in experiments:
            score = self.ledger.assess(exp.action, context_signature=context_signature)
            same = [
                ev for ev in self.ledger.recent
                if ev.context_signature == context_signature and
                self.ledger.assess(ev.action, context_signature=context_signature).action_signature
                == score.action_signature
            ]
            cost = sum(ev.action_cost for ev in same) / len(same) if same else 1.0
            uncertainty = 1.0 / (1.0 + len(same))
            expected = max(0.0, score.expected_gain or 0.0)
            # Bounded repetition penalty: scientifically valuable independent
            # confirmations still earn positive progress and can be repeated.
            utility = (
                1.5 * expected
                + 0.08 * score.independent_confirmation_count
                + 0.3 * uncertainty
                - 1.2 * score.stagnation_penalty
                - 0.02 * cost
            )
            evaluations.append((
                exp, SelectionBreakdown(
                    exp.experiment_id, float(eig.get(exp.experiment_id, 0.0)),
                    uncertainty, expected, score.stagnation_penalty, cost, utility,
                )
            ))
        # Lexicographic EIG protects validated science priorities. If no
        # candidate has EIG, progress decides among public legal candidates.
        chosen = min(evaluations, key=lambda item: (
            -item[1].information_gain_bits, -item[1].total_score,
            item[0].experiment_id,
        ))
        return chosen
