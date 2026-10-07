from __future__ import annotations

from .base import (
    AcquisitionTrace,
    ActionModelProposal,
    LearnerFailure,
)
from ..hypotheses import WorldContractHypothesis


class TraceOnlyPredicateLearner:
    learner_id = "trace-only-lifted"

    def update(
        self,
        traces: tuple[AcquisitionTrace, ...],
        current_contracts: tuple[WorldContractHypothesis, ...],
    ) -> ActionModelProposal | LearnerFailure:
        del traces, current_contracts
        return LearnerFailure(
            self.learner_id,
            "backend-not-implemented",
        )
