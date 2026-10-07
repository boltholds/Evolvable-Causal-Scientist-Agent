import pytest

from ecsa.world_model.hypotheses import WorldContractHypothesis
from ecsa.world_model.learners.base import (
    AcquisitionActionStep,
    AcquisitionTrace,
    ActionModelLearner,
    ActionModelProposal,
    LearnerFailure,
)


class EchoLearner:
    learner_id = "echo"

    def update(
        self,
        traces: tuple[AcquisitionTrace, ...],
        current_contracts: tuple[WorldContractHypothesis, ...],
    ) -> ActionModelProposal | LearnerFailure:
        evidence = tuple(
            step.evidence_id
            for trace in traces
            for step in trace.steps
        )
        return ActionModelProposal(
            learner_id=self.learner_id,
            entity_types=(),
            predicates=(),
            actions=(),
            supporting_evidence_ids=evidence,
        )


def _trace() -> AcquisitionTrace:
    return AcquisitionTrace(
        trace_id="trace-1",
        steps=(
            AcquisitionActionStep(
                schema_id="A17",
                object_refs=("o1",),
                evidence_id="e1",
            ),
            AcquisitionActionStep(
                schema_id="B9",
                object_refs=("o1", "o2"),
                evidence_id="e2",
            ),
        ),
    )


def test_action_model_learner_protocol_returns_ecsa_proposals_only() -> None:
    learner: ActionModelLearner = EchoLearner()

    result = learner.update((_trace(),), ())

    assert isinstance(result, ActionModelProposal)
    assert result.learner_id == "echo"
    assert result.supporting_evidence_ids == ("e1", "e2")
    assert not hasattr(result, "model")


def test_acquisition_trace_rejects_empty_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        AcquisitionTrace(trace_id="empty", steps=())


def test_action_step_requires_nonempty_object_refs() -> None:
    with pytest.raises(ValueError, match="object"):
        AcquisitionActionStep(
            schema_id="A17",
            object_refs=("",),
            evidence_id="e1",
        )


def test_learner_failure_is_typed_and_nonempty() -> None:
    failure = LearnerFailure(
        learner_id="backend",
        reason="backend-not-installed",
    )

    assert failure.learner_id == "backend"
    assert failure.reason == "backend-not-installed"
