import importlib.util

import pytest

from ecsa.world_model.learners.base import (
    AcquisitionActionStep,
    AcquisitionTrace,
    ActionModelProposal,
    LearnerFailure,
)
from ecsa.world_model.learners.macq import MacqLocmLearner


def _locm_trace() -> AcquisitionTrace:
    return AcquisitionTrace(
        trace_id="trace-locm",
        steps=(
            AcquisitionActionStep("open", ("c1",), "e1"),
            AcquisitionActionStep("fetch", ("item1", "c1"), "e2"),
            AcquisitionActionStep("close", ("c1",), "e3"),
            AcquisitionActionStep("open", ("c2",), "e4"),
            AcquisitionActionStep("fetch", ("item2", "c2"), "e5"),
            AcquisitionActionStep("close", ("c2",), "e6"),
        ),
    )


def test_missing_macq_returns_typed_learner_failure(monkeypatch) -> None:
    learner = MacqLocmLearner()
    monkeypatch.setattr(
        "ecsa.world_model.learners.macq.importlib.util.find_spec",
        lambda name: None if name == "macq" else importlib.util.find_spec(name),
    )

    result = learner.update((_locm_trace(),), ())

    assert isinstance(result, LearnerFailure)
    assert result.learner_id == "macq-locm"
    assert "installed" in result.reason


def test_macq_locm_proposal_contains_no_macq_types() -> None:
    pytest.importorskip("macq")

    result = MacqLocmLearner().update((_locm_trace(),), ())

    assert isinstance(result, ActionModelProposal), result
    assert result.learner_id == "macq-locm"
    assert result.actions
    for value in (
        *result.entity_types,
        *result.predicates,
        *result.actions,
    ):
        assert not type(value).__module__.startswith("macq")


def test_incompatible_partial_trace_returns_typed_failure_not_kernel_exception() -> None:
    pytest.importorskip("macq")
    trace = AcquisitionTrace(
        trace_id="too-short",
        steps=(
            AcquisitionActionStep(
                schema_id="A17",
                object_refs=("o1",),
                evidence_id="e1",
            ),
        ),
    )

    result = MacqLocmLearner().update((trace,), ())

    assert isinstance(result, LearnerFailure)
    assert result.learner_id == "macq-locm"
