"""Regression: opaque public action values must not become invented LOCM objects."""
from __future__ import annotations

import pytest

from ecsa.world_model.contracts import (
    GroundAction, InteractionTransition, RawActionOutcome,
    RawObservation, freeze_raw_value,
)
from ecsa.world_model.grounding import InteractionGrounder
from ecsa.world_model.kernel import WorldModelAcquisitionKernel
from ecsa.world_model.learners.base import LearnerFailure
from ecsa.world_model.perception.base import PerceptualObservation


class CapturingLearner:
    learner_id = "capture-typed-trace"

    def __init__(self) -> None:
        self.traces = []

    def update(self, traces, current_contracts):
        self.traces.append(traces)
        return LearnerFailure(self.learner_id, "no hypothesis is asserted")


def grounding(index: int, arguments: tuple):
    before = RawObservation(
        f"raw-{index}-before", index, freeze_raw_value({"public": "visible"}),
    )
    after = RawObservation(
        f"raw-{index}-after", index + 1,
        freeze_raw_value({"public": "visible"}),
    )
    transition = InteractionTransition(
        f"transition-{index}", before,
        GroundAction("opaque-action", tuple(freeze_raw_value(x) for x in arguments)),
        RawActionOutcome(False, freeze_raw_value({"errors": []})),
        after,
    )
    pre = PerceptualObservation(before.observation_id, (), ())
    post = PerceptualObservation(after.observation_id, (), ())
    return InteractionGrounder().observe(transition, pre, post)


@pytest.mark.parametrize(
    "bad_value",
    ("", "   ", None, True, False, [], {}, 1.25),
    ids=("empty", "whitespace", "null", "bool-true",
         "bool-false", "sequence", "mapping", "float-literal"),
)
def test_non_object_raw_arguments_retain_evidence_without_becoming_locm_refs(bad_value):
    learner = CapturingLearner()
    kernel = WorldModelAcquisitionKernel(learners=(learner,))
    kernel.observe_grounding(grounding(0, ("object-A",)))
    bad = grounding(1, (bad_value,))
    update = kernel.observe_grounding(bad)

    assert update.new_evidence_ids
    assert bad.transition.outcome_evidence.evidence_id in update.new_evidence_ids
    assert bad.transition.participation[0].evidence_id in update.new_evidence_ids

    kernel.observe_grounding(grounding(2, (0,)))
    assert learner.traces, "two valid object-bearing observations must reach LOCM2"
    latest = learner.traces[-1][0].steps
    assert [step.object_refs for step in latest] == [("object-A",), ("0",)]
    assert bad.transition.outcome_evidence.evidence_id not in {
        step.evidence_id for step in latest
    }
    assert kernel.contract_hypotheses() == ()


def test_invalid_argument_does_not_reduce_object_arity_or_invent_partial_trace():
    learner = CapturingLearner()
    kernel = WorldModelAcquisitionKernel(learners=(learner,))
    kernel.observe_grounding(grounding(0, ("object-A", "object-B")))
    rejected = grounding(1, ("object-A", ""))
    kernel.observe_grounding(rejected)
    kernel.observe_grounding(grounding(2, ("object-C", "object-D")))

    assert learner.traces
    assert [
        step.object_refs for step in learner.traces[-1][0].steps
    ] == [
        ("object-A", "object-B"),
        ("object-C", "object-D"),
    ]


def test_parameterless_actions_remain_valid_for_symbolic_learning():
    learner = CapturingLearner()
    kernel = WorldModelAcquisitionKernel(learners=(learner,))
    kernel.observe_grounding(grounding(0, ()))
    kernel.observe_grounding(grounding(1, ()))
    assert [step.object_refs for step in learner.traces[-1][0].steps] == [(), ()]


def test_many_public_actions_with_empty_string_never_abort_evidence_collection():
    kernel = WorldModelAcquisitionKernel()
    for i in range(760):
        update = kernel.observe_grounding(grounding(i, ("",)))
        assert update.new_evidence_ids
    assert kernel.contract_hypotheses() == ()
