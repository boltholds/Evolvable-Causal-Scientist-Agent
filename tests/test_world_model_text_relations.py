"""Whole-state text acquisition cannot inspect Y during action choice."""
from __future__ import annotations

import json

import pytest

from ecsa.world_model.contracts import (
    GroundAction, InteractionTransition, RawActionOutcome, RawObservation, freeze_raw_value,
)
from ecsa.world_model.text_relations import (
    FullTextTransitionProjector, TextRelationAcquisition, canonical_text,
)


def transition(*, ident="t1", before=None, after=None, arguments=("a",)):
    before = before if before is not None else {"message": "A valve is closed", "reading": 4.5}
    after = after if after is not None else {"message": "A valve is open", "reading": 1.1}
    return InteractionTransition(
        ident,
        RawObservation(ident+":before", 3, freeze_raw_value(before)),
        GroundAction("OPAQUE-7", tuple(freeze_raw_value(a) for a in arguments)),
        RawActionOutcome(True, freeze_raw_value({"ack": "observed"})),
        RawObservation(ident+":after", 4, freeze_raw_value(after)),
    )


def test_whole_text_serializes_every_payload_field_and_action():
    t = transition(before={"emoji": "открыт", "unknown": {"nested": [3, True, "a"]}})
    sample = FullTextTransitionProjector().project(t)
    assert json.loads(sample.x_text) == {
        "before": {"emoji": "открыт", "unknown": {"nested": [3, True, "a"]}},
        "action": {"schema_id": "OPAQUE-7", "arguments": ["a"]},
    }
    assert json.loads(sample.y_text) == {
        "after": {"message": "A valve is open", "reading": 1.1},
        "outcome": {"success": True, "payload": {"ack": "observed"}},
    }
    assert t.after.observation_id not in sample.x_text
    assert "ack" not in sample.x_text


def test_canonical_json_reorders_object_keys_but_preserves_content():
    assert canonical_text({"b": 1, "a": {"y": 2, "x": 3}}) == canonical_text(
        {"a": {"x": 3, "y": 2}, "b": 1})
    assert "-0.125" in canonical_text({"number": -0.125})
    with pytest.raises(ValueError):
        canonical_text({"number": float("nan")})


def test_candidate_encoding_does_not_know_future_y():
    projector = FullTextTransitionProjector()
    ta = transition(ident="a", after={"message": "secret AFTER"})
    tb = transition(ident="b", after={"message": "completely DIFFERENT"})
    assert projector.encode_candidate(ta.before, ta.action) == projector.encode_candidate(tb.before, tb.action)
    assert projector.encode_outcome(ta) != projector.encode_outcome(tb)


def test_store_is_bounded_deduplicated_and_selects_ready_schemas():
    sink_samples = []
    class Sink:
        def observe_text_pair(self, sample):
            sink_samples.append(sample)
    store = TextRelationAcquisition(max_pairs_per_schema=2, sink=Sink())
    assert store.observe_transition(transition(ident="t1")) is not None
    assert store.observe_transition(transition(ident="t1")) is None
    store.observe_transition(transition(ident="t2", before={"status":"open"}))
    store.observe_transition(transition(ident="t3", before={"status":"closed"}))
    assert store.total_samples == 2
    assert len(sink_samples) == 3
    assert [r.transition_id for r in store.samples_for("OPAQUE-7")] == ["t2", "t3"]
    assert store.ready_schemas(min_examples=2, min_distinct_x=2) == ("OPAQUE-7",)
    assert store.ready_schemas(min_examples=3) == ()


def test_invalid_capacity_and_sink_are_rejected():
    with pytest.raises(ValueError):
        TextRelationAcquisition(max_pairs_per_schema=0)
    with pytest.raises(TypeError):
        TextRelationAcquisition(sink=object())


def test_autonomous_scientist_receives_text_pairs_without_neural_dependencies():
    from ecsa.autonomy.scientist import AutonomousScientist
    from ecsa.world_model.kernel import WorldModelAcquisitionKernel
    from ecsa.world_model.experiments import ContractExperimentCoordinator
    from ecsa.world_model.perception.base import PerceptualObservation

    class EmptyPerception:
        def perceive(self, raw):
            return PerceptualObservation(raw.observation_id, (), ())

    store = TextRelationAcquisition()
    scientist = AutonomousScientist(
        world_model=WorldModelAcquisitionKernel(),
        experiments=ContractExperimentCoordinator(),
        perception=EmptyPerception(),
        text_relations=store,
    )
    update = scientist.observe_transition(transition())
    assert update.new_evidence_ids
    assert store.total_samples == 1
    assert "torch" not in __import__("ecsa.world_model.text_relations", fromlist=["-"]).__dict__
