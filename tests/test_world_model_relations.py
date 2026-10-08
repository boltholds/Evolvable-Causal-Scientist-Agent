"""RED gates: generic observed transitions become numeric X/Y relation samples.

No environment-specific semantic labels or neural framework is required.
"""
from __future__ import annotations

import inspect

from ecsa.autonomy.scientist import AutonomousScientist
from ecsa.world_model.contracts import (
    GroundAction,
    InteractionTransition,
    RawActionOutcome,
    RawObservation,
    freeze_raw_value,
)
from ecsa.world_model.kernel import WorldModelAcquisitionKernel
from ecsa.world_model.experiments import ContractExperimentCoordinator
from ecsa.world_model.perception.base import (
    EntityObservation,
    ObservedFeature,
    PerceptualObservation,
)
from ecsa.world_model.relations import (
    NumericRelationAcquisition,
    NumericRelationProjector,
    RelationScope,
)


def _entity(ref: str, obs: str, number: float, *, numeric_id: int | None = None):
    return EntityObservation(
        local_ref="local-" + ref,
        source_identity=ref,
        features=(
            ObservedFeature("opaque-f0", freeze_raw_value(number), obs),
            ObservedFeature("public-text", freeze_raw_value("uninterpreted"), obs),
            ObservedFeature("flag", freeze_raw_value(True), obs),
            *(
                (ObservedFeature("entity-index", freeze_raw_value(numeric_id), obs),)
                if numeric_id is not None else ()
            ),
        ),
        provenance_id=obs,
    )


def _view(obs: str, ref: str, number: float, *, numeric_id: int | None = None):
    return PerceptualObservation(
        observation_id=obs,
        entities=(_entity(ref, obs, number, numeric_id=numeric_id),),
        global_features=(
            ObservedFeature("global-input", freeze_raw_value(1.5), obs),
        ),
    )


def _transition(
    ref: str, *,
    before_id: str = "before",
    after_id: str = "after",
    success: bool = True,
):
    return InteractionTransition(
        transition_id=f"trial-{ref}-{before_id}",
        before=RawObservation(before_id, 0, freeze_raw_value({"state": before_id})),
        action=GroundAction("Q17", (
            freeze_raw_value(ref), freeze_raw_value(2.5),
        )),
        outcome=RawActionOutcome(success, freeze_raw_value({})),
        after=RawObservation(after_id, 1, freeze_raw_value({"state": after_id})),
    )


def test_projector_binds_arg_position_not_entity_id():
    projector = NumericRelationProjector()
    first = projector.project(
        _transition("id-11"),
        _view("before", "id-11", 3.5),
        _view("after", "id-11", 8.5),
    )
    second = projector.project(
        _transition("other-id"),
        _view("before", "other-id", 3.5),
        _view("after", "other-id", 8.5),
    )
    assert len(first) == len(second) == 2  # numeric entity and numeric global
    left = next(v for v in first if v.target.scope is RelationScope.ARGUMENT)
    right = next(v for v in second if v.target.scope is RelationScope.ARGUMENT)
    assert left.signature == right.signature
    assert left.target.argument_index == 0
    assert left.y == right.y == 8.5
    assert left.changed is True
    assert dict((x.key, x.value) for x in left.x) == {
        "arg[0].opaque-f0": 3.5,
        "arg_value[1]": 2.5,
        "global.global-input": 1.5,
    }
    assert "id-11" not in repr(left.x)


def test_id_like_numeric_features_and_bool_are_excluded():
    transition = _transition("11")
    samples = NumericRelationProjector().project(
        transition,
        _view("before", "11", 3.5, numeric_id=11),
        _view("after", "11", 4.5, numeric_id=11),
    )
    assert all(
        not any(feature.key.endswith("entity-index") for feature in sample.x)
        for sample in samples
    )
    assert {sample.target.feature_id for sample in samples} == {
        "opaque-f0", "global-input",
    }


def test_failures_remain_observations_but_do_not_train_effect_model_by_default():
    history = NumericRelationAcquisition()
    transition = _transition("id-11", success=False)
    results = history.observe_transition(
        transition,
        _view("before", "id-11", 1.0),
        _view("after", "id-11", 1.0),
    )
    assert len(results) == 2
    assert all(not sample.success and not sample.changed for sample in results)
    assert history.total_samples == 2
    assert history.ready_datasets(min_examples=1, min_unique_x=1) == ()
    assert history.observe_transition(
        transition,
        _view("before", "id-11", 1.0),
        _view("after", "id-11", 1.0),
    ) == ()
    assert history.total_samples == 2


def test_ready_dataset_groups_different_identities_with_same_signature():
    history = NumericRelationAcquisition()
    for index in range(4):
        ref = f"new-{index}"
        history.observe_transition(
            _transition(ref, before_id=f"before-{index}", after_id=f"after-{index}"),
            _view(f"before-{index}", ref, float(index)),
            _view(f"after-{index}", ref, float(index + 2)),
        )
    datasets = history.ready_datasets(min_examples=3, min_unique_x=3)
    assert len(datasets) == 2
    by_scope = {data.signature.target.scope: data for data in datasets}
    assert len(by_scope[RelationScope.ARGUMENT].samples) == 4
    assert all(
        sample.signature == by_scope[RelationScope.ARGUMENT].signature
        for sample in by_scope[RelationScope.ARGUMENT].samples
    )


def test_missing_or_ambiguous_identity_abstains_and_never_invents_y():
    transition = _transition("lost")
    before = _view("before", "lost", 1.)
    after = PerceptualObservation("after", (), ())
    assert NumericRelationProjector().project(transition, before, after) == ()
    empty = PerceptualObservation("before", (), ())
    empty_after = PerceptualObservation("after", (), ())
    assert NumericRelationProjector().project(transition, empty, empty_after) == ()


class NumericPerception:
    def perceive(self, raw: RawObservation) -> PerceptualObservation:
        value = 2.0 if raw.observation_id == "before" else 4.0
        return _view(raw.observation_id, "object", value)


def test_autonomous_scientist_emits_numeric_relations_on_real_transition():
    relations = NumericRelationAcquisition()
    scientist = AutonomousScientist(
        world_model=WorldModelAcquisitionKernel(),
        experiments=ContractExperimentCoordinator(),
        perception=NumericPerception(),
        relations=relations,
    )
    transition = _transition("object")
    update = scientist.observe_transition(transition)
    assert update.new_evidence_ids
    assert relations.total_samples == 2
    assert any(sample.y == 4.0 for sample in relations.samples)
    import ecsa.world_model.relations as module
    assert "torch" not in inspect.getsource(module)


def test_structured_discoveryworld_decoder_preserves_uninterpreted_numeric():
    from ecsa.benchmarks.discoveryworld.perception import DiscoveryWorldStructuredDecoder
    from ecsa.world_model.perception.structured import StructuredObservationFrontend
    raw = RawObservation(
        "observation-1",
        0,
        freeze_raw_value({
            "observation": {"ui": {
                "accessibleEnvironmentObjects": [
                    {"uuid": 37, "name": "object-37", "n7": 2.75, "v8": False}
                ],
            }},
        }),
    )
    observed = StructuredObservationFrontend(
        DiscoveryWorldStructuredDecoder()
    ).perceive(raw)
    assert observed.entities[0].source_identity == "37"
    assert any(
        feature.feature_id == "public:n7" and feature.value.thaw() == 2.75
        for feature in observed.entities[0].features
    )
    assert not any(feature.feature_id == "public:uuid" for feature in observed.entities[0].features)
