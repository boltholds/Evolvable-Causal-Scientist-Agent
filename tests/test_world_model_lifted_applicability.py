"""RED gates for argument-relative applicability and identity-held-out transfer."""

from __future__ import annotations

from ecsa.world_model.contracts import GroundAction, freeze_raw_value
from ecsa.world_model.perception.base import (
    EntityObservation,
    ObservedFeature,
    PerceptualObservation,
)
from ecsa.world_model.lifted_applicability import (
    LiftedApplicabilityLearner,
    LiftedRuleKind,
    lifted_argument_features,
)


def action(*refs: str) -> GroundAction:
    return GroundAction("opaque", tuple(freeze_raw_value(ref) for ref in refs))


def observed(*objects: tuple[str, bool]) -> PerceptualObservation:
    return PerceptualObservation(
        observation_id="public",
        entities=tuple(
            EntityObservation(
                local_ref="local-" + ref,
                source_identity=ref,
                features=(
                    ObservedFeature("marker", freeze_raw_value(ref), "visible"),
                    ObservedFeature("f0", freeze_raw_value(flag), "visible"),
                ),
                provenance_id="public",
            )
            for ref, flag in objects
        ),
        global_features=(),
    )


def trained(prefix: str = "") -> LiftedApplicabilityLearner:
    learner = LiftedApplicabilityLearner()
    for ref, flag in (
        ("a", True), ("b", True), ("c", True),
        ("d", False), ("e", False), ("f", False),
    ):
        ref = prefix + ref
        learner.observe(action(ref), success=flag, before=observed((ref, flag)))
    return learner


def expected_success(
    learner: LiftedApplicabilityLearner,
    ref: str,
    flag: bool,
) -> float:
    belief = learner.belief("opaque", 1)
    assert belief is not None
    predictions = learner.predictions(
        belief, experiment_id="heldout-" + ref,
        action=action(ref), before=observed((ref, flag)),
    )
    return sum(
        mass * next(p for p in predictions if p.theory_id == key).probability(("success",))
        for key, mass in belief.posterior.probabilities
    )


def test_lifted_belief_generalizes_to_unseen_object_ids() -> None:
    learner = trained()
    belief = learner.belief("opaque", 1)
    assert belief is not None
    assert belief.evidence_count == 6
    assert any(
        h.kind is LiftedRuleKind.CONJUNCTION or h.kind is LiftedRuleKind.ARGUMENT_FEATURE
        for h in belief.hypotheses
    )
    positive = expected_success(learner, "never-seen-positive", True)
    negative = expected_success(learner, "never-seen-negative", False)
    assert positive > 0.70
    assert negative < 0.30


def test_ground_object_identifiers_are_not_lifted_predicates() -> None:
    features = lifted_argument_features(
        action("new-ref"), observed(("new-ref", True)),
    )
    assert features[0] is not None
    assert {literal.feature_id for literal in features[0]} == {"f0"}


def test_lifted_theory_fingerprints_do_not_depend_on_entity_ids() -> None:
    original = trained()
    renamed = trained(prefix="new-world-")
    a = original.belief("opaque", 1)
    b = renamed.belief("opaque", 1)
    assert a is not None and b is not None
    assert {h.theory_id for h in a.hypotheses} == {h.theory_id for h in b.hypotheses}
    assert expected_success(original, "held-a", True) == expected_success(
        renamed, "other-held-a", True,
    )


def test_unobserved_action_argument_does_not_count_as_feature_absence() -> None:
    learner = trained()
    belief = learner.belief("opaque", 1)
    assert belief is not None
    predictions = learner.predictions(
        belief,
        experiment_id="missing-entity",
        action=action("invisible"),
        before=observed(("other", True)),
    )
    for h, prediction in zip(belief.hypotheses, predictions):
        if h.kind not in (LiftedRuleKind.ALWAYS, LiftedRuleKind.NEVER):
            assert prediction.probability(("success",)) == 0.5


def test_arg1_rule_does_not_confuse_arg0_or_other_entities() -> None:
    learner = LiftedApplicabilityLearner()
    for first, second, ready in (
        ("u1", "a1", True), ("u2", "a2", True),
        ("u3", "b1", False), ("u4", "b2", False),
    ):
        learner.observe(
            action(first, second),
            success=ready,
            before=observed((first, False), (second, ready), ("distractor", True)),
        )
    belief = learner.belief("opaque", 2)
    assert belief is not None
    assert any(
        condition.argument_index == 1
        for h in belief.hypotheses
        for condition in h.conditions
    )
    def predict(second_id: str, ready: bool) -> float:
        before = observed(("unseen-first", False), (second_id, ready), ("distractor", True))
        predictions = learner.predictions(
            belief, experiment_id=second_id,
            action=action("unseen-first", second_id), before=before,
        )
        return sum(
            weight * next(p for p in predictions if p.theory_id == key).probability(("success",))
            for key, weight in belief.posterior.probabilities
        )
    assert predict("held-ready", True) > predict("held-not-ready", False) + 0.25


def test_insufficient_distinct_positive_objects_remains_underdetermined() -> None:
    learner = LiftedApplicabilityLearner()
    for _ in range(3):
        learner.observe(action("a"), success=True, before=observed(("a", True)))
    learner.observe(action("b"), success=False, before=observed(("b", False)))
    assert learner.belief("opaque", 1) is None


def test_uninformative_shared_feature_is_not_admitted_as_discriminator() -> None:
    learner = LiftedApplicabilityLearner()
    for ref, success in (("a", True), ("b", True), ("c", False), ("d", False)):
        learner.observe(action(ref), success=success, before=observed((ref, True)))
    assert learner.belief("opaque", 1) is None
