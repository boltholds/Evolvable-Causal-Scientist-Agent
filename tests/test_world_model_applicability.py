from __future__ import annotations

from ecsa.contracts import TheoryPosterior
from ecsa.world_model.applicability import (
    ActiveApplicabilityLearner,
    ApplicabilityRule,
)
from ecsa.world_model.contracts import GroundAction, freeze_raw_value
from ecsa.world_model.experiments import (
    ContractExperiment,
    ContractExperimentCoordinator,
    ContractOutcomePrediction,
)
from ecsa.world_model.perception.base import ObservedFeature, PerceptualObservation


def state(value: int | None) -> PerceptualObservation:
    return PerceptualObservation(
        observation_id="opaque-observation",
        entities=(),
        global_features=() if value is None else (
            ObservedFeature("f", freeze_raw_value(value), "public"),
        ),
    )


def action(value: str) -> GroundAction:
    return GroundAction("opaque", (freeze_raw_value(value),))


def experiment(value: str) -> ContractExperiment:
    return ContractExperiment("trial-" + value, action(value), ())


def trained_coordinator() -> ContractExperimentCoordinator:
    coordinator = ContractExperimentCoordinator()
    coordinator.record_outcome(action("x"), success=True, before=state(1))
    coordinator.record_outcome(action("x"), success=False, before=state(0))
    coordinator.record_outcome(action("y"), success=False, before=state(1))
    return coordinator


def test_outcomes_create_competing_joint_argument_and_state_hypotheses() -> None:
    learner = trained_coordinator().applicability
    belief = learner.belief("opaque", 1)
    assert belief is not None
    assert belief.evidence_count == 3
    rules = {hypothesis.rule for hypothesis in belief.hypotheses}
    assert {
        ApplicabilityRule.ALWAYS,
        ApplicabilityRule.NEVER,
        ApplicabilityRule.ARGUMENT_EQUALS,
        ApplicabilityRule.FEATURE_PRESENT,
        ApplicabilityRule.ARGUMENT_AND_FEATURE,
    }.issubset(rules)
    joint = next(h for h in belief.hypotheses if h.rule is ApplicabilityRule.ARGUMENT_AND_FEATURE)
    unconditional = next(h for h in belief.hypotheses if h.rule is ApplicabilityRule.ALWAYS)
    assert belief.posterior.probability(joint.theory_id) > belief.posterior.probability(unconditional.theory_id)
    predictions = learner.predictions(
        belief, experiment_id="test-x", action=action("x"), before=state(1),
    )
    joint_prediction = next(p for p in predictions if p.theory_id == joint.theory_id)
    assert joint_prediction.probability(("success",)) == 0.9
    changed = learner.predictions(
        belief, experiment_id="test-y", action=action("y"), before=state(1),
    )
    joint_changed = next(p for p in changed if p.theory_id == joint.theory_id)
    assert joint_changed.probability(("success",)) < 0.11


def test_failure_only_does_not_invent_applicability_preconditions() -> None:
    coordinator = ContractExperimentCoordinator()
    coordinator.record_outcome(action("x"), success=False, before=state(0))
    assert coordinator.applicability.belief("opaque", 1) is None
    assert coordinator.select_applicability(
        perception=state(0),
        experiments=(experiment("x"), experiment("y")),
    ).action == action("y")


def test_absence_of_observed_feature_can_be_proposed_not_proven() -> None:
    learner = ActiveApplicabilityLearner()
    learner.observe(action("x"), success=True, before=state(None))
    learner.observe(action("x"), success=False, before=state(0))
    belief = learner.belief("opaque", 1)
    assert belief is not None
    assert ApplicabilityRule.FEATURE_ABSENT in {
        hypothesis.rule for hypothesis in belief.hypotheses
    }


def test_applicability_selection_uses_same_science_kernel_eig() -> None:
    coordinator = trained_coordinator()
    candidates = (experiment("x"), experiment("y"))
    belief = coordinator.applicability.belief("opaque", 1)
    assert belief is not None
    scores = {
        candidate.experiment_id: coordinator.science.score_experiment(
            belief.posterior,
            coordinator.applicability.predictions(
                belief,
                experiment_id=candidate.experiment_id,
                action=candidate.action,
                before=state(1),
            ),
        ).information_gain_bits
        for candidate in candidates
    }
    assert abs(scores["trial-x"] - scores["trial-y"]) > 1e-6
    selected = coordinator.select_applicability(
        perception=state(1),
        experiments=candidates,
    )
    assert selected.experiment_id == max(scores, key=scores.get)


def test_disabling_applicability_is_a_real_ablation() -> None:
    coordinator = ContractExperimentCoordinator(use_applicability_selection=False)
    for before, candidate, success in (
        (state(1), action("x"), True),
        (state(0), action("x"), False),
        (state(1), action("y"), False),
    ):
        coordinator.record_outcome(candidate, success=success, before=before)
    candidates = (experiment("x"), experiment("y"))
    assert coordinator.select_applicability(
        perception=state(1), experiments=candidates,
    ) == coordinator.select_bootstrap(candidates)


def test_positive_world_contract_eig_keeps_priority_over_applicability() -> None:
    coordinator = trained_coordinator()
    informative = ContractExperiment(
        "informative", action("x"),
        (
            ContractOutcomePrediction("h1", "informative", 0.99),
            ContractOutcomePrediction("h2", "informative", 0.01),
        ),
    )
    neutral = ContractExperiment(
        "neutral", action("y"),
        (
            ContractOutcomePrediction("h1", "neutral", 0.5),
            ContractOutcomePrediction("h2", "neutral", 0.5),
        ),
    )
    selected = coordinator.select_active(
        posterior=TheoryPosterior((("h1", 0.5), ("h2", 0.5))),
        experiments=(neutral, informative),
        perception=state(1),
    )
    assert selected == informative


def test_observations_are_bounded_per_schema() -> None:
    learner = ActiveApplicabilityLearner(max_history_per_schema=3)
    for value, success in ((1, True), (0, False), (1, True), (0, False)):
        learner.observe(action("x"), success=success, before=state(value))
    belief = learner.belief("opaque", 1)
    assert belief is not None
    assert belief.evidence_count == 3


def test_invalid_hypothesis_likelihood_is_rejected() -> None:
    try:
        ActiveApplicabilityLearner(reliability=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a deterministic zero-noise likelihood must be rejected")
