"""Held-out tests of empirical applicability, independent of action names."""

from ecsa.world_model.applicability import ApplicabilityRule
from ecsa.world_model.contracts import GroundAction, freeze_raw_value
from ecsa.world_model.experiments import (
    ContractExperiment,
    ContractExperimentCoordinator,
    ExperimentSelectionMode,
)
from ecsa.world_model.perception.base import ObservedFeature, PerceptualObservation


def before(bit: int) -> PerceptualObservation:
    return PerceptualObservation(
        "observation", (),
        (ObservedFeature("opaque-feature", freeze_raw_value(bit), "public"),),
    )


def action(argument: str) -> GroundAction:
    return GroundAction("hidden-schema", (freeze_raw_value(argument),))


def test_held_out_argument_state_combination_improves_prediction() -> None:
    """Known outcome mechanism: success exactly for (argument=a, feature=1).

    The oracle's semantics are used only for scoring, never supplied to ECSA.
    """
    coordinator = ContractExperimentCoordinator()
    for value, flag, outcome in (
        ("a", 1, True),
        ("a", 0, False),
        ("b", 1, False),
    ):
        coordinator.record_outcome(action(value), success=outcome, before=before(flag))

    belief = coordinator.applicability.belief("hidden-schema", 1)
    assert belief is not None

    # b in state 0 was NEVER among the training observations.
    held_out = action("b")
    predictions = coordinator.applicability.predictions(
        belief, experiment_id="held-out", action=held_out, before=before(0),
    )
    by_theory = {p.theory_id: p.probability(("success",)) for p in predictions}
    learned = sum(
        mass * by_theory[theory_id]
        for theory_id, mass in belief.posterior.probabilities
    )
    schema_frequency_baseline = 2 / 5  # Beta(1,1) on 1 success / 2 failures
    true_outcome = 0.0

    assert (learned - true_outcome) ** 2 < (
        schema_frequency_baseline - true_outcome
    ) ** 2


def test_counterexample_reweights_joint_mechanism() -> None:
    coordinator = ContractExperimentCoordinator()
    for value, flag, outcome in (
        ("a", 1, True),
        ("a", 0, False),
        ("b", 1, False),
    ):
        coordinator.record_outcome(action(value), success=outcome, before=before(flag))
    prior = coordinator.applicability.belief("hidden-schema", 1)
    assert prior is not None
    joint = next(h for h in prior.hypotheses if h.rule is ApplicabilityRule.ARGUMENT_AND_FEATURE)
    previous_mass = prior.posterior.probability(joint.theory_id)

    # This unexpected success contradicts the earlier joint rule.
    coordinator.record_outcome(action("b"), success=True, before=before(0))
    posterior = coordinator.applicability.belief("hidden-schema", 1)
    assert posterior is not None
    assert posterior.posterior.probability(joint.theory_id) < previous_mass


def test_selection_exposes_actual_active_or_structural_mode() -> None:
    coordinator = ContractExperimentCoordinator()
    candidates = (
        ContractExperiment("a", action("a"), ()),
        ContractExperiment("b", action("b"), ()),
    )
    assert coordinator.select_applicability(
        perception=before(1), experiments=candidates,
    ) in candidates
    assert coordinator.last_selection_mode is ExperimentSelectionMode.STRUCTURAL
    for value, flag, outcome in (
        ("a", 1, True), ("a", 0, False), ("b", 1, False),
    ):
        coordinator.record_outcome(action(value), success=outcome, before=before(flag))
    assert coordinator.select_applicability(
        perception=before(1), experiments=candidates,
    ) in candidates
    assert coordinator.last_selection_mode is ExperimentSelectionMode.APPLICABILITY_EIG
