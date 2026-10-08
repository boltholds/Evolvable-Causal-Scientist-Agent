from ecsa.world_model.applicability import ApplicabilityRule
from ecsa.world_model.contracts import GroundAction, freeze_raw_value
from ecsa.world_model.experiments import (
    ContractExperiment,
    ContractExperimentCoordinator,
    ExperimentSelectionMode,
)
from ecsa.world_model.perception.base import ObservedFeature, PerceptualObservation


def observation(value: str) -> PerceptualObservation:
    return PerceptualObservation(
        "observation", (),
        (ObservedFeature("state-attribute", freeze_raw_value(value), "visible"),),
    )


def action(value: str) -> GroundAction:
    return GroundAction("opaque-action", (freeze_raw_value(value),))


def candidates() -> tuple[ContractExperiment, ...]:
    return (
        ContractExperiment("trial-x", action("x"), ()),
        ContractExperiment("trial-y", action("y"), ()),
    )


def test_two_failures_support_uncertain_differing_argument_rule_not_proof() -> None:
    coordinator = ContractExperimentCoordinator()
    for _ in range(2):
        coordinator.record_outcome(action("x"), success=False, before=observation("s"))
    belief = coordinator.applicability.belief("opaque-action", 1)
    assert belief is not None
    assert ApplicabilityRule.ARGUMENT_DIFFERS in {
        hypothesis.rule for hypothesis in belief.hypotheses
    }
    # The separate conservative precondition learner does not infer
    # necessary conditions from failures only.
    assert coordinator.preconditions.hypotheses(action("x")) == ()
    selected = coordinator.select_applicability(
        perception=observation("s"), experiments=candidates(),
    )
    assert selected.action == action("y")
    assert coordinator.last_selection_mode is ExperimentSelectionMode.APPLICABILITY_EIG


def test_two_successes_can_motivate_discriminating_alternative() -> None:
    coordinator = ContractExperimentCoordinator()
    for _ in range(2):
        coordinator.record_outcome(action("x"), success=True, before=observation("s"))
    belief = coordinator.applicability.belief("opaque-action", 1)
    assert belief is not None
    selected = coordinator.select_applicability(
        perception=observation("s"), experiments=candidates(),
    )
    assert selected.action == action("y")
    assert coordinator.last_selection_mode is ExperimentSelectionMode.APPLICABILITY_EIG


def test_single_failure_stays_in_structural_bootstrap() -> None:
    coordinator = ContractExperimentCoordinator()
    coordinator.record_outcome(action("x"), success=False, before=observation("s"))
    assert coordinator.applicability.belief("opaque-action", 1) is None
    coordinator.select_applicability(
        perception=observation("s"), experiments=candidates(),
    )
    assert coordinator.last_selection_mode is ExperimentSelectionMode.STRUCTURAL
