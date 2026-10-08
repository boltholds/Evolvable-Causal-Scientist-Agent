"""Scheduler RED gates for lifted mechanism transfer and ablation."""

from ecsa.contracts import TheoryPosterior
from ecsa.world_model.contracts import GroundAction, freeze_raw_value
from ecsa.world_model.experiments import (
    ContractExperiment,
    ContractExperimentCoordinator,
    ContractOutcomePrediction,
    ExperimentSelectionMode,
)
from ecsa.world_model.perception.base import (
    EntityObservation,
    ObservedFeature,
    PerceptualObservation,
)


def action(ref: str) -> GroundAction:
    return GroundAction("opaque", (freeze_raw_value(ref),))


def perception(*entries: tuple[str, bool]) -> PerceptualObservation:
    return PerceptualObservation(
        observation_id="visible",
        entities=tuple(
            EntityObservation(
                local_ref="local-" + ref,
                source_identity=ref,
                features=(
                    ObservedFeature("identifying", freeze_raw_value(ref), "visible"),
                    ObservedFeature("f0", freeze_raw_value(flag), "visible"),
                ),
                provenance_id="visible",
            )
            for ref, flag in entries
        ),
        global_features=(),
    )


def experiment(ref: str) -> ContractExperiment:
    return ContractExperiment("test-" + ref, action(ref), ())


def train(coordinator: ContractExperimentCoordinator) -> None:
    for ref, state, result in (
        ("a", True, True), ("b", True, True), ("c", True, True),
        ("d", False, False), ("e", False, False), ("f", False, False),
    ):
        coordinator.record_outcome(action(ref), success=result, before=perception((ref, state)))


def test_lifted_mechanism_enters_science_kernel_for_unseen_arguments() -> None:
    coordinator = ContractExperimentCoordinator()
    train(coordinator)
    assert coordinator.lifted_applicability.belief("opaque", 1) is not None
    selected = coordinator.select_applicability(
        perception=perception(("never-seen-yes", True), ("never-seen-no", False)),
        experiments=(experiment("never-seen-yes"), experiment("never-seen-no")),
    )
    assert selected in (experiment("never-seen-yes"), experiment("never-seen-no"))
    assert coordinator.last_selection_mode is ExperimentSelectionMode.LIFTED_APPLICABILITY_EIG


def test_lifted_backend_is_independently_ablated() -> None:
    coordinator = ContractExperimentCoordinator(use_lifted_selection=False)
    train(coordinator)
    coordinator.select_applicability(
        perception=perception(("never-seen-yes", True), ("never-seen-no", False)),
        experiments=(experiment("never-seen-yes"), experiment("never-seen-no")),
    )
    assert coordinator.last_selection_mode is not ExperimentSelectionMode.LIFTED_APPLICABILITY_EIG


def test_world_contract_positive_eig_outranks_lifted() -> None:
    coordinator = ContractExperimentCoordinator()
    train(coordinator)
    first = ContractExperiment(
        "contract-discriminator", action("never-seen-yes"),
        (
            ContractOutcomePrediction("h1", "contract-discriminator", 0.99),
            ContractOutcomePrediction("h2", "contract-discriminator", 0.01),
        ),
    )
    second = ContractExperiment(
        "contract-neutral", action("never-seen-no"),
        (
            ContractOutcomePrediction("h1", "contract-neutral", 0.5),
            ContractOutcomePrediction("h2", "contract-neutral", 0.5),
        ),
    )
    selected = coordinator.select_active(
        posterior=TheoryPosterior((("h1", 0.5), ("h2", 0.5))),
        experiments=(first, second),
        perception=perception(("never-seen-yes", True), ("never-seen-no", False)),
    )
    assert selected == first
    assert coordinator.last_selection_mode is ExperimentSelectionMode.WORLD_CONTRACT_EIG


def test_equally_informative_nonzero_eig_does_not_fall_back_to_novelty() -> None:
    coordinator = ContractExperimentCoordinator(use_lifted_selection=False)
    for _ in range(2):
        coordinator.record_outcome(action("seen"), success=True, before=perception(("seen", True)))
    coordinator.select_applicability(
        perception=perception(("unseen-1", False), ("unseen-2", False)),
        experiments=(experiment("unseen-1"), experiment("unseen-2")),
    )
    assert coordinator.last_selection_mode is ExperimentSelectionMode.APPLICABILITY_EIG
