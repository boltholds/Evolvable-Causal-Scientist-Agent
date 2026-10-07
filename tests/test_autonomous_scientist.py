import inspect

from ecsa.autonomy.scientist import AutonomousScientist
from ecsa.world_model.contracts import (
    GroundAction,
    InteractionTransition,
    RawActionOutcome,
    RawActionSchema,
    RawObservation,
    freeze_raw_value,
)
from ecsa.world_model.experiments import (
    ContractExperimentCoordinator,
    ExperimentBudget,
)
from ecsa.world_model.kernel import WorldModelAcquisitionKernel
from ecsa.world_model.perception.base import (
    PerceptualObservation,
)


class EmptyPerception:
    def perceive(
        self,
        raw_observation: RawObservation,
    ) -> PerceptualObservation:
        return PerceptualObservation(
            observation_id=raw_observation.observation_id,
            entities=(),
            global_features=(),
        )


def _observation(name: str, step: int) -> RawObservation:
    return RawObservation(
        observation_id=name,
        step=step,
        payload=freeze_raw_value({"opaque": step}),
    )


def test_autonomous_scientist_chooses_from_contract_experiments_not_action_roles() -> None:
    scientist = AutonomousScientist(
        world_model=WorldModelAcquisitionKernel(),
        experiments=ContractExperimentCoordinator(),
        perception=EmptyPerception(),
    )
    schemas = (
        RawActionSchema(
            schema_id="Z9",
            parameters=(),
            public_metadata=freeze_raw_value({}),
        ),
        RawActionSchema(
            schema_id="A1",
            parameters=(),
            public_metadata=freeze_raw_value({}),
        ),
    )

    experiment = scientist.choose_experiment(
        observation=_observation("obs-0", 0),
        actions=schemas,
        goal=freeze_raw_value({"goal": "opaque"}),
        budget=ExperimentBudget(max_ground_actions=8),
    )

    assert experiment.action == GroundAction(
        schema_id="A1",
        arguments=(),
    )


def test_autonomous_scientist_updates_world_model_after_every_outcome() -> None:
    kernel = WorldModelAcquisitionKernel()
    scientist = AutonomousScientist(
        world_model=kernel,
        experiments=ContractExperimentCoordinator(),
        perception=EmptyPerception(),
    )
    transition = InteractionTransition(
        transition_id="transition-1",
        before=_observation("obs-0", 0),
        action=GroundAction("A1", ()),
        outcome=RawActionOutcome(
            success=False,
            payload=freeze_raw_value({"error": "rejected"}),
        ),
        after=_observation("obs-1", 1),
    )

    update = scientist.observe_transition(transition)

    assert update.new_evidence_ids
    assert kernel.contract_hypotheses() == ()


def test_autonomous_scientist_has_no_benchmark_semantic_action_categories() -> None:
    import ecsa.autonomy.scientist as module

    source = inspect.getsource(module)
    forbidden = (
        "ActionRole",
        "PROBE_BINARY",
        "OBSERVE_UNARY",
        "ACQUIRE",
        "PLACE",
        "_ROLE_MAP",
        "Reactor",
        "Crystal",
    )

    assert not any(token in source for token in forbidden)
