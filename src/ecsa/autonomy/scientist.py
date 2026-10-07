from __future__ import annotations

from ecsa.contracts import TheoryPosterior

from ..world_model.contracts import (
    FrozenRawValue,
    InteractionTransition,
    RawActionSchema,
    RawObservation,
)
from ..world_model.experiments import (
    ContractExperiment,
    ContractExperimentCoordinator,
    ExperimentBudget,
)
from ..world_model.grounding import InteractionGrounder
from ..world_model.kernel import (
    WorldModelAcquisitionKernel,
    WorldModelUpdate,
)
from ..world_model.perception.base import PerceptionFrontend


class AutonomousScientist:
    """Coordinate raw-world experiments through induced world contracts."""

    def __init__(
        self,
        *,
        world_model: WorldModelAcquisitionKernel,
        experiments: ContractExperimentCoordinator,
        perception: PerceptionFrontend,
    ) -> None:
        if not isinstance(world_model, WorldModelAcquisitionKernel):
            raise TypeError(
                "world_model must be WorldModelAcquisitionKernel"
            )
        if not isinstance(
            experiments,
            ContractExperimentCoordinator,
        ):
            raise TypeError(
                "experiments must be ContractExperimentCoordinator"
            )
        self.world_model = world_model
        self.experiments = experiments
        self.perception = perception
        self._grounder = InteractionGrounder()

    def choose_experiment(
        self,
        observation: RawObservation,
        actions: tuple[RawActionSchema, ...],
        goal: FrozenRawValue,
        budget: ExperimentBudget,
    ) -> ContractExperiment:
        if not isinstance(observation, RawObservation):
            raise TypeError("observation must be RawObservation")
        if not isinstance(goal, FrozenRawValue):
            raise TypeError("goal must be FrozenRawValue")
        del goal

        perception = self.perception.perceive(observation)
        contracts = self.world_model.contract_hypotheses()
        candidates = self.experiments.propose(
            contracts=contracts,
            action_schemas=actions,
            perception=perception,
            budget=budget,
        )
        if not candidates:
            raise ValueError(
                "no contract experiment can be grounded from the public surface"
            )
        if len(contracts) < 2:
            return self.experiments.select_bootstrap(candidates)

        probability = 1.0 / len(contracts)
        posterior = TheoryPosterior(
            tuple(
                (contract.contract_id, probability)
                for contract in contracts
            )
        )
        return self.experiments.select(
            posterior=posterior,
            experiments=candidates,
        )

    def observe_transition(
        self,
        transition: InteractionTransition,
    ) -> WorldModelUpdate:
        if not isinstance(transition, InteractionTransition):
            raise TypeError(
                "transition must be InteractionTransition"
            )
        self.experiments.record_outcome(
            transition.action,
            success=transition.outcome.success,
        )
        before = self.perception.perceive(transition.before)
        after = self.perception.perceive(transition.after)
        grounding = self._grounder.observe(
            transition,
            before,
            after,
        )
        return self.world_model.observe_grounding(grounding)
