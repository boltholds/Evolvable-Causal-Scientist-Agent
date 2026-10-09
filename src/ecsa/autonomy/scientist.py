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
from ..world_model.learning_progress import ProgressEvidence
from ..world_model.relations import NumericRelationAcquisition
from ..world_model.text_relations import TextRelationAcquisition
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
        relations: NumericRelationAcquisition | None = None,
        text_relations: TextRelationAcquisition | None = None,
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
        if relations is not None and not isinstance(relations, NumericRelationAcquisition):
            raise TypeError("relations must be NumericRelationAcquisition")
        if text_relations is not None and not isinstance(text_relations, TextRelationAcquisition):
            raise TypeError("text_relations must be TextRelationAcquisition")
        self.text_relations = text_relations
        self.relations = relations
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
            return self.experiments.select_applicability(
                perception=perception,
                experiments=candidates,
            )

        probability = 1.0 / len(contracts)
        posterior = TheoryPosterior(
            tuple(
                (contract.contract_id, probability)
                for contract in contracts
            )
        )
        return self.experiments.select_active(
            posterior=posterior,
            experiments=candidates,
            perception=perception,
        )

    def observe_transition(
        self,
        transition: InteractionTransition,
    ) -> WorldModelUpdate:
        if not isinstance(transition, InteractionTransition):
            raise TypeError(
                "transition must be InteractionTransition"
            )
        before = self.perception.perceive(transition.before)
        self.experiments.record_outcome(
            transition.action,
            success=transition.outcome.success,
            before=before,
        )
        after = self.perception.perceive(transition.after)
        grounding = self._grounder.observe(
            transition,
            before,
            after,
        )
        update = self.world_model.observe_grounding(grounding)
        if self.relations is not None:
            self.relations.observe_transition(transition, before, after)
        if self.text_relations is not None:
            self.text_relations.observe_transition(transition)
        if self.experiments.use_progress_scoring:
            # Only public pre/post-action evidence: new raw observations and
            # proposed contracts are not independent scientific confirmations.
            def state_signature(perceptual):
                return tuple(sorted((
                    f"global:{f.feature_id}:{f.value!r}"
                    for f in perceptual.global_features
                ))) + tuple(sorted((
                    f"entity:{feature.feature_id}:{feature.value!r}"
                    for entity in perceptual.entities for feature in entity.features
                )))
            self.experiments.progress.observe(ProgressEvidence(
                transition_id=transition.transition_id,
                action=transition.action,
                context_signature=self.experiments.context_signature(before),
                before_signature=state_signature(before),
                after_signature=state_signature(after),
                predictive_gain=None,
                uncertainty_reduction=0.0,
                confirmed_hypothesis_ids=(),
                contradicted_hypothesis_ids=(),
                action_cost=1.0,
                independent_trial_group=None,
            ))
        return update
