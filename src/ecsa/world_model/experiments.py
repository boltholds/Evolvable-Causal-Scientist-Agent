from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite

from ecsa.contracts import (
    PredictiveDistribution,
    TheoryPosterior,
)
from ecsa.science import ScienceKernel

from .affordances import ActiveAffordanceLearner
from .applicability import ActiveApplicabilityLearner
from .preconditions import observable_conditions
from .preconditions import ActivePreconditionLearner
from .candidate_generation import (
    ExperimentHistory,
    StructuralCandidateGenerator,
)
from .contracts import (
    FrozenRawValue,
    GroundAction,
    RawActionSchema,
)
from .hypotheses import WorldContractHypothesis
from .perception.base import PerceptualObservation


class ExperimentSelectionMode(StrEnum):
    STRUCTURAL = "structural"
    WORLD_CONTRACT_EIG = "world_contract_eig"
    APPLICABILITY_EIG = "applicability_eig"


@dataclass(frozen=True)
class ExperimentBudget:
    max_ground_actions: int

    def __post_init__(self) -> None:
        if (
            type(self.max_ground_actions) is not int
            or self.max_ground_actions < 1
        ):
            raise ValueError("max_ground_actions must be positive")


@dataclass(frozen=True)
class ContractOutcomePrediction:
    contract_id: str
    experiment_id: str
    success_probability: float

    def __post_init__(self) -> None:
        if not self.contract_id or not self.experiment_id:
            raise ValueError("prediction identifiers are required")
        if (
            not isinstance(self.success_probability, (int, float))
            or isinstance(self.success_probability, bool)
            or not isfinite(float(self.success_probability))
            or not 0.0 <= float(self.success_probability) <= 1.0
        ):
            raise ValueError(
                "success_probability must be finite in [0,1]"
            )

    def as_predictive_distribution(
        self,
    ) -> PredictiveDistribution:
        success = round(float(self.success_probability), 12)
        failure = round(1.0 - success, 12)
        return PredictiveDistribution(
            theory_id=self.contract_id,
            experiment_id=self.experiment_id,
            probabilities=(
                (("success",), success),
                (("failure",), failure),
            ),
        )


@dataclass(frozen=True)
class ContractExperiment:
    experiment_id: str
    action: GroundAction
    predictions: tuple[ContractOutcomePrediction, ...]

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not isinstance(self.action, GroundAction):
            raise ValueError("contract experiment requires GroundAction")
        if not isinstance(self.predictions, tuple) or not all(
            isinstance(value, ContractOutcomePrediction)
            for value in self.predictions
        ):
            raise ValueError("predictions must be immutable typed values")


def _experiment_id(action: GroundAction) -> str:
    material = (
        action.schema_id,
        tuple(
            repr(argument.thaw())
            for argument in action.arguments
        ),
    )
    return (
        "contract-experiment:"
        + sha256(repr(material).encode()).hexdigest()
    )


def _argument_member_ref(argument: FrozenRawValue) -> str:
    value = argument.thaw()
    return value if isinstance(value, str) else str(value)


def _argument_type_ids(
    contract: WorldContractHypothesis,
    arguments: tuple[FrozenRawValue, ...],
) -> tuple[str, ...] | None:
    resolved: list[str] = []
    for argument in arguments:
        member_ref = _argument_member_ref(argument)
        matches = sorted(
            entity_type.hypothesis_id
            for entity_type in contract.entity_types
            if member_ref in entity_type.member_refs
        )
        if len(matches) != 1:
            return None
        resolved.append(matches[0])
    return tuple(resolved)


def _success_probability(
    contract: WorldContractHypothesis,
    action: GroundAction,
) -> float:
    argument_types = _argument_type_ids(
        contract,
        action.arguments,
    )
    if argument_types is None:
        return 0.5
    actions = tuple(
        candidate
        for candidate in contract.actions
        if candidate.source_schema_id == action.schema_id
        and len(candidate.parameter_type_ids)
        == len(action.arguments)
        and candidate.parameter_type_ids == argument_types
    )
    if not actions:
        return 0.5
    action_ids = {
        candidate.hypothesis_id
        for candidate in actions
    }
    affordances = tuple(
        value
        for value in contract.affordances
        if value.action_schema_id in action_ids
        and value.argument_type_ids == argument_types
    )
    if not affordances:
        return 0.5
    return max(
        affordances,
        key=lambda value: (
            float(value.confidence),
            value.hypothesis_id,
        ),
    ).success_probability


class ContractExperimentCoordinator:
    def __init__(
        self,
        *,
        science: ScienceKernel | None = None,
        history: ExperimentHistory | None = None,
        candidates: StructuralCandidateGenerator | None = None,
        affordances: ActiveAffordanceLearner | None = None,
        preconditions: ActivePreconditionLearner | None = None,
        applicability: ActiveApplicabilityLearner | None = None,
        use_affordance_scoring: bool = True,
        use_applicability_selection: bool = True,
    ) -> None:
        if type(use_affordance_scoring) is not bool:
            raise TypeError("use_affordance_scoring must be bool")
        if type(use_applicability_selection) is not bool:
            raise TypeError("use_applicability_selection must be bool")
        self.use_affordance_scoring = use_affordance_scoring
        self.use_applicability_selection = use_applicability_selection
        self.last_selection_mode = ExperimentSelectionMode.STRUCTURAL
        self.applicability = applicability if applicability is not None else ActiveApplicabilityLearner()
        self.science = science or ScienceKernel()
        self.affordances = affordances if affordances is not None else ActiveAffordanceLearner()
        self.preconditions = preconditions if preconditions is not None else ActivePreconditionLearner()
        self.history = history or ExperimentHistory()
        self.candidates = candidates or StructuralCandidateGenerator(
            history=self.history,
        )

    def record_outcome(
        self,
        action: GroundAction,
        *,
        success: bool,
        state_id: str | None = None,
        before: PerceptualObservation | None = None,
    ) -> None:
        self.history.record(action, success=success)
        if before is not None and state_id is None:
            # Equality of public state is useful contrast evidence, not proof
            # of equality of unobserved latent world state.
            conditions = tuple(sorted(observable_conditions(before), key=repr))
            state_id = sha256(repr(conditions).encode()).hexdigest()
        self.affordances.observe(action, success=success, state_id=state_id)
        if before is not None:
            self.preconditions.observe(action, success=success, before=before)
            self.applicability.observe(action, success=success, before=before)

    def select_bootstrap(
        self,
        experiments: tuple[ContractExperiment, ...],
    ) -> ContractExperiment:
        if not experiments:
            raise ValueError("at least one contract experiment is required")
        self.last_selection_mode = ExperimentSelectionMode.STRUCTURAL
        return min(
            experiments,
            key=lambda experiment: (
                -self.history.bootstrap_score(
                    experiment.action,
                )[0],
                -self.history.bootstrap_score(
                    experiment.action,
                )[1],
                -(self.affordances.score(experiment.action)[0] if self.use_affordance_scoring else 0.0),
                -(self.affordances.score(experiment.action)[1] if self.use_affordance_scoring else 0.0),
                experiment.action.schema_id,
                experiment.experiment_id,
            ),
        )

    def propose(
        self,
        *,
        contracts: tuple[WorldContractHypothesis, ...],
        action_schemas: tuple[RawActionSchema, ...],
        perception: PerceptualObservation,
        budget: ExperimentBudget,
    ) -> tuple[ContractExperiment, ...]:
        if not isinstance(contracts, tuple) or not all(
            isinstance(value, WorldContractHypothesis)
            for value in contracts
        ):
            raise TypeError("contracts must be immutable world contracts")
        if not isinstance(action_schemas, tuple) or not all(
            isinstance(value, RawActionSchema)
            for value in action_schemas
        ):
            raise TypeError("action_schemas must be immutable raw schemas")
        if not isinstance(perception, PerceptualObservation):
            raise TypeError("perception must be PerceptualObservation")
        if not isinstance(budget, ExperimentBudget):
            raise TypeError("budget must be ExperimentBudget")

        ground_actions = self.candidates.propose(
            action_schemas=action_schemas,
            perception=perception,
            max_ground_actions=budget.max_ground_actions,
        )
        experiments: list[ContractExperiment] = []
        for action in ground_actions:
            experiment_id = _experiment_id(action)
            predictions = tuple(
                ContractOutcomePrediction(
                    contract_id=contract.contract_id,
                    experiment_id=experiment_id,
                    success_probability=_success_probability(
                        contract,
                        action,
                    ),
                )
                for contract in contracts
            )
            experiments.append(
                ContractExperiment(
                    experiment_id=experiment_id,
                    action=action,
                    predictions=predictions,
                )
            )
        return tuple(experiments)
        return tuple(experiments)

    def select_applicability(
        self,
        *,
        perception: PerceptualObservation,
        experiments: tuple[ContractExperiment, ...],
        min_information_gain_bits: float = 1e-9,
    ) -> ContractExperiment:
        """Use ScienceKernel on competing applicability rules, else explore."""
        if not experiments:
            raise ValueError("at least one contract experiment is required")
        if not isinstance(perception, PerceptualObservation):
            raise TypeError("perception must be PerceptualObservation")
        if not self.use_applicability_selection:
            return self.select_bootstrap(experiments)
        groups: dict[tuple[str, int], list[tuple[float, ContractExperiment]]] = {}
        for experiment in experiments:
            action = experiment.action
            signature = (action.schema_id, len(action.arguments))
            belief = self.applicability.belief(*signature)
            if belief is None:
                continue
            predictions = self.applicability.predictions(
                belief,
                experiment_id=experiment.experiment_id,
                action=action,
                before=perception,
            )
            score = self.science.score_experiment(belief.posterior, predictions)
            groups.setdefault(signature, []).append((score.information_gain_bits, experiment))

        # An equal information score for every grounding provides no active
        # ordering signal. Preserve the structural exploration fallback.
        informative: list[tuple[float, ContractExperiment]] = []
        for scored in groups.values():
            values = tuple(score for score, _ in scored)
            if len(values) < 2 or max(values) - min(values) <= min_information_gain_bits:
                continue
            informative.extend(
                (score, experiment)
                for score, experiment in scored
                if score > min_information_gain_bits
            )
        if not informative:
            return self.select_bootstrap(experiments)
        self.last_selection_mode = ExperimentSelectionMode.APPLICABILITY_EIG
        return min(
            informative,
            key=lambda item: (
                -item[0],
                -self.history.bootstrap_score(item[1].action)[0],
                -self.history.bootstrap_score(item[1].action)[1],
                item[1].experiment_id,
            ),
        )[1]

    def select_active(
        self,
        *,
        posterior: TheoryPosterior,
        experiments: tuple[ContractExperiment, ...],
        min_information_gain_bits: float = 1e-12,
        perception: PerceptualObservation | None = None,
    ) -> ContractExperiment:
        if not experiments:
            raise ValueError("at least one contract experiment is required")
        if (
            not isinstance(min_information_gain_bits, (int, float))
            or isinstance(min_information_gain_bits, bool)
            or float(min_information_gain_bits) < 0.0
        ):
            raise ValueError(
                "min_information_gain_bits must be nonnegative"
            )
        if any(not experiment.predictions for experiment in experiments):
            raise ValueError(
                "contract experiments require predictions for active selection"
            )

        scored = tuple(
            (
                self.science.score_experiment(
                    posterior,
                    tuple(
                        prediction.as_predictive_distribution()
                        for prediction in experiment.predictions
                    ),
                ),
                experiment,
            )
            for experiment in experiments
        )
        best_score, best_experiment = min(
            scored,
            key=lambda item: (
                -item[0].information_gain_bits,
                -self.history.bootstrap_score(
                    item[1].action,
                )[0],
                -self.history.bootstrap_score(
                    item[1].action,
                )[1],
                item[1].experiment_id,
            ),
        )
        if (
            best_score.information_gain_bits
            <= float(min_information_gain_bits)
        ):
            if perception is not None:
                return self.select_applicability(
                    perception=perception, experiments=experiments,
                )
            return self.select_bootstrap(experiments)
        self.last_selection_mode = ExperimentSelectionMode.WORLD_CONTRACT_EIG
        return best_experiment

    def select(
        self,
        *,
        posterior: TheoryPosterior,
        experiments: tuple[ContractExperiment, ...],
    ) -> ContractExperiment:
        if not experiments:
            raise ValueError("at least one contract experiment is required")
        if any(not experiment.predictions for experiment in experiments):
            raise ValueError(
                "contract experiments require predictions for selection"
            )
        selected = self.science.select_experiment(
            posterior,
            tuple(
                tuple(
                    prediction.as_predictive_distribution()
                    for prediction in experiment.predictions
                )
                for experiment in experiments
            ),
        )
        self.last_selection_mode = ExperimentSelectionMode.WORLD_CONTRACT_EIG
        return next(
            experiment
            for experiment in experiments
            if experiment.experiment_id == selected.experiment_id
        )
