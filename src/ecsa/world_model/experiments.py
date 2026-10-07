from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from math import isfinite

from ecsa.contracts import (
    PredictiveDistribution,
    TheoryPosterior,
)
from ecsa.science import ScienceKernel

from .contracts import (
    FrozenRawValue,
    GroundAction,
    RawActionSchema,
    freeze_raw_value,
)
from .hypotheses import WorldContractHypothesis
from .perception.base import PerceptualObservation


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


def _entity_argument_candidates(
    perception: PerceptualObservation,
) -> tuple[FrozenRawValue, ...]:
    return tuple(
        freeze_raw_value(
            entity.source_identity or entity.local_ref
        )
        for entity in sorted(
            perception.entities,
            key=lambda value: value.local_ref,
        )
    )


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
    ) -> None:
        self.science = science or ScienceKernel()

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

        entity_candidates = _entity_argument_candidates(perception)
        streams: list[
            tuple[
                RawActionSchema,
                object,
            ]
        ] = []
        for schema in sorted(
            action_schemas,
            key=lambda value: value.schema_id,
        ):
            candidate_sets: list[tuple[FrozenRawValue, ...]] = []
            viable = True
            for parameter in schema.parameters:
                values = (
                    parameter.public_candidates
                    if parameter.public_candidates
                    else entity_candidates
                )
                if not values:
                    viable = False
                    break
                candidate_sets.append(values)
            if not viable:
                continue
            combinations = iter(
                product(*candidate_sets)
                if candidate_sets
                else ((),)
            )
            streams.append((schema, combinations))

        experiments: list[ContractExperiment] = []
        active = streams
        while active and len(experiments) < budget.max_ground_actions:
            next_active: list[
                tuple[
                    RawActionSchema,
                    object,
                ]
            ] = []
            for schema, combinations in active:
                try:
                    combination = next(combinations)  # type: ignore[arg-type]
                except StopIteration:
                    continue

                action = GroundAction(
                    schema_id=schema.schema_id,
                    arguments=tuple(combination),
                )
                action.validate_against(schema)
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
                next_active.append((schema, combinations))
                if len(experiments) >= budget.max_ground_actions:
                    break
            active = next_active
        return tuple(experiments)

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
        return next(
            experiment
            for experiment in experiments
            if experiment.experiment_id == selected.experiment_id
        )
