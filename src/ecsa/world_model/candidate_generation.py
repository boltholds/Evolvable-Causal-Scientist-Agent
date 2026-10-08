from __future__ import annotations

from collections import Counter
from itertools import chain, islice, product
from math import prod

from .contracts import (
    FrozenRawValue,
    GroundAction,
    RawActionSchema,
)
from .perception.base import PerceptualObservation


SchemaSignature = tuple[str, int]
ActionSignature = tuple[str, int, tuple[str, ...]]


def _entity_argument_candidates(
    perception: PerceptualObservation,
) -> tuple[FrozenRawValue, ...]:
    from .contracts import freeze_raw_value

    return tuple(
        freeze_raw_value(
            entity.source_identity or entity.local_ref
        )
        for entity in sorted(
            perception.entities,
            key=lambda value: value.local_ref,
        )
    )


def _schema_signature(action: GroundAction) -> SchemaSignature:
    return (action.schema_id, len(action.arguments))


def _action_signature(action: GroundAction) -> ActionSignature:
    return (
        action.schema_id,
        len(action.arguments),
        tuple(repr(argument.thaw()) for argument in action.arguments),
    )


class ExperimentHistory:
    """Structural history for contract bootstrapping.

    Names identify schemas within the current environment but are never
    interpreted semantically.
    """

    def __init__(self) -> None:
        self._schema_successes: Counter[SchemaSignature] = Counter()
        self._schema_failures: Counter[SchemaSignature] = Counter()
        self._action_attempts: Counter[ActionSignature] = Counter()

    def record(
        self,
        action: GroundAction,
        *,
        success: bool,
    ) -> None:
        if not isinstance(action, GroundAction):
            raise TypeError("action must be GroundAction")
        if type(success) is not bool:
            raise TypeError("success must be bool")
        schema = _schema_signature(action)
        if success:
            self._schema_successes[schema] += 1
        else:
            self._schema_failures[schema] += 1
        self._action_attempts[_action_signature(action)] += 1

    def schema_attempts(
        self,
        schema_id: str,
        arity: int,
    ) -> int:
        signature = (schema_id, arity)
        return (
            self._schema_successes[signature]
            + self._schema_failures[signature]
        )

    def schema_success_probability(
        self,
        schema_id: str,
        arity: int,
    ) -> float:
        signature = (schema_id, arity)
        return (
            1.0 + self._schema_successes[signature]
        ) / (
            2.0 + self._schema_successes[signature]
            + self._schema_failures[signature]
        )

    def schema_uncertainty(
        self,
        schema_id: str,
        arity: int,
    ) -> float:
        signature = (schema_id, arity)
        alpha = 1.0 + self._schema_successes[signature]
        beta = 1.0 + self._schema_failures[signature]
        total = alpha + beta
        return (
            alpha
            * beta
            / (total * total * (total + 1.0))
        )

    def action_attempts(self, action: GroundAction) -> int:
        return self._action_attempts[_action_signature(action)]

    def bootstrap_score(
        self,
        action: GroundAction,
    ) -> tuple[float, float]:
        return (
            self.schema_uncertainty(
                action.schema_id,
                len(action.arguments),
            ),
            1.0 / (1.0 + self.action_attempts(action)),
        )


class StructuralCandidateGenerator:
    """Generate a bounded, schema-stratified ground-action set."""

    def __init__(
        self,
        *,
        history: ExperimentHistory | None = None,
    ) -> None:
        self.history = history or ExperimentHistory()

    def propose(
        self,
        *,
        action_schemas: tuple[RawActionSchema, ...],
        perception: PerceptualObservation,
        max_ground_actions: int,
    ) -> tuple[GroundAction, ...]:
        if type(max_ground_actions) is not int or max_ground_actions < 1:
            raise ValueError("max_ground_actions must be positive")
        if not isinstance(action_schemas, tuple) or not all(
            isinstance(value, RawActionSchema)
            for value in action_schemas
        ):
            raise TypeError("action_schemas must be immutable raw schemas")
        if not isinstance(perception, PerceptualObservation):
            raise TypeError("perception must be PerceptualObservation")

        entity_candidates = _entity_argument_candidates(perception)
        iterators: list[tuple[RawActionSchema, object]] = []

        ordered_schemas = sorted(
            action_schemas,
            key=lambda schema: (
                -self.history.schema_uncertainty(
                    schema.schema_id,
                    len(schema.parameters),
                ),
                self.history.schema_attempts(
                    schema.schema_id,
                    len(schema.parameters),
                ),
                schema.schema_id,
            ),
        )

        for schema in ordered_schemas:
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
            if candidate_sets:
                total = prod(len(values) for values in candidate_sets)
                offset = (
                    self.history.schema_attempts(
                        schema.schema_id, len(schema.parameters)
                    ) % total
                )
                combinations = chain(
                    islice(product(*candidate_sets), offset, None),
                    islice(product(*candidate_sets), offset),
                )
            else:
                combinations = iter(((),))
            iterators.append((schema, combinations))

        actions: list[GroundAction] = []
        active = iterators
        while active and len(actions) < max_ground_actions:
            next_active: list[tuple[RawActionSchema, object]] = []
            for schema, combinations in active:
                if len(actions) >= max_ground_actions:
                    break
                try:
                    combination = next(combinations)  # type: ignore[arg-type]
                except StopIteration:
                    continue
                action = GroundAction(
                    schema_id=schema.schema_id,
                    arguments=tuple(combination),
                )
                action.validate_against(schema)
                actions.append(action)
                next_active.append((schema, combinations))
            active = next_active

        return tuple(actions)
