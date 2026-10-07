from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    ADMITTED = "admitted"


def _validate_common(
    *,
    hypothesis_id: str,
    supporting_evidence_ids: tuple[str, ...],
    contradicting_evidence_ids: tuple[str, ...],
    confidence: float,
    status: HypothesisStatus,
) -> None:
    if not hypothesis_id:
        raise ValueError("hypothesis_id is required")
    for label, values in (
        ("supporting evidence", supporting_evidence_ids),
        ("contradicting evidence", contradicting_evidence_ids),
    ):
        if not isinstance(values, tuple) or not all(
            isinstance(value, str) and value
            for value in values
        ):
            raise ValueError(f"{label} must be immutable strings")
        if len(set(values)) != len(values):
            raise ValueError(f"{label} ids must be unique")
    if set(supporting_evidence_ids) & set(contradicting_evidence_ids):
        raise ValueError(
            "supporting and contradicting evidence must be disjoint"
        )
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError("confidence must be finite in [0,1]")
    if not isinstance(status, HypothesisStatus):
        raise ValueError("typed hypothesis status required")


@dataclass(frozen=True)
class EntityTypeHypothesis:
    hypothesis_id: str
    member_refs: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    confidence: float
    status: HypothesisStatus

    def __post_init__(self) -> None:
        _validate_common(
            hypothesis_id=self.hypothesis_id,
            supporting_evidence_ids=self.supporting_evidence_ids,
            contradicting_evidence_ids=self.contradicting_evidence_ids,
            confidence=self.confidence,
            status=self.status,
        )
        if not isinstance(self.member_refs, tuple) or not all(
            isinstance(value, str) and value
            for value in self.member_refs
        ):
            raise ValueError("member refs must be immutable strings")
        if len(set(self.member_refs)) != len(self.member_refs):
            raise ValueError("member refs must be unique")


@dataclass(frozen=True)
class PredicateHypothesis:
    hypothesis_id: str
    argument_type_ids: tuple[str, ...]
    symmetric: bool
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    confidence: float
    status: HypothesisStatus

    def __post_init__(self) -> None:
        _validate_common(
            hypothesis_id=self.hypothesis_id,
            supporting_evidence_ids=self.supporting_evidence_ids,
            contradicting_evidence_ids=self.contradicting_evidence_ids,
            confidence=self.confidence,
            status=self.status,
        )
        if not isinstance(self.argument_type_ids, tuple) or not all(
            isinstance(value, str) and value
            for value in self.argument_type_ids
        ):
            raise ValueError("predicate argument type ids must be strings")
        if type(self.symmetric) is not bool:
            raise ValueError("predicate symmetric must be bool")


@dataclass(frozen=True)
class NumericFluentHypothesis:
    hypothesis_id: str
    argument_type_ids: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    confidence: float
    status: HypothesisStatus

    def __post_init__(self) -> None:
        _validate_common(
            hypothesis_id=self.hypothesis_id,
            supporting_evidence_ids=self.supporting_evidence_ids,
            contradicting_evidence_ids=self.contradicting_evidence_ids,
            confidence=self.confidence,
            status=self.status,
        )
        if not isinstance(self.argument_type_ids, tuple) or not all(
            isinstance(value, str) and value
            for value in self.argument_type_ids
        ):
            raise ValueError("numeric fluent argument type ids must be strings")


@dataclass(frozen=True)
class ArgumentRoleHypothesis:
    hypothesis_id: str
    action_schema_id: str
    parameter_index: int
    entity_type_id: str
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    confidence: float
    status: HypothesisStatus

    def __post_init__(self) -> None:
        _validate_common(
            hypothesis_id=self.hypothesis_id,
            supporting_evidence_ids=self.supporting_evidence_ids,
            contradicting_evidence_ids=self.contradicting_evidence_ids,
            confidence=self.confidence,
            status=self.status,
        )
        if not self.action_schema_id or not self.entity_type_id:
            raise ValueError("argument role references are required")
        if type(self.parameter_index) is not int or self.parameter_index < 0:
            raise ValueError("parameter_index must be nonnegative")


@dataclass(frozen=True)
class ActionSchemaHypothesis:
    hypothesis_id: str
    source_schema_id: str
    parameter_type_ids: tuple[str, ...]
    precondition_predicate_ids: tuple[str, ...]
    add_effect_predicate_ids: tuple[str, ...]
    delete_effect_predicate_ids: tuple[str, ...]
    numeric_effect_ids: tuple[str, ...]
    symmetric_parameter_groups: tuple[tuple[int, ...], ...]
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    confidence: float
    status: HypothesisStatus

    def __post_init__(self) -> None:
        _validate_common(
            hypothesis_id=self.hypothesis_id,
            supporting_evidence_ids=self.supporting_evidence_ids,
            contradicting_evidence_ids=self.contradicting_evidence_ids,
            confidence=self.confidence,
            status=self.status,
        )
        if not self.source_schema_id:
            raise ValueError("source_schema_id is required")
        if not isinstance(self.parameter_type_ids, tuple) or not all(
            isinstance(value, str) and value
            for value in self.parameter_type_ids
        ):
            raise ValueError("action parameter type ids must be strings")
        for label, values in (
            ("precondition", self.precondition_predicate_ids),
            ("add effect", self.add_effect_predicate_ids),
            ("delete effect", self.delete_effect_predicate_ids),
            ("numeric effect", self.numeric_effect_ids),
        ):
            if not isinstance(values, tuple) or not all(
                isinstance(value, str) and value
                for value in values
            ):
                raise ValueError(f"{label} ids must be immutable strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{label} ids must be unique")
        used: set[int] = set()
        for group in self.symmetric_parameter_groups:
            if (
                not isinstance(group, tuple)
                or len(group) < 2
                or len(set(group)) != len(group)
                or not all(
                    type(index) is int
                    and 0 <= index < len(self.parameter_type_ids)
                    for index in group
                )
            ):
                raise ValueError("invalid symmetric parameter group")
            if used & set(group):
                raise ValueError(
                    "symmetric parameter groups must not overlap"
                )
            used.update(group)


@dataclass(frozen=True)
class AffordanceHypothesis:
    hypothesis_id: str
    action_schema_id: str
    argument_type_ids: tuple[str, ...]
    success_probability: float
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    confidence: float
    status: HypothesisStatus

    def __post_init__(self) -> None:
        _validate_common(
            hypothesis_id=self.hypothesis_id,
            supporting_evidence_ids=self.supporting_evidence_ids,
            contradicting_evidence_ids=self.contradicting_evidence_ids,
            confidence=self.confidence,
            status=self.status,
        )
        if not self.action_schema_id:
            raise ValueError("affordance action_schema_id is required")
        if not isinstance(self.argument_type_ids, tuple) or not all(
            isinstance(value, str) and value
            for value in self.argument_type_ids
        ):
            raise ValueError("affordance argument type ids must be strings")
        if (
            not isinstance(self.success_probability, (int, float))
            or isinstance(self.success_probability, bool)
            or not isfinite(float(self.success_probability))
            or not 0.0 <= float(self.success_probability) <= 1.0
        ):
            raise ValueError(
                "affordance success_probability must be finite in [0,1]"
            )


@dataclass(frozen=True)
class WorldContractHypothesis:
    contract_id: str
    entity_types: tuple[EntityTypeHypothesis, ...]
    predicates: tuple[PredicateHypothesis, ...]
    numeric_fluents: tuple[NumericFluentHypothesis, ...]
    argument_roles: tuple[ArgumentRoleHypothesis, ...]
    actions: tuple[ActionSchemaHypothesis, ...]
    affordances: tuple[AffordanceHypothesis, ...]
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    confidence: float
    status: HypothesisStatus

    def __post_init__(self) -> None:
        _validate_common(
            hypothesis_id=self.contract_id,
            supporting_evidence_ids=self.supporting_evidence_ids,
            contradicting_evidence_ids=self.contradicting_evidence_ids,
            confidence=self.confidence,
            status=self.status,
        )
        typed_groups = (
            (self.entity_types, EntityTypeHypothesis, "entity type"),
            (self.predicates, PredicateHypothesis, "predicate"),
            (
                self.numeric_fluents,
                NumericFluentHypothesis,
                "numeric fluent",
            ),
            (
                self.argument_roles,
                ArgumentRoleHypothesis,
                "argument role",
            ),
            (self.actions, ActionSchemaHypothesis, "action"),
            (self.affordances, AffordanceHypothesis, "affordance"),
        )
        all_ids: set[str] = set()
        for values, cls, label in typed_groups:
            if not isinstance(values, tuple) or not all(
                isinstance(value, cls)
                for value in values
            ):
                raise ValueError(f"{label} values must be immutable typed")
            ids = tuple(value.hypothesis_id for value in values)
            if len(set(ids)) != len(ids):
                raise ValueError(f"{label} hypothesis ids must be unique")
            if all_ids & set(ids):
                raise ValueError("hypothesis ids must be globally unique")
            all_ids.update(ids)

        type_ids = {
            value.hypothesis_id
            for value in self.entity_types
        }
        predicate_ids = {
            value.hypothesis_id
            for value in self.predicates
        }
        numeric_ids = {
            value.hypothesis_id
            for value in self.numeric_fluents
        }
        action_ids = {
            value.hypothesis_id
            for value in self.actions
        }

        def require_types(values: tuple[str, ...]) -> None:
            if not set(values).issubset(type_ids):
                raise ValueError("hypothesis references unknown entity type")

        for predicate in self.predicates:
            require_types(predicate.argument_type_ids)
        for fluent in self.numeric_fluents:
            require_types(fluent.argument_type_ids)
        for role in self.argument_roles:
            if role.action_schema_id not in action_ids:
                raise ValueError("argument role references unknown action")
            require_types((role.entity_type_id,))
            action = next(
                value
                for value in self.actions
                if value.hypothesis_id == role.action_schema_id
            )
            if role.parameter_index >= len(action.parameter_type_ids):
                raise ValueError("argument role parameter index is out of range")
        for action in self.actions:
            require_types(action.parameter_type_ids)
            predicate_refs = (
                set(action.precondition_predicate_ids)
                | set(action.add_effect_predicate_ids)
                | set(action.delete_effect_predicate_ids)
            )
            if not predicate_refs.issubset(predicate_ids):
                raise ValueError("action references unknown predicate")
            if not set(action.numeric_effect_ids).issubset(numeric_ids):
                raise ValueError("action references unknown numeric fluent")
        for affordance in self.affordances:
            if affordance.action_schema_id not in action_ids:
                raise ValueError("affordance references unknown action")
            require_types(affordance.argument_type_ids)
