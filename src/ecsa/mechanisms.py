from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from .contracts import Scalar


class MechanismKind(StrEnum):
    PROGRAM = "program"
    CAUSAL_MODEL = "causal_model"
    STATE_FEATURE = "state_feature"
    SYMBOLIC_RULE = "symbolic_rule"


class MechanismStatus(StrEnum):
    CANDIDATE = "candidate"
    QUALIFIED = "qualified"
    ADMITTED = "admitted"
    DEPRECATED = "deprecated"


class MechanismTransferStatus(StrEnum):
    SHARED = "shared"
    CONTEXT_SPECIALIZED = "context_specialized"
    SPLIT = "split"


class MechanismRelationKind(StrEnum):
    ABSTRACTED_FROM = "abstracted_from"
    SPECIALIZES = "specializes"
    SUPERSEDES = "supersedes"
    COMPOSED_OF = "composed_of"


def _validate_ids(name: str, values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be immutable")
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"{name} values must be nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} values must be unique")


@dataclass(frozen=True)
class MechanismRelation:
    kind: MechanismRelationKind
    target_mechanism_id: str
    target_version: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MechanismRelationKind):
            raise ValueError("typed mechanism relation kind required")
        if not self.target_mechanism_id:
            raise ValueError("target_mechanism_id is required")
        if type(self.target_version) is not int or self.target_version < 1:
            raise ValueError("target_version must be a positive integer")


@dataclass(frozen=True)
class MechanismScope:
    context_ids: tuple[str, ...] = ()
    regime_ids: tuple[str, ...] = ()
    domain_ids: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()
    required_assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_ids("context_ids", self.context_ids)
        _validate_ids("regime_ids", self.regime_ids)
        _validate_ids("domain_ids", self.domain_ids)
        _validate_ids("task_ids", self.task_ids)
        _validate_ids("required_assumptions", self.required_assumptions)

    def matches(self, context: "ApplicabilityContext") -> bool:
        def matches_dimension(
            allowed: tuple[str, ...],
            actual: str | None,
        ) -> bool:
            if not allowed:
                return True
            return actual is not None and actual in allowed

        return (
            matches_dimension(self.context_ids, context.context_id)
            and matches_dimension(self.regime_ids, context.regime_id)
            and matches_dimension(self.domain_ids, context.domain_id)
            and matches_dimension(self.task_ids, context.task_id)
            and set(self.required_assumptions).issubset(context.assumptions)
        )


@dataclass(frozen=True)
class ApplicabilityContext:
    context_id: str | None = None
    regime_id: str | None = None
    domain_id: str | None = None
    task_id: str | None = None
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("context_id", self.context_id),
            ("regime_id", self.regime_id),
            ("domain_id", self.domain_id),
            ("task_id", self.task_id),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be None or a nonempty string")
        _validate_ids("assumptions", self.assumptions)


@dataclass(frozen=True)
class MechanismRecord:
    mechanism_id: str
    version: int
    kind: MechanismKind
    status: MechanismStatus
    representation_artifact_id: str
    scope: MechanismScope
    transfer: MechanismTransferStatus
    parameter_bindings: tuple[tuple[str, Scalar], ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    provenance_artifact_ids: tuple[str, ...] = ()
    relations: tuple[MechanismRelation, ...] = ()

    def __post_init__(self) -> None:
        if not self.mechanism_id:
            raise ValueError("mechanism_id is required")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("version must be a positive integer")
        if not isinstance(self.kind, MechanismKind):
            raise ValueError("typed mechanism kind required")
        if not isinstance(self.status, MechanismStatus):
            raise ValueError("typed mechanism status required")
        if not self.representation_artifact_id:
            raise ValueError("representation_artifact_id is required")
        if not isinstance(self.scope, MechanismScope):
            raise ValueError("typed mechanism scope required")
        if not isinstance(self.transfer, MechanismTransferStatus):
            raise ValueError("typed transfer status required")

        if not isinstance(self.parameter_bindings, tuple):
            raise ValueError("parameter_bindings must be immutable")
        parameter_names: set[str] = set()
        for name, value in self.parameter_bindings:
            if not isinstance(name, str) or not name or name in parameter_names:
                raise ValueError("parameter names must be unique and nonempty")
            parameter_names.add(name)
            if not isinstance(value, (bool, int, float, str)):
                raise ValueError("unsupported mechanism parameter value")

        _validate_ids("supporting_evidence_ids", self.supporting_evidence_ids)
        _validate_ids("contradicting_evidence_ids", self.contradicting_evidence_ids)
        _validate_ids("provenance_artifact_ids", self.provenance_artifact_ids)
        if set(self.supporting_evidence_ids) & set(self.contradicting_evidence_ids):
            raise ValueError(
                "the same evidence cannot be both supporting and contradicting"
            )

        if not isinstance(self.relations, tuple) or not all(
            isinstance(relation, MechanismRelation)
            for relation in self.relations
        ):
            raise ValueError("relations must be immutable typed values")
        relation_keys = tuple(
            (
                relation.kind,
                relation.target_mechanism_id,
                relation.target_version,
            )
            for relation in self.relations
        )
        if len(set(relation_keys)) != len(relation_keys):
            raise ValueError("mechanism relations must be unique")

        if (
            self.status is MechanismStatus.ADMITTED
            and self.transfer is MechanismTransferStatus.SPLIT
            and not (
                self.scope.context_ids
                or self.scope.regime_ids
                or self.scope.domain_ids
                or self.scope.task_ids
            )
        ):
            raise ValueError(
                "an admitted split mechanism requires explicit applicability scope"
            )

    @property
    def identity(self) -> tuple[str, int]:
        return self.mechanism_id, self.version

    def to_wire(self) -> dict:
        return {
            "schema": "ecsa.mechanism-record.v1",
            "mechanism_id": self.mechanism_id,
            "version": self.version,
            "kind": self.kind.value,
            "status": self.status.value,
            "representation_artifact_id": self.representation_artifact_id,
            "scope": {
                "context_ids": list(self.scope.context_ids),
                "regime_ids": list(self.scope.regime_ids),
                "domain_ids": list(self.scope.domain_ids),
                "task_ids": list(self.scope.task_ids),
                "required_assumptions": list(
                    self.scope.required_assumptions
                ),
            },
            "transfer": self.transfer.value,
            "parameter_bindings": [
                [name, value]
                for name, value in self.parameter_bindings
            ],
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(
                self.contradicting_evidence_ids
            ),
            "provenance_artifact_ids": list(self.provenance_artifact_ids),
            "relations": [
                {
                    "kind": relation.kind.value,
                    "target_mechanism_id": relation.target_mechanism_id,
                    "target_version": relation.target_version,
                }
                for relation in self.relations
            ],
        }

    @classmethod
    def from_wire(cls, value: dict) -> "MechanismRecord":
        if value.get("schema") != "ecsa.mechanism-record.v1":
            raise ValueError("unsupported mechanism record schema")
        scope = value["scope"]
        return cls(
            mechanism_id=value["mechanism_id"],
            version=int(value["version"]),
            kind=MechanismKind(value["kind"]),
            status=MechanismStatus(value["status"]),
            representation_artifact_id=value["representation_artifact_id"],
            scope=MechanismScope(
                context_ids=tuple(scope["context_ids"]),
                regime_ids=tuple(scope["regime_ids"]),
                domain_ids=tuple(scope["domain_ids"]),
                task_ids=tuple(scope["task_ids"]),
                required_assumptions=tuple(
                    scope["required_assumptions"]
                ),
            ),
            transfer=MechanismTransferStatus(value["transfer"]),
            parameter_bindings=tuple(
                (item[0], item[1])
                for item in value.get("parameter_bindings", ())
            ),
            supporting_evidence_ids=tuple(
                value["supporting_evidence_ids"]
            ),
            contradicting_evidence_ids=tuple(
                value["contradicting_evidence_ids"]
            ),
            provenance_artifact_ids=tuple(
                value["provenance_artifact_ids"]
            ),
            relations=tuple(
                MechanismRelation(
                    MechanismRelationKind(item["kind"]),
                    item["target_mechanism_id"],
                    int(item["target_version"]),
                )
                for item in value["relations"]
            ),
        )


@dataclass(frozen=True)
class MechanismLineage:
    record: MechanismRecord
    representation_artifact_id: str
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    provenance_artifact_ids: tuple[str, ...]
    execution_kind: str
    prior_version: tuple[str, int] | None = None


@runtime_checkable
class MechanismRepository(Protocol):
    def admit(self, record: MechanismRecord) -> MechanismRecord: ...

    def get(
        self,
        mechanism_id: str,
        version: int | None = None,
    ) -> MechanismRecord | None: ...

    def find_applicable(
        self,
        context: ApplicabilityContext,
    ) -> tuple[MechanismRecord, ...]: ...

    def lineage(
        self,
        mechanism_id: str,
        version: int | None = None,
    ) -> MechanismLineage | None: ...
