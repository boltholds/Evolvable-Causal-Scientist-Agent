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


class EpistemicStatus(StrEnum):
    CANDIDATE = "candidate"
    QUALIFIED = "qualified"
    ADMITTED = "admitted"
    DEPRECATED = "deprecated"


MechanismStatus = EpistemicStatus


class TransferStatus(StrEnum):
    SHARED = "shared"
    CONTEXT_SPECIALIZED = "context_specialized"
    SPLIT = "split"


MechanismTransferStatus = TransferStatus


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


@dataclass(frozen=True, order=True)
class MechanismVersionRef:
    mechanism_id: str
    version: int

    def __post_init__(self) -> None:
        if not self.mechanism_id:
            raise ValueError("mechanism_id is required")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("version must be a positive integer")


@dataclass(frozen=True, init=False)
class MechanismRelation:
    kind: MechanismRelationKind
    target: MechanismVersionRef

    def __init__(
        self,
        kind: MechanismRelationKind,
        target: MechanismVersionRef | str,
        target_version: int | None = None,
    ) -> None:
        if not isinstance(kind, MechanismRelationKind):
            raise ValueError("typed mechanism relation kind required")
        if isinstance(target, MechanismVersionRef):
            if target_version is not None:
                raise ValueError(
                    "target_version is redundant with MechanismVersionRef"
                )
            ref = target
        else:
            if target_version is None:
                raise ValueError("target_version is required")
            ref = MechanismVersionRef(target, target_version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "target", ref)

    @property
    def target_mechanism_id(self) -> str:
        return self.target.mechanism_id

    @property
    def target_version(self) -> int:
        return self.target.version


@dataclass(frozen=True)
class MechanismScope:
    context_ids: tuple[str, ...]
    regime_ids: tuple[str, ...] = ()
    domain_ids: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()
    required_assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_ids("context_ids", self.context_ids)
        if not self.context_ids:
            raise ValueError(
                "mechanism scope requires at least one explicit context"
            )
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
            if value is not None and (
                not isinstance(value, str) or not value
            ):
                raise ValueError(
                    f"{name} must be None or a nonempty string"
                )
        _validate_ids("assumptions", self.assumptions)


@dataclass(frozen=True, init=False)
class MechanismRecord:
    mechanism_id: str
    version: int
    kind: MechanismKind
    epistemic_status: EpistemicStatus
    representation_artifact: str
    scope: MechanismScope
    transfer_status: TransferStatus
    parameters: tuple[tuple[str, Scalar], ...]
    supporting_evidence: tuple[str, ...]
    contradicting_evidence: tuple[str, ...]
    provenance: tuple[str, ...]
    relations: tuple[MechanismRelation, ...]

    def __init__(
        self,
        mechanism_id: str,
        version: int,
        kind: MechanismKind,
        epistemic_status: EpistemicStatus | None = None,
        representation_artifact: str | None = None,
        scope: MechanismScope | None = None,
        transfer_status: TransferStatus | None = None,
        parameters: tuple[tuple[str, Scalar], ...] | None = None,
        supporting_evidence: tuple[str, ...] | None = None,
        contradicting_evidence: tuple[str, ...] | None = None,
        provenance: tuple[str, ...] | None = None,
        relations: tuple[MechanismRelation, ...] = (),
        *,
        status: EpistemicStatus | None = None,
        representation_artifact_id: str | None = None,
        transfer: TransferStatus | None = None,
        parameter_bindings: tuple[tuple[str, Scalar], ...] | None = None,
        supporting_evidence_ids: tuple[str, ...] | None = None,
        contradicting_evidence_ids: tuple[str, ...] | None = None,
        provenance_artifact_ids: tuple[str, ...] | None = None,
    ) -> None:
        def choose(name: str, canonical, alias, default=None):
            if canonical is not None and alias is not None:
                if canonical != alias:
                    raise ValueError(
                        f"conflicting canonical/compatibility values for {name}"
                    )
                return canonical
            if canonical is not None:
                return canonical
            if alias is not None:
                return alias
            return default

        resolved_status = choose(
            "epistemic_status",
            epistemic_status,
            status,
        )
        resolved_representation = choose(
            "representation_artifact",
            representation_artifact,
            representation_artifact_id,
        )
        resolved_transfer = choose(
            "transfer_status",
            transfer_status,
            transfer,
        )
        resolved_parameters = choose(
            "parameters",
            parameters,
            parameter_bindings,
            (),
        )
        resolved_support = choose(
            "supporting_evidence",
            supporting_evidence,
            supporting_evidence_ids,
            (),
        )
        resolved_contradictions = choose(
            "contradicting_evidence",
            contradicting_evidence,
            contradicting_evidence_ids,
            (),
        )
        resolved_provenance = choose(
            "provenance",
            provenance,
            provenance_artifact_ids,
            (),
        )

        object.__setattr__(self, "mechanism_id", mechanism_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "epistemic_status",
            resolved_status,
        )
        object.__setattr__(
            self,
            "representation_artifact",
            resolved_representation,
        )
        object.__setattr__(self, "scope", scope)
        object.__setattr__(
            self,
            "transfer_status",
            resolved_transfer,
        )
        object.__setattr__(
            self,
            "parameters",
            resolved_parameters,
        )
        object.__setattr__(
            self,
            "supporting_evidence",
            resolved_support,
        )
        object.__setattr__(
            self,
            "contradicting_evidence",
            resolved_contradictions,
        )
        object.__setattr__(
            self,
            "provenance",
            resolved_provenance,
        )
        object.__setattr__(self, "relations", relations)
        self._validate()

    def _validate(self) -> None:
        if not self.mechanism_id:
            raise ValueError("mechanism_id is required")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("version must be a positive integer")
        if not isinstance(self.kind, MechanismKind):
            raise ValueError("typed mechanism kind required")
        if not isinstance(self.epistemic_status, EpistemicStatus):
            raise ValueError("typed epistemic status required")
        if not self.representation_artifact:
            raise ValueError("representation_artifact is required")
        if not isinstance(self.scope, MechanismScope):
            raise ValueError("typed mechanism scope required")
        if not isinstance(self.transfer_status, TransferStatus):
            raise ValueError("typed transfer status required")

        if not isinstance(self.parameters, tuple):
            raise ValueError("parameters must be immutable")
        parameter_names: set[str] = set()
        for name, value in self.parameters:
            if (
                not isinstance(name, str)
                or not name
                or name in parameter_names
            ):
                raise ValueError(
                    "parameter names must be unique and nonempty"
                )
            parameter_names.add(name)
            if not isinstance(value, (bool, int, float, str)):
                raise ValueError(
                    "unsupported mechanism parameter value"
                )

        _validate_ids(
            "supporting_evidence",
            self.supporting_evidence,
        )
        _validate_ids(
            "contradicting_evidence",
            self.contradicting_evidence,
        )
        _validate_ids("provenance", self.provenance)
        if set(self.supporting_evidence) & set(
            self.contradicting_evidence
        ):
            raise ValueError(
                "the same evidence cannot be both supporting and contradicting"
            )

        if not isinstance(self.relations, tuple) or not all(
            isinstance(relation, MechanismRelation)
            for relation in self.relations
        ):
            raise ValueError(
                "relations must be immutable typed values"
            )
        relation_keys = tuple(
            (relation.kind, relation.target)
            for relation in self.relations
        )
        if len(set(relation_keys)) != len(relation_keys):
            raise ValueError("mechanism relations must be unique")

    @property
    def ref(self) -> MechanismVersionRef:
        return MechanismVersionRef(
            self.mechanism_id,
            self.version,
        )

    @property
    def identity(self) -> tuple[str, int]:
        return self.mechanism_id, self.version

    @property
    def status(self) -> EpistemicStatus:
        return self.epistemic_status

    @property
    def representation_artifact_id(self) -> str:
        return self.representation_artifact

    @property
    def transfer(self) -> TransferStatus:
        return self.transfer_status

    @property
    def parameter_bindings(self) -> tuple[tuple[str, Scalar], ...]:
        return self.parameters

    @property
    def supporting_evidence_ids(self) -> tuple[str, ...]:
        return self.supporting_evidence

    @property
    def contradicting_evidence_ids(self) -> tuple[str, ...]:
        return self.contradicting_evidence

    @property
    def provenance_artifact_ids(self) -> tuple[str, ...]:
        return self.provenance

    def to_wire(self) -> dict:
        return {
            "schema": "ecsa.mechanism-record.v1",
            "mechanism_id": self.mechanism_id,
            "version": self.version,
            "kind": self.kind.value,
            "epistemic_status": self.epistemic_status.value,
            "representation_artifact": self.representation_artifact,
            "scope": {
                "context_ids": list(self.scope.context_ids),
                "regime_ids": list(self.scope.regime_ids),
                "domain_ids": list(self.scope.domain_ids),
                "task_ids": list(self.scope.task_ids),
                "required_assumptions": list(
                    self.scope.required_assumptions
                ),
            },
            "transfer_status": self.transfer_status.value,
            "parameters": [
                [name, value]
                for name, value in self.parameters
            ],
            "supporting_evidence": list(
                self.supporting_evidence
            ),
            "contradicting_evidence": list(
                self.contradicting_evidence
            ),
            "provenance": list(self.provenance),
            "relations": [
                {
                    "kind": relation.kind.value,
                    "target": {
                        "mechanism_id": relation.target.mechanism_id,
                        "version": relation.target.version,
                    },
                }
                for relation in self.relations
            ],
        }

    @classmethod
    def from_wire(cls, value: dict) -> "MechanismRecord":
        if value.get("schema") != "ecsa.mechanism-record.v1":
            raise ValueError(
                "unsupported mechanism record schema"
            )
        scope = value["scope"]

        epistemic_status = value.get(
            "epistemic_status",
            value.get("status"),
        )
        representation = value.get(
            "representation_artifact",
            value.get("representation_artifact_id"),
        )
        transfer_status = value.get(
            "transfer_status",
            value.get("transfer"),
        )
        parameters = value.get(
            "parameters",
            value.get("parameter_bindings", ()),
        )
        supporting = value.get(
            "supporting_evidence",
            value.get("supporting_evidence_ids", ()),
        )
        contradicting = value.get(
            "contradicting_evidence",
            value.get("contradicting_evidence_ids", ()),
        )
        provenance = value.get(
            "provenance",
            value.get("provenance_artifact_ids", ()),
        )

        relations = []
        for item in value["relations"]:
            target = item.get("target")
            if target is None:
                target = {
                    "mechanism_id": item["target_mechanism_id"],
                    "version": item["target_version"],
                }
            relations.append(
                MechanismRelation(
                    MechanismRelationKind(item["kind"]),
                    MechanismVersionRef(
                        target["mechanism_id"],
                        int(target["version"]),
                    ),
                )
            )

        return cls(
            mechanism_id=value["mechanism_id"],
            version=int(value["version"]),
            kind=MechanismKind(value["kind"]),
            epistemic_status=EpistemicStatus(
                epistemic_status
            ),
            representation_artifact=representation,
            scope=MechanismScope(
                context_ids=tuple(scope["context_ids"]),
                regime_ids=tuple(scope.get("regime_ids", ())),
                domain_ids=tuple(scope.get("domain_ids", ())),
                task_ids=tuple(scope.get("task_ids", ())),
                required_assumptions=tuple(
                    scope["required_assumptions"]
                ),
            ),
            transfer_status=TransferStatus(
                transfer_status
            ),
            parameters=tuple(
                (item[0], item[1])
                for item in parameters
            ),
            supporting_evidence=tuple(supporting),
            contradicting_evidence=tuple(contradicting),
            provenance=tuple(provenance),
            relations=tuple(relations),
        )


@dataclass(frozen=True)
class MechanismLineage:
    mechanism: MechanismVersionRef
    record: MechanismRecord
    representation_artifact_id: str
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    provenance_artifact_ids: tuple[str, ...]
    relations: tuple[MechanismRelation, ...]
    related_mechanisms: tuple[MechanismVersionRef, ...]
    execution_kind: str
    prior_version: tuple[str, int] | None = None


@runtime_checkable
class MechanismRepository(Protocol):
    def admit(self, record: MechanismRecord) -> MechanismRecord: ...

    def get(
        self,
        mechanism: MechanismVersionRef | str,
        version: int | None = None,
    ) -> MechanismRecord | None: ...

    def find_applicable(
        self,
        context: ApplicabilityContext,
    ) -> tuple[MechanismRecord, ...]: ...

    def find_transfer_candidates(
        self,
        context: ApplicabilityContext,
    ) -> tuple[MechanismRecord, ...]: ...

    def lineage(
        self,
        mechanism: MechanismVersionRef | str,
        version: int | None = None,
    ) -> MechanismLineage | None: ...
