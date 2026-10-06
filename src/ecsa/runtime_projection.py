from __future__ import annotations

from dataclasses import dataclass

from ecsa.adapters.dreamcoder import (
    DreamCoderRepairEngine,
    ProgramMechanismProjector,
)
from ecsa.mechanisms import (
    ApplicabilityContext,
    MechanismKind,
    MechanismRecord,
    MechanismRepository,
    MechanismVersionRef,
    TransferStatus,
)


@dataclass(frozen=True)
class RuntimeMechanismEntry:
    mechanism: MechanismVersionRef
    representation_artifact_id: str
    transfer_status: TransferStatus


@dataclass(frozen=True)
class MechanismArtifactProjection:
    kind: MechanismKind
    entries: tuple[RuntimeMechanismEntry, ...]


class MechanismRuntimeProjector:
    """Typed runtime views over mechanisms applicable to one context.

    This projector intentionally does not deserialize backend-specific
    artifacts. It only exposes typed references for runtimes whose concrete
    artifact loaders are owned elsewhere.
    """

    def __init__(self, repository: MechanismRepository) -> None:
        self.repository = repository

    def causal_models(
        self,
        context: ApplicabilityContext,
    ) -> MechanismArtifactProjection:
        return self._project_kind(context, MechanismKind.CAUSAL_MODEL)

    def state_features(
        self,
        context: ApplicabilityContext,
    ) -> MechanismArtifactProjection:
        return self._project_kind(context, MechanismKind.STATE_FEATURE)

    def symbolic_rules(
        self,
        context: ApplicabilityContext,
    ) -> MechanismArtifactProjection:
        return self._project_kind(context, MechanismKind.SYMBOLIC_RULE)

    def _project_kind(
        self,
        context: ApplicabilityContext,
        kind: MechanismKind,
    ) -> MechanismArtifactProjection:
        records = self.repository.find_applicable(context)
        return MechanismArtifactProjection(
            kind=kind,
            entries=tuple(
                RuntimeMechanismEntry(
                    mechanism=record.ref,
                    representation_artifact_id=(
                        record.representation_artifact_id
                    ),
                    transfer_status=record.transfer_status,
                )
                for record in records
                if record.kind is kind
            ),
        )


@dataclass(frozen=True)
class DreamCoderGrammarEntry:
    mechanism: MechanismVersionRef
    representation_artifact_id: str
    source_program_artifact_id: str
    program: str


@dataclass(frozen=True)
class DreamCoderGrammarProjection:
    entries: tuple[DreamCoderGrammarEntry, ...]

    @property
    def programs(self) -> tuple[str, ...]:
        return tuple(entry.program for entry in self.entries)


class DreamCoderGrammarProjector:
    """Resolve applicable PROGRAM mechanisms into DreamCoder inventions."""

    def __init__(
        self,
        repository: MechanismRepository,
        engine: DreamCoderRepairEngine,
    ) -> None:
        self.repository = repository
        self.engine = engine
        self.program_projector = ProgramMechanismProjector(engine)

    def project(
        self,
        context: ApplicabilityContext,
    ) -> DreamCoderGrammarProjection:
        records = tuple(
            record
            for record in self.repository.find_applicable(context)
            if record.kind is MechanismKind.PROGRAM
        )
        return DreamCoderGrammarProjection(
            tuple(self._entry(record) for record in records)
        )

    def _entry(
        self,
        record: MechanismRecord,
    ) -> DreamCoderGrammarEntry:
        representation = record.representation_artifact_id

        if representation.startswith("dreamcoder-program:sha256:"):
            source_program_artifact_id = representation
        elif representation.startswith("program-mechanism:sha256:"):
            projection = self.program_projector.load_projection_artifact(
                representation
            )
            source_program_artifact_id = projection.get(
                "source_program_artifact_id"
            )
            if not isinstance(source_program_artifact_id, str):
                raise ValueError(
                    "program-mechanism artifact has no source DreamCoder program"
                )
        else:
            raise ValueError(
                "unsupported PROGRAM representation for DreamCoder runtime: "
                f"{representation}"
            )

        artifact = self.engine.load_artifact(source_program_artifact_id)
        if artifact.get("schema") != "ecsa.dreamcoder-program.v1":
            raise ValueError("unsupported DreamCoder program artifact schema")
        program = artifact.get("program")
        if not isinstance(program, str) or not program:
            raise ValueError("DreamCoder program artifact has no executable program")

        return DreamCoderGrammarEntry(
            mechanism=record.ref,
            representation_artifact_id=representation,
            source_program_artifact_id=source_program_artifact_id,
            program=program,
        )
