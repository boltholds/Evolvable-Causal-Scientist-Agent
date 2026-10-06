from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ml_metadata import errors as mlmd_errors
from ml_metadata import errors as mlmd_errors
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2

from ecsa.mechanisms import (
    ApplicabilityContext,
    MechanismLineage,
    MechanismRecord,
    MechanismRelationKind,
    MechanismRepository,
    MechanismStatus,
    MechanismTransferStatus,
    MechanismVersionRef,
)


_MECHANISM_TYPE = "ECSA.Mechanism"
_EXTERNAL_ARTIFACT_TYPE = "ECSA.ExternalArtifact"
_ADMISSION_EXECUTION = "MechanismAdmission"
_DEPRECATION_EXECUTION = "MechanismDeprecation"


class MLMDMechanismRepository(MechanismRepository):
    """MLMD-backed persistence and lineage for ECSA mechanism records."""

    def __init__(
        self,
        store_or_path: metadata_store.MetadataStore | Path | str,
    ) -> None:
        if isinstance(store_or_path, (str, Path)):
            path = Path(store_or_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            config = metadata_store_pb2.ConnectionConfig()
            config.sqlite.filename_uri = str(path)
            config.sqlite.connection_mode = 3  # READWRITE_OPENCREATE
            self._store = metadata_store.MetadataStore(config)
        else:
            self._store = store_or_path
        self._mechanism_type_id = self._ensure_artifact_type(
            _MECHANISM_TYPE
        )
        self._external_type_id = self._ensure_artifact_type(
            _EXTERNAL_ARTIFACT_TYPE
        )
        self._admission_type_id = self._ensure_execution_type(
            _ADMISSION_EXECUTION
        )
        self._deprecation_type_id = self._ensure_execution_type(
            _DEPRECATION_EXECUTION
        )

    @classmethod
    def sqlite(cls, path: Path) -> "MLMDMechanismRepository":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = metadata_store_pb2.ConnectionConfig()
        config.sqlite.filename_uri = str(path)
        config.sqlite.connection_mode = 3  # READWRITE_OPENCREATE
        return cls(metadata_store.MetadataStore(config))

    def admit(self, record: MechanismRecord) -> MechanismRecord:
        if record.status is not MechanismStatus.ADMITTED:
            raise ValueError(
                "MechanismRepository.admit requires status=ADMITTED"
            )
        return self._persist(
            record,
            execution_kind=_ADMISSION_EXECUTION,
            execution_type_id=self._admission_type_id,
        )

    def get(
        self,
        mechanism: MechanismVersionRef | str,
        version: int | None = None,
    ) -> MechanismRecord | None:
        if isinstance(mechanism, MechanismVersionRef):
            if version is not None:
                raise ValueError(
                    "version is redundant with MechanismVersionRef"
                )
            mechanism_id = mechanism.mechanism_id
            version = mechanism.version
        else:
            mechanism_id = mechanism
        if not mechanism_id:
            raise ValueError("mechanism_id is required")
        if version is not None:
            if type(version) is not int or version < 1:
                raise ValueError("version must be a positive integer")
            artifact = self._get_mechanism_artifact(
                mechanism_id,
                version,
            )
            return None if artifact is None else self._decode_record(artifact)

        matches = [
            self._decode_record(artifact)
            for artifact in self._store.get_artifacts_by_type(
                _MECHANISM_TYPE
            )
            if self._decode_record(artifact).mechanism_id == mechanism_id
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: item.version)

    def find_applicable(
        self,
        context: ApplicabilityContext,
    ) -> tuple[MechanismRecord, ...]:
        if not isinstance(context, ApplicabilityContext):
            raise ValueError("typed ApplicabilityContext required")

        latest: dict[str, MechanismRecord] = {}
        for artifact in self._store.get_artifacts_by_type(
            _MECHANISM_TYPE
        ):
            record = self._decode_record(artifact)
            previous = latest.get(record.mechanism_id)
            if previous is None or record.version > previous.version:
                latest[record.mechanism_id] = record

        candidates = [
            record
            for record in latest.values()
            if (
                record.status is MechanismStatus.ADMITTED
                and record.scope.matches(context)
                and self._transfer_allows_reuse(record)
            )
        ]

        superseded = {
            (
                relation.target_mechanism_id,
                relation.target_version,
            )
            for record in candidates
            for relation in record.relations
            if relation.kind is MechanismRelationKind.SUPERSEDES
        }
        candidates = [
            record
            for record in candidates
            if record.identity not in superseded
        ]
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    item.mechanism_id,
                    item.version,
                ),
            )
        )

    def lineage(
        self,
        mechanism: MechanismVersionRef | str,
        version: int | None = None,
    ) -> MechanismLineage | None:
        record = self.get(mechanism, version)
        if record is None:
            return None
        artifact = self._get_mechanism_artifact(
            record.mechanism_id,
            record.version,
        )
        if artifact is None:
            raise RuntimeError("mechanism record exists without MLMD artifact")

        output_events = [
            event
            for event in self._store.get_events_by_artifact_ids(
                [artifact.id]
            )
            if event.type == metadata_store_pb2.Event.DECLARED_OUTPUT
        ]
        if len(output_events) != 1:
            raise RuntimeError(
                "mechanism artifact must have exactly one producing execution"
            )
        execution_id = output_events[0].execution_id
        executions = self._store.get_executions_by_id([execution_id])
        if len(executions) != 1:
            raise RuntimeError("mechanism producing execution is missing")
        execution = executions[0]
        types = self._store.get_execution_types_by_id(
            [execution.type_id]
        )
        if len(types) != 1:
            raise RuntimeError("mechanism execution type is missing")

        input_events = [
            event
            for event in self._store.get_events_by_execution_ids(
                [execution_id]
            )
            if event.type == metadata_store_pb2.Event.DECLARED_INPUT
        ]
        input_artifacts = self._store.get_artifacts_by_id(
            [event.artifact_id for event in input_events]
        )
        external_ids = {
            item.external_id
            for item in input_artifacts
            if item.external_id
        }
        expected_sources = {
            record.representation_artifact_id,
            *record.supporting_evidence_ids,
            *record.contradicting_evidence_ids,
            *record.provenance_artifact_ids,
        }
        missing = expected_sources - external_ids
        if missing:
            raise RuntimeError(
                f"MLMD lineage is missing mechanism inputs: {sorted(missing)}"
            )

        prior_version = None
        prior_id = self._custom_string(
            execution,
            "prior_mechanism_id",
        )
        prior_version_text = self._custom_string(
            execution,
            "prior_version",
        )
        if prior_id and prior_version_text:
            prior_version = (prior_id, int(prior_version_text))

        return MechanismLineage(
            mechanism=record.ref,
            record=record,
            representation_artifact_id=record.representation_artifact_id,
            supporting_evidence_ids=record.supporting_evidence_ids,
            contradicting_evidence_ids=record.contradicting_evidence_ids,
            provenance_artifact_ids=record.provenance_artifact_ids,
            relations=record.relations,
            related_mechanisms=tuple(
                relation.target
                for relation in record.relations
            ),
            execution_kind=types[0].name,
            prior_version=prior_version,
        )

    def deprecate(
        self,
        mechanism_id: str,
        *,
        reason: str,
    ) -> MechanismRecord:
        if not reason:
            raise ValueError("deprecation reason is required")
        previous = self.get(mechanism_id)
        if previous is None:
            raise ValueError(f"unknown mechanism: {mechanism_id}")
        if previous.status is MechanismStatus.DEPRECATED:
            raise ValueError("mechanism is already deprecated")

        deprecated = replace(
            previous,
            version=previous.version + 1,
            status=MechanismStatus.DEPRECATED,
        )
        return self._persist(
            deprecated,
            execution_kind=_DEPRECATION_EXECUTION,
            execution_type_id=self._deprecation_type_id,
            prior=previous,
            reason=reason,
        )

    def _persist(
        self,
        record: MechanismRecord,
        *,
        execution_kind: str,
        execution_type_id: int,
        prior: MechanismRecord | None = None,
        reason: str | None = None,
    ) -> MechanismRecord:
        if self._get_mechanism_artifact(
            record.mechanism_id,
            record.version,
        ) is not None:
            raise ValueError(
                "mechanism identity/version already exists"
            )

        relation_artifacts = []
        for relation in record.relations:
            target = self._get_mechanism_artifact(
                relation.target_mechanism_id,
                relation.target_version,
            )
            if target is None:
                raise ValueError(
                    "mechanism relation target is not persisted: "
                    f"{relation.target_mechanism_id}@{relation.target_version}"
                )
            relation_artifacts.append(target)

        source_ids = tuple(
            dict.fromkeys(
                (
                    record.representation_artifact_id,
                    *record.supporting_evidence_ids,
                    *record.contradicting_evidence_ids,
                    *record.provenance_artifact_ids,
                )
            )
        )
        source_artifacts = [
            self._get_or_create_external_artifact(source_id)
            for source_id in source_ids
        ]

        prior_artifact = None
        if prior is not None:
            prior_artifact = self._get_mechanism_artifact(
                prior.mechanism_id,
                prior.version,
            )
            if prior_artifact is None:
                raise RuntimeError("prior mechanism version is missing")

        mechanism_artifact = metadata_store_pb2.Artifact()
        mechanism_artifact.type_id = self._mechanism_type_id
        mechanism_artifact.external_id = self._mechanism_external_id(
            record.mechanism_id,
            record.version,
        )
        mechanism_artifact.custom_properties[
            "record_json"
        ].string_value = json.dumps(
            record.to_wire(),
            sort_keys=True,
            separators=(",", ":"),
        )
        mechanism_artifact.custom_properties[
            "mechanism_id"
        ].string_value = record.mechanism_id
        mechanism_artifact.custom_properties[
            "version"
        ].int_value = record.version
        mechanism_artifact.custom_properties[
            "status"
        ].string_value = record.status.value
        [mechanism_artifact_id] = self._store.put_artifacts(
            [mechanism_artifact]
        )

        execution = metadata_store_pb2.Execution()
        execution.type_id = execution_type_id
        execution.external_id = (
            f"ecsa:{execution_kind}:"
            f"{record.mechanism_id}:v{record.version}"
        )
        execution.custom_properties[
            "mechanism_id"
        ].string_value = record.mechanism_id
        execution.custom_properties[
            "version"
        ].int_value = record.version
        if reason is not None:
            execution.custom_properties[
                "reason"
            ].string_value = reason
        if prior is not None:
            execution.custom_properties[
                "prior_mechanism_id"
            ].string_value = prior.mechanism_id
            execution.custom_properties[
                "prior_version"
            ].string_value = str(prior.version)
        [execution_id] = self._store.put_executions([execution])

        input_artifact_ids = [
            artifact.id
            for artifact in source_artifacts
        ]
        input_artifact_ids.extend(
            artifact.id
            for artifact in relation_artifacts
        )
        if prior_artifact is not None:
            input_artifact_ids.append(prior_artifact.id)
        input_artifact_ids = list(
            dict.fromkeys(input_artifact_ids)
        )

        events = []
        for artifact_id in input_artifact_ids:
            event = metadata_store_pb2.Event()
            event.artifact_id = artifact_id
            event.execution_id = execution_id
            event.type = metadata_store_pb2.Event.DECLARED_INPUT
            events.append(event)

        output = metadata_store_pb2.Event()
        output.artifact_id = mechanism_artifact_id
        output.execution_id = execution_id
        output.type = metadata_store_pb2.Event.DECLARED_OUTPUT
        events.append(output)
        self._store.put_events(events)

        self._attach_scope_contexts(
            record,
            mechanism_artifact_id,
            execution_id,
        )
        return record

    def _attach_scope_contexts(
        self,
        record: MechanismRecord,
        artifact_id: int,
        execution_id: int,
    ) -> None:
        dimensions = (
            ("Environment", record.scope.context_ids),
            ("Regime", record.scope.regime_ids),
            ("Domain", record.scope.domain_ids),
            ("Task", record.scope.task_ids),
        )
        attributions = []
        associations = []
        for type_name, names in dimensions:
            for name in names:
                context_id = self._get_or_create_context(
                    f"ECSA.{type_name}",
                    name,
                )
                attribution = metadata_store_pb2.Attribution()
                attribution.artifact_id = artifact_id
                attribution.context_id = context_id
                attributions.append(attribution)

                association = metadata_store_pb2.Association()
                association.execution_id = execution_id
                association.context_id = context_id
                associations.append(association)

        if attributions or associations:
            self._store.put_attributions_and_associations(
                attributions,
                associations,
            )

    def _get_or_create_external_artifact(
        self,
        external_id: str,
    ):
        try:
            existing = self._store.get_artifacts_by_external_ids(
                [external_id]
            )
        except mlmd_errors.NotFoundError:
            existing = []
        external = [
            artifact
            for artifact in existing
            if artifact.type_id == self._external_type_id
        ]
        if external:
            if len(external) != 1:
                raise RuntimeError(
                    f"duplicate external artifact identity: {external_id}"
                )
            return external[0]

        artifact = metadata_store_pb2.Artifact()
        artifact.type_id = self._external_type_id
        artifact.external_id = external_id
        artifact.custom_properties[
            "source_ref"
        ].string_value = external_id
        [artifact_id] = self._store.put_artifacts([artifact])
        [stored] = self._store.get_artifacts_by_id([artifact_id])
        return stored

    def _get_mechanism_artifact(
        self,
        mechanism_id: str,
        version: int,
    ):
        external_id = self._mechanism_external_id(
            mechanism_id,
            version,
        )
        try:
            by_external_id = self._store.get_artifacts_by_external_ids(
                [external_id]
            )
        except mlmd_errors.NotFoundError:
            by_external_id = []
        matches = [
            artifact
            for artifact in by_external_id
            if artifact.type_id == self._mechanism_type_id
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError(
                "duplicate MLMD mechanism artifact identity"
            )
        return matches[0]

    def _get_artifacts_by_external_ids_or_empty(
        self,
        external_id: str,
    ):
        try:
            return self._store.get_artifacts_by_external_ids(
                [external_id]
            )
        except mlmd_errors.NotFoundError:
            return []

    @staticmethod
    def _mechanism_external_id(
        mechanism_id: str,
        version: int,
    ) -> str:
        return f"ecsa:mechanism:{mechanism_id}:v{version}"

    @staticmethod
    def _decode_record(artifact) -> MechanismRecord:
        value = artifact.custom_properties.get("record_json")
        if value is None or not value.string_value:
            raise RuntimeError(
                "MLMD mechanism artifact has no record_json"
            )
        return MechanismRecord.from_wire(
            json.loads(value.string_value)
        )

    @staticmethod
    def _transfer_allows_reuse(
        record: MechanismRecord,
    ) -> bool:
        if record.transfer is not MechanismTransferStatus.SPLIT:
            return True
        return bool(
            record.scope.context_ids
            or record.scope.regime_ids
            or record.scope.domain_ids
            or record.scope.task_ids
        )

    def _ensure_artifact_type(self, name: str) -> int:
        existing = {
            item.name: item.id
            for item in self._store.get_artifact_types()
        }
        if name in existing:
            return existing[name]
        value = metadata_store_pb2.ArtifactType()
        value.name = name
        return self._store.put_artifact_type(value)

    def _ensure_execution_type(self, name: str) -> int:
        existing = {
            item.name: item.id
            for item in self._store.get_execution_types()
        }
        if name in existing:
            return existing[name]
        value = metadata_store_pb2.ExecutionType()
        value.name = name
        return self._store.put_execution_type(value)

    def _ensure_context_type(self, name: str) -> int:
        existing = {
            item.name: item.id
            for item in self._store.get_context_types()
        }
        if name in existing:
            return existing[name]
        value = metadata_store_pb2.ContextType()
        value.name = name
        return self._store.put_context_type(value)

    def _get_or_create_context(
        self,
        type_name: str,
        name: str,
    ) -> int:
        type_id = self._ensure_context_type(type_name)
        matches = [
            context
            for context in self._store.get_contexts_by_type(type_name)
            if context.name == name
        ]
        if matches:
            if len(matches) != 1:
                raise RuntimeError(
                    f"duplicate MLMD context: {type_name}/{name}"
                )
            return matches[0].id

        context = metadata_store_pb2.Context()
        context.type_id = type_id
        context.name = name
        [context_id] = self._store.put_contexts([context])
        return context_id

    @staticmethod
    def _custom_string(node, key: str) -> str | None:
        value = node.custom_properties.get(key)
        if value is None or not value.string_value:
            return None
        return value.string_value
