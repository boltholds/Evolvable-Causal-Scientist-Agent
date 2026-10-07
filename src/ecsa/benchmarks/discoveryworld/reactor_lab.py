from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path

from ecsa.mechanisms import (
    ApplicabilityContext,
    EpistemicStatus,
    MechanismKind,
    MechanismRecord,
    MechanismRelation,
    MechanismRelationKind,
    MechanismScope,
    TransferStatus,
)

from .contracts import (
    ActionPacket,
    DiscoveryWorldActionResult,
    GenericNumericHypothesis,
    GenericScalarEvidence,
    JSONValue,
    MeasurementKind,
    ParsedMeasurement,
    ReactorFrequencyPrediction,
    ReactorMeasurement,
    ReactorMechanismHypothesis,
    ReactorValidationEvent,
)


_NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"


def reactor_context(seed: int) -> ApplicabilityContext:
    if type(seed) is not int or seed not in range(5):
        raise ValueError("Reactor Lab benchmark seed must be 0..4")
    return ApplicabilityContext(
        context_id=f"discoveryworld:reactor-lab:normal:seed-{seed}",
        regime_id="normal",
        domain_id="discoveryworld",
        task_id="reactor-lab",
        assumptions=("public-observation-only", "linear-family"),
    )


def parse_public_measurement(message: str) -> ParsedMeasurement | None:
    if not isinstance(message, str):
        raise ValueError("measurement message must be a string")
    lowered = message.lower()
    if "inconclusive" in lowered:
        return None

    channels = re.findall(
        rf"channel\s+(\d+)\s*:\s*{_NUMBER}",
        message,
        flags=re.IGNORECASE,
    )
    if channels:
        indexed = sorted(
            ((int(index), float(value)) for index, value in channels),
            key=lambda pair: pair[0],
        )
        expected = list(range(1, len(indexed) + 1))
        if [index for index, _ in indexed] != expected:
            return None
        values = tuple(value for _, value in indexed)
        return ParsedMeasurement(
            MeasurementKind.SPECTRUM,
            values,
            channel_values=values,
        )

    patterns = (
        (
            MeasurementKind.DENSITY,
            rf"density\b.*?approximately\s+{_NUMBER}\s+grams per cubic centimeter",
        ),
        (
            MeasurementKind.TEMPERATURE,
            rf"temperature\s+of\s+{_NUMBER}\s+degrees Celsius",
        ),
        (
            MeasurementKind.QUANTUM_SIZE,
            rf"quantum gap\b.*?{_NUMBER}\s*nm",
        ),
        (
            MeasurementKind.RADIATION,
            rf"radiation meter reports a level of\s+{_NUMBER}\s+micro Seiverts per hour",
        ),
    )
    for kind, pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE | re.DOTALL)
        if match is not None:
            return ParsedMeasurement(kind, (float(match.group(1)),))
    return None


def _public_objects(
    observation: dict[str, JSONValue],
) -> dict[int, dict[str, JSONValue]]:
    ui = observation.get("ui")
    if not isinstance(ui, dict):
        return {}

    found: dict[int, dict[str, JSONValue]] = {}

    def add_many(value: object) -> None:
        if not isinstance(value, list):
            return
        for item in value:
            if not isinstance(item, dict):
                continue
            uuid = item.get("uuid")
            if type(uuid) is int:
                found[uuid] = item

    add_many(ui.get("inventoryObjects"))
    add_many(ui.get("accessibleEnvironmentObjects"))

    nearby = ui.get("nearbyObjects")
    if isinstance(nearby, dict):
        groups = nearby.get("objects")
        if isinstance(groups, dict):
            for values in groups.values():
                add_many(values)
    return found


def _public_name(
    observation: dict[str, JSONValue],
    uuid: int,
) -> str | None:
    item = _public_objects(observation).get(uuid)
    if item is None:
        return None
    name = item.get("name")
    return name if isinstance(name, str) else None


def _public_reactor_frequency(
    observation: dict[str, JSONValue],
) -> float | None:
    ui = observation.get("ui")
    if not isinstance(ui, dict):
        return None
    dialog = ui.get("dialog_box")
    if not isinstance(dialog, dict):
        return None
    dialog_in = dialog.get("dialogIn")
    if not isinstance(dialog_in, str):
        return None
    match = re.search(
        rf"current resonance frequenc(?:e|y) is:\s*{_NUMBER}\s*Hertz",
        dialog_in,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return float(match.group(1))


def _instrument_measurement_kind(name: str) -> MeasurementKind | None:
    lowered = name.lower()
    if "densitometer" in lowered:
        return MeasurementKind.DENSITY
    if "thermometer" in lowered:
        return MeasurementKind.TEMPERATURE
    if "spectrometer" in lowered:
        return MeasurementKind.SPECTRUM
    if "microscope" in lowered:
        return MeasurementKind.QUANTUM_SIZE
    if "radiation meter" in lowered:
        return MeasurementKind.RADIATION
    return None


def measurement_action_key(
    observation: dict[str, JSONValue],
    action: ActionPacket,
) -> tuple[int, int] | None:
    if action.get("action") != "USE":
        return None
    instrument_uuid = action.get("arg1")
    crystal_uuid = action.get("arg2")
    if type(instrument_uuid) is not int or type(crystal_uuid) is not int:
        return None
    public = _public_objects(observation)
    instrument = public.get(instrument_uuid)
    crystal = public.get(crystal_uuid)
    if instrument is None or crystal is None:
        return None
    instrument_name = instrument.get("name")
    crystal_name = crystal.get("name")
    if not isinstance(instrument_name, str) or not isinstance(crystal_name, str):
        return None
    if _instrument_measurement_kind(instrument_name) is None:
        return None
    if "quantum crystal" not in crystal_name.lower():
        return None
    return instrument_uuid, crystal_uuid


class ReactorLabScientificSidecar:
    def __init__(self) -> None:
        self._measurements: list[ReactorMeasurement] = []
        self._hypotheses: dict[str, ReactorMechanismHypothesis] = {}
        self._validated: set[tuple[str, int]] = set()

    @property
    def measurements(self) -> tuple[ReactorMeasurement, ...]:
        return tuple(self._measurements)

    def record_transition(
        self,
        *,
        step: int,
        context_id: str,
        pre_observation: dict[str, JSONValue],
        action: ActionPacket,
        action_result: DiscoveryWorldActionResult,
        post_observation: dict[str, JSONValue],
    ) -> tuple[ReactorMeasurement, ...]:
        if not action_result.success or action.get("action") != "USE":
            return ()
        instrument_uuid = action.get("arg1")
        crystal_uuid = action.get("arg2")
        if type(instrument_uuid) is not int or type(crystal_uuid) is not int:
            return ()

        public = _public_objects(pre_observation)
        instrument = public.get(instrument_uuid)
        crystal = public.get(crystal_uuid)
        if instrument is None or crystal is None:
            return ()
        instrument_name = instrument.get("name")
        crystal_name = crystal.get("name")
        if not isinstance(instrument_name, str) or not isinstance(
            crystal_name, str
        ):
            return ()
        if "quantum crystal" not in crystal_name.lower():
            return ()
        expected_kind = _instrument_measurement_kind(instrument_name)
        if expected_kind is None:
            return ()

        ui = post_observation.get("ui")
        if not isinstance(ui, dict):
            return ()
        raw = ui.get("lastActionMessage")
        if not isinstance(raw, str) or not raw:
            return ()
        parsed = parse_public_measurement(raw)
        if parsed is None or parsed.kind is not expected_kind:
            return ()

        evidence_id = (
            f"dw-reactor-measurement:{context_id}:{step}:"
            f"{instrument_uuid}:{crystal_uuid}"
        )
        measurement = ReactorMeasurement(
            evidence_id=evidence_id,
            step=step,
            context_id=context_id,
            crystal_uuid=crystal_uuid,
            crystal_name=crystal_name,
            instrument_uuid=instrument_uuid,
            instrument_name=instrument_name,
            kind=parsed.kind,
            values=parsed.values,
            raw_message=raw,
        )
        self._measurements.append(measurement)
        return (measurement,)

    def freeze_hypothesis(
        self,
        hypothesis: ReactorMechanismHypothesis,
    ) -> None:
        if not isinstance(hypothesis, ReactorMechanismHypothesis):
            raise ValueError("typed ReactorMechanismHypothesis required")
        if hypothesis.hypothesis_id in self._hypotheses:
            raise ValueError("hypothesis_id is already frozen")
        self._hypotheses[hypothesis.hypothesis_id] = hypothesis

    def observe_validation(
        self,
        *,
        step: int,
        pre_observation: dict[str, JSONValue],
        post_observation: dict[str, JSONValue],
        hypothesis_ids: tuple[str, ...] | None = None,
    ) -> tuple[ReactorValidationEvent, ...]:
        events: list[ReactorValidationEvent] = []
        if hypothesis_ids is None:
            hypotheses = tuple(self._hypotheses.values())
        else:
            missing = [
                hypothesis_id
                for hypothesis_id in hypothesis_ids
                if hypothesis_id not in self._hypotheses
            ]
            if missing:
                raise ValueError(
                    f"validation references unfrozen hypotheses: {missing}"
                )
            hypotheses = tuple(
                self._hypotheses[hypothesis_id]
                for hypothesis_id in hypothesis_ids
            )

        for hypothesis in hypotheses:
            for prediction in hypothesis.predictions:
                key = (hypothesis.hypothesis_id, prediction.target_reactor_uuid)
                if key in self._validated or prediction.frozen_step >= step:
                    continue
                before = _public_name(
                    pre_observation, prediction.target_reactor_uuid
                )
                after = _public_name(
                    post_observation, prediction.target_reactor_uuid
                )
                if before is None or after is None:
                    continue
                if "(activated)" in before.lower():
                    continue
                observed_frequency = _public_reactor_frequency(
                    post_observation
                )
                if "(activated)" in after.lower():
                    if observed_frequency is None:
                        continue
                    success = (
                        abs(
                            observed_frequency
                            - prediction.predicted_frequency
                        )
                        < 2.0
                    )
                elif "(uncalibrated)" in after.lower():
                    success = False
                else:
                    continue
                event = ReactorValidationEvent(
                    event_id=(
                        f"dw-reactor-validation:{hypothesis.hypothesis_id}:"
                        f"{prediction.target_reactor_uuid}:{step}"
                    ),
                    hypothesis_id=hypothesis.hypothesis_id,
                    target_crystal_uuid=prediction.target_crystal_uuid,
                    target_reactor_uuid=prediction.target_reactor_uuid,
                    predicted_frequency=prediction.predicted_frequency,
                    validation_step=step,
                    success=success,
                    observed_frequency=observed_frequency,
                )
                self._validated.add(key)
                events.append(event)
        return tuple(events)


def _prediction_to_wire(prediction: ReactorFrequencyPrediction) -> dict:
    return {
        "target_crystal_uuid": prediction.target_crystal_uuid,
        "target_reactor_uuid": prediction.target_reactor_uuid,
        "predicted_frequency": prediction.predicted_frequency,
        "frozen_step": prediction.frozen_step,
    }


def _validation_to_wire(validation: ReactorValidationEvent) -> dict:
    return {
        "event_id": validation.event_id,
        "hypothesis_id": validation.hypothesis_id,
        "target_crystal_uuid": validation.target_crystal_uuid,
        "target_reactor_uuid": validation.target_reactor_uuid,
        "predicted_frequency": validation.predicted_frequency,
        "validation_step": validation.validation_step,
        "success": validation.success,
        "observed_frequency": validation.observed_frequency,
    }


def bridge_generic_numeric_hypothesis(
    *,
    hypothesis: GenericNumericHypothesis,
    generic_evidence: tuple[GenericScalarEvidence, ...],
    measurements: tuple[ReactorMeasurement, ...],
) -> ReactorMechanismHypothesis:
    if hypothesis.polynomial_degree != 1:
        raise ValueError(
            "Reactor Lab qualification currently supports degree-1 generic programs"
        )
    evidence_by_id = {
        item.evidence_id: item
        for item in generic_evidence
    }
    mapped: list[ReactorMeasurement] = []
    for evidence_id in hypothesis.source_evidence_ids:
        generic = evidence_by_id.get(evidence_id)
        if generic is None or len(generic.entity_ids) < 2:
            continue
        for measurement in measurements:
            if measurement.step != generic.step:
                continue
            if (
                measurement.instrument_uuid not in generic.entity_ids
                or measurement.crystal_uuid not in generic.entity_ids
            ):
                continue
            if any(
                abs(value - generic.value) < 1e-9
                for value in measurement.values
            ):
                mapped.append(measurement)
                break

    mapped = list(
        {
            item.evidence_id: item
            for item in mapped
        }.values()
    )
    if len(mapped) < 2:
        raise ValueError(
            "generic hypothesis lacks two public typed measurements for qualification"
        )
    kinds = {item.kind for item in mapped}
    if len(kinds) != 1:
        raise ValueError(
            "generic hypothesis source evidence spans incompatible measurement kinds"
        )
    [kind] = kinds

    offset, slope = hypothesis.coefficients
    return ReactorMechanismHypothesis(
        hypothesis_id=hypothesis.hypothesis_id,
        measurement_kind=kind,
        slope=slope,
        offset=offset,
        source_evidence_ids=tuple(
            item.evidence_id
            for item in mapped
        ),
        predictions=tuple(
            ReactorFrequencyPrediction(
                target_crystal_uuid=item.subject_entity_id,
                target_reactor_uuid=item.control_entity_id,
                predicted_frequency=item.predicted_value,
                frozen_step=item.frozen_step,
            )
            for item in hypothesis.predictions
        ),
        source_mechanism=hypothesis.source_mechanism,
    )


def add_generic_transfer_annotations(
    mechanism: MechanismRecord,
    hypothesis: GenericNumericHypothesis,
) -> MechanismRecord:
    annotations = (
        ("transfer_key", hypothesis.transfer_key),
        ("generic_feature_key", hypothesis.feature_key),
        ("generic_control_feature_key", hypothesis.control_feature_key),
        ("generic_degree", hypothesis.polynomial_degree),
    )
    existing_names = {name for name, _ in mechanism.parameters}
    parameters = mechanism.parameters + tuple(
        item
        for item in annotations
        if item[0] not in existing_names
    )
    return MechanismRecord(
        mechanism_id=mechanism.mechanism_id,
        version=mechanism.version,
        kind=mechanism.kind,
        epistemic_status=mechanism.epistemic_status,
        representation_artifact=mechanism.representation_artifact,
        scope=mechanism.scope,
        transfer_status=mechanism.transfer_status,
        parameters=parameters,
        supporting_evidence=mechanism.supporting_evidence,
        contradicting_evidence=mechanism.contradicting_evidence,
        provenance=mechanism.provenance,
        relations=mechanism.relations,
    )


class ReactorRuleArtifactStore:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = Path(cache_root)

    def write(
        self,
        hypothesis: ReactorMechanismHypothesis,
        validation: ReactorValidationEvent,
    ) -> str:
        if validation.hypothesis_id != hypothesis.hypothesis_id:
            raise ValueError("validation does not match hypothesis")
        wire = {
            "schema": "ecsa.discoveryworld-reactor-rule.v1",
            "hypothesis_id": hypothesis.hypothesis_id,
            "functional_family": "linear",
            "measurement_kind": hypothesis.measurement_kind.value,
            "slope": hypothesis.slope,
            "offset": hypothesis.offset,
            "source_evidence_ids": list(hypothesis.source_evidence_ids),
            "predictions": [
                _prediction_to_wire(value)
                for value in hypothesis.predictions
            ],
            "source_mechanism": (
                {
                    "mechanism_id": hypothesis.source_mechanism.mechanism_id,
                    "version": hypothesis.source_mechanism.version,
                }
                if hypothesis.source_mechanism is not None
                else None
            ),
            "validation": _validation_to_wire(validation),
        }
        encoded = json.dumps(
            wire, sort_keys=True, separators=(",", ":")
        ).encode()
        digest = sha256(encoded).hexdigest()
        path = self._path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = encoded.decode() + "\n"
        if path.exists() and path.read_text() != serialized:
            raise ValueError("conflicting Reactor rule artifact")
        if not path.exists():
            path.write_text(serialized)
        return f"reactor-rule:sha256:{digest}"

    def load(self, artifact_id: str) -> dict[str, JSONValue]:
        prefix = "reactor-rule:sha256:"
        if not artifact_id.startswith(prefix):
            raise ValueError("invalid Reactor rule artifact id")
        digest = artifact_id[len(prefix):]
        if len(digest) != 64:
            raise ValueError("invalid Reactor rule digest")
        path = self._path(digest)
        if not path.exists():
            raise FileNotFoundError(artifact_id)
        wire = json.loads(path.read_text())
        actual = sha256(
            json.dumps(
                wire, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if actual != digest:
            raise ValueError("Reactor rule artifact digest mismatch")
        return wire

    def _path(self, digest: str) -> Path:
        return self.cache_root / "reactor-rules" / f"{digest}.json"


def build_admitted_reactor_mechanism(
    *,
    hypothesis: ReactorMechanismHypothesis,
    validation: ReactorValidationEvent,
    context_id: str,
    artifact_store: ReactorRuleArtifactStore,
) -> MechanismRecord:
    if not validation.success:
        raise ValueError("only successful prospective validation is admissible")
    if validation.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError("validation does not match hypothesis")
    if not context_id:
        raise ValueError("context_id is required")

    artifact_id = artifact_store.write(hypothesis, validation)
    digest = artifact_id.rsplit(":", 1)[-1]
    relations = (
        (
            MechanismRelation(
                MechanismRelationKind.SPECIALIZES,
                hypothesis.source_mechanism,
            ),
        )
        if hypothesis.source_mechanism is not None
        else ()
    )
    supporting = tuple(
        dict.fromkeys(
            (*hypothesis.source_evidence_ids, validation.event_id)
        )
    )
    return MechanismRecord(
        mechanism_id=f"dw-reactor-linear:{digest[:16]}",
        version=1,
        kind=MechanismKind.SYMBOLIC_RULE,
        epistemic_status=EpistemicStatus.ADMITTED,
        representation_artifact=artifact_id,
        scope=MechanismScope(
            context_ids=(context_id,),
            regime_ids=("normal",),
            domain_ids=("discoveryworld",),
            task_ids=("reactor-lab",),
            required_assumptions=(
                "public-observation-only",
                "linear-family",
            ),
        ),
        transfer_status=TransferStatus.CONTEXT_SPECIALIZED,
        parameters=(
            ("measurement_kind", hypothesis.measurement_kind.value),
            ("slope", hypothesis.slope),
            ("offset", hypothesis.offset),
        ),
        supporting_evidence=supporting,
        provenance=(),
        relations=relations,
    )
