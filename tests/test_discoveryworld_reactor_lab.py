from pathlib import Path

import pytest

from ecsa.benchmarks.discoveryworld.contracts import (
    DiscoveryWorldActionResult,
)
from ecsa.benchmarks.discoveryworld.reactor_lab import (
    MeasurementKind,
    ReactorFrequencyPrediction,
    ReactorLabScientificSidecar,
    ReactorMechanismHypothesis,
    ReactorRuleArtifactStore,
    build_admitted_reactor_mechanism,
    measurement_action_key,
    parse_public_measurement,
    reactor_context,
)
from ecsa.mechanisms import (
    MechanismKind,
    MechanismRelationKind,
    MechanismStatus,
    MechanismTransferStatus,
    MechanismVersionRef,
)


@pytest.mark.parametrize(
    ("message", "kind", "values"),
    (
        (
            "The density of the quantum crystal appears to be approximately "
            "12.340 grams per cubic centimeter.",
            MeasurementKind.DENSITY,
            (12.34,),
        ),
        (
            "The thermometer reports a temperature of 17.25 degrees Celsius.",
            MeasurementKind.TEMPERATURE,
            (17.25,),
        ),
        (
            "The quantum gap of this crystal appears to be 11.4 nm",
            MeasurementKind.QUANTUM_SIZE,
            (11.4,),
        ),
        (
            "The radiation meter reports a level of 8.22 micro Seiverts per hour.",
            MeasurementKind.RADIATION,
            (8.22,),
        ),
    ),
)
def test_parse_public_scalar_measurements(
    message: str,
    kind: MeasurementKind,
    values: tuple[float, ...],
) -> None:
    parsed = parse_public_measurement(message)
    assert parsed is not None
    assert parsed.kind is kind
    assert parsed.values == values


def test_parse_public_spectrum_measurement() -> None:
    parsed = parse_public_measurement(
        "The results are as follows:\n"
        "- Channel 1: 3.10\n"
        "- Channel 2: 4.20\n"
        "- Channel 3: 5.30\n"
        "- Channel 4: 6.40\n"
        "- Channel 5: 19.20\n"
    )
    assert parsed is not None
    assert parsed.kind is MeasurementKind.SPECTRUM
    assert parsed.channel_values[4] == 19.2
    assert parsed.values == (3.1, 4.2, 5.3, 6.4, 19.2)


def test_inconclusive_measurement_is_not_invented() -> None:
    assert parse_public_measurement("The results are inconclusive.") is None


def observation(
    *,
    message: str = "",
    reactor_name: str = "crystal reactor (uncalibrated)",
    reactor_frequency: float | None = None,
) -> dict:
    dialog_box = {}
    if reactor_frequency is not None:
        dialog_box = {
            "dialogIn": (
                "Hello, I am Crystal Reactor #3.\n"
                f"The current resonance frequence is: {reactor_frequency} Hertz.\n"
                "The allowable range is 0 to 10,000 Hz."
            ),
            "dialogOptions": {},
        }
    return {
        "ui": {
            "lastActionMessage": message,
            "dialog_box": dialog_box,
            "inventoryObjects": [
                {"uuid": 101, "name": "densitometer", "description": "densitometer"}
            ],
            "accessibleEnvironmentObjects": [
                {"uuid": 202, "name": "quantum crystal 3", "description": "quantum crystal 3"},
                {"uuid": 404, "name": reactor_name, "description": reactor_name},
            ],
            "nearbyObjects": {"objects": {}},
        }
    }




def test_measurement_action_key_counts_use_even_before_result_parsing() -> None:
    assert measurement_action_key(
        observation(),
        {"action": "USE", "arg1": 101, "arg2": 202},
    ) == (101, 202)
    assert measurement_action_key(
        observation(),
        {"action": "PICKUP", "arg1": 202},
    ) is None

def test_record_transition_uses_only_public_use_result() -> None:
    sidecar = ReactorLabScientificSidecar()
    measurements = sidecar.record_transition(
        step=3,
        context_id=reactor_context(0).context_id,
        pre_observation=observation(),
        action={"action": "USE", "arg1": 101, "arg2": 202},
        action_result=DiscoveryWorldActionResult(True, ()),
        post_observation=observation(
            message=(
                "You use the densitometer to examine the quantum crystal 3.\n"
                "The results are as follows:\n"
                "The density of the quantum crystal 3 appears to be approximately "
                "12.340 grams per cubic centimeter.\n"
            )
        ),
    )
    assert len(measurements) == 1
    measurement = measurements[0]
    assert measurement.kind is MeasurementKind.DENSITY
    assert measurement.crystal_uuid == 202
    assert measurement.instrument_uuid == 101
    assert measurement.values == (12.34,)
    assert "12.340" in measurement.raw_message


def hypothesis(
    *,
    source_mechanism: MechanismVersionRef | None = None,
) -> ReactorMechanismHypothesis:
    return ReactorMechanismHypothesis(
        hypothesis_id="linear-density",
        measurement_kind=MeasurementKind.DENSITY,
        slope=100.0,
        offset=90.0,
        source_evidence_ids=("measurement-e1",),
        predictions=(
            ReactorFrequencyPrediction(
                target_crystal_uuid=202,
                target_reactor_uuid=404,
                predicted_frequency=1324.0,
                frozen_step=7,
            ),
        ),
        source_mechanism=source_mechanism,
    )


def test_frozen_prediction_validates_only_on_new_public_activation() -> None:
    sidecar = ReactorLabScientificSidecar()
    sidecar.freeze_hypothesis(hypothesis())

    events = sidecar.observe_validation(
        step=8,
        pre_observation=observation(
            reactor_name="crystal reactor (uncalibrated)"
        ),
        post_observation=observation(
            reactor_name="crystal reactor (activated)"
        ),
    )

    assert len(events) == 1
    event = events[0]
    assert event.hypothesis_id == "linear-density"
    assert event.target_reactor_uuid == 404
    assert event.predicted_frequency == 1324.0
    assert event.success is True
    assert event.validation_step == 8






def test_activation_at_different_public_frequency_rejects_prediction() -> None:
    sidecar = ReactorLabScientificSidecar()
    sidecar.freeze_hypothesis(hypothesis())

    events = sidecar.observe_validation(
        step=8,
        pre_observation=observation(
            reactor_name="crystal reactor (uncalibrated)",
            reactor_frequency=1300.0,
        ),
        post_observation=observation(
            reactor_name="crystal reactor (activated)",
            reactor_frequency=1500.0,
        ),
    )

    assert len(events) == 1
    assert events[0].predicted_frequency == 1324.0
    assert events[0].success is False


def test_new_public_uncalibrated_state_is_failed_prospective_validation() -> None:
    source = MechanismVersionRef("source-reactor-law", 1)
    sidecar = ReactorLabScientificSidecar()
    sidecar.freeze_hypothesis(hypothesis(source_mechanism=source))

    events = sidecar.observe_validation(
        step=8,
        pre_observation=observation(
            reactor_name="crystal reactor (uncalibrated)"
        ),
        post_observation=observation(
            reactor_name="crystal reactor (uncalibrated)"
        ),
    )

    assert len(events) == 1
    assert events[0].success is False
    assert events[0].target_reactor_uuid == 404

def test_already_activated_reactor_is_not_retrospective_validation() -> None:
    sidecar = ReactorLabScientificSidecar()
    sidecar.freeze_hypothesis(hypothesis())

    events = sidecar.observe_validation(
        step=8,
        pre_observation=observation(
            reactor_name="crystal reactor (activated)"
        ),
        post_observation=observation(
            reactor_name="crystal reactor (activated)"
        ),
    )
    assert events == ()


def test_build_admitted_rule_preserves_scope_and_transfer_lineage(
    tmp_path: Path,
) -> None:
    parent = MechanismVersionRef("source-reactor-law", 1)
    h = hypothesis(source_mechanism=parent)
    sidecar = ReactorLabScientificSidecar()
    sidecar.freeze_hypothesis(h)
    [validation] = sidecar.observe_validation(
        step=8,
        pre_observation=observation(),
        post_observation=observation(
            reactor_name="crystal reactor (activated)"
        ),
    )
    store = ReactorRuleArtifactStore(tmp_path)

    record = build_admitted_reactor_mechanism(
        hypothesis=h,
        validation=validation,
        context_id=reactor_context(1).context_id,
        artifact_store=store,
    )

    assert record.kind is MechanismKind.SYMBOLIC_RULE
    assert record.status is MechanismStatus.ADMITTED
    assert record.transfer is MechanismTransferStatus.CONTEXT_SPECIALIZED
    assert record.scope.context_ids == (
        "discoveryworld:reactor-lab:normal:seed-1",
    )
    assert record.scope.regime_ids == ("normal",)
    assert record.scope.domain_ids == ("discoveryworld",)
    assert record.scope.task_ids == ("reactor-lab",)
    assert record.relations[0].kind is MechanismRelationKind.SPECIALIZES
    assert record.relations[0].target == parent
    artifact = store.load(record.representation_artifact_id)
    assert artifact["schema"] == "ecsa.discoveryworld-reactor-rule.v1"
    assert artifact["measurement_kind"] == "density"
    assert artifact["validation"]["target_reactor_uuid"] == 404
