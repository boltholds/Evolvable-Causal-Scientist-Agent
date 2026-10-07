from ecsa.benchmarks.discoveryworld.contracts import (
    GenericNumericHypothesis,
    GenericNumericPrediction,
    GenericScalarEvidence,
    MeasurementKind,
    ReactorMeasurement,
)
from ecsa.benchmarks.discoveryworld.reactor_lab import (
    add_generic_transfer_annotations,
    bridge_generic_numeric_hypothesis,
    build_admitted_reactor_mechanism,
    ReactorRuleArtifactStore,
)
from ecsa.benchmarks.discoveryworld.contracts import ReactorValidationEvent


def test_generic_numeric_hypothesis_bridges_only_through_public_measurements(
    tmp_path,
) -> None:
    generic = (
        GenericScalarEvidence(
            "g1", 1, "USE", (10, 20), "f", 1.0, "value 1"
        ),
        GenericScalarEvidence(
            "g2", 2, "USE", (10, 21), "f", 2.0, "value 2"
        ),
    )
    measurements = (
        ReactorMeasurement(
            "m1", 1, "ctx", 20, "subject 1", 10, "tool",
            MeasurementKind.DENSITY, (1.0,), "value 1"
        ),
        ReactorMeasurement(
            "m2", 2, "ctx", 21, "subject 2", 10, "tool",
            MeasurementKind.DENSITY, (2.0,), "value 2"
        ),
    )
    hypothesis = GenericNumericHypothesis(
        hypothesis_id="generic-h",
        transfer_key="scalar-interaction-to-control",
        feature_key="f",
        control_feature_key="c",
        polynomial_degree=1,
        coefficients=(2.0, 10.0),
        source_evidence_ids=("g1", "g2"),
        predictions=(
            GenericNumericPrediction(22, 30, 32.0, 3),
        ),
    )

    bridged = bridge_generic_numeric_hypothesis(
        hypothesis=hypothesis,
        generic_evidence=generic,
        measurements=measurements,
    )

    assert bridged.slope == 10.0
    assert bridged.offset == 2.0
    assert bridged.source_evidence_ids == ("m1", "m2")

    validation = ReactorValidationEvent(
        event_id="v1",
        hypothesis_id="generic-h",
        target_crystal_uuid=22,
        target_reactor_uuid=30,
        predicted_frequency=32.0,
        validation_step=4,
        success=True,
        observed_frequency=32.0,
    )
    mechanism = build_admitted_reactor_mechanism(
        hypothesis=bridged,
        validation=validation,
        context_id="ctx",
        artifact_store=ReactorRuleArtifactStore(tmp_path),
    )
    annotated = add_generic_transfer_annotations(
        mechanism,
        hypothesis,
    )
    params = dict(annotated.parameters)
    assert params["transfer_key"] == "scalar-interaction-to-control"
    assert params["generic_degree"] == 1



def test_generic_constant_hypothesis_can_bridge_as_zero_slope() -> None:
    generic = (
        GenericScalarEvidence(
            "g1", 1, "USE", (10, 20), "f", 7.0, "value 7"
        ),
        GenericScalarEvidence(
            "g2", 2, "USE", (10, 21), "f", 9.0, "value 9"
        ),
    )
    measurements = (
        ReactorMeasurement(
            "m1", 1, "ctx", 20, "subject 1", 10, "tool",
            MeasurementKind.DENSITY, (7.0,), "value 7"
        ),
        ReactorMeasurement(
            "m2", 2, "ctx", 21, "subject 2", 10, "tool",
            MeasurementKind.DENSITY, (9.0,), "value 9"
        ),
    )
    hypothesis = GenericNumericHypothesis(
        hypothesis_id="constant",
        transfer_key="scalar-interaction-to-control",
        feature_key="f",
        control_feature_key="c",
        polynomial_degree=0,
        coefficients=(42.0,),
        source_evidence_ids=("g1", "g2"),
        predictions=(GenericNumericPrediction(22, 30, 42.0, 3),),
    )

    bridged = bridge_generic_numeric_hypothesis(
        hypothesis=hypothesis,
        generic_evidence=generic,
        measurements=measurements,
    )

    assert bridged is not None
    assert bridged.slope == 0.0
    assert bridged.offset == 42.0


def test_generic_higher_order_hypothesis_stays_unqualified() -> None:
    hypothesis = GenericNumericHypothesis(
        hypothesis_id="quadratic",
        transfer_key="scalar-interaction-to-control",
        feature_key="f",
        control_feature_key="c",
        polynomial_degree=2,
        coefficients=(1.0, 2.0, 3.0),
        source_evidence_ids=("g1", "g2"),
        predictions=(GenericNumericPrediction(22, 30, 10.0, 3),),
    )

    assert bridge_generic_numeric_hypothesis(
        hypothesis=hypothesis,
        generic_evidence=(),
        measurements=(),
    ) is None
