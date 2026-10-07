from ecsa.benchmarks.discoveryworld.contracts import (
    GenericNumericHypothesis,
    GenericNumericPrediction,
    PolicyDecision,
    ScientificContext,
)
from ecsa.benchmarks.discoveryworld.generic_evidence import (
    GenericEvidenceLedger,
    extract_public_scalar_evidence,
)


def observation(*, message: str = "", dialog: str = "") -> dict:
    return {
        "ui": {
            "lastActionMessage": message,
            "dialog_box": {
                "dialogIn": dialog,
                "dialogOptions": {},
            },
        }
    }


def test_public_scalar_extraction_is_semantic_template_stable() -> None:
    first = extract_public_scalar_evidence(
        step=1,
        context_id="ctx",
        action={"action": "USE", "arg1": 11, "arg2": 22},
        post_observation=observation(
            message="Probe reports response 12.5 units."
        ),
    )
    second = extract_public_scalar_evidence(
        step=2,
        context_id="ctx",
        action={"action": "USE", "arg1": 11, "arg2": 23},
        post_observation=observation(
            message="Probe reports response 13.75 units."
        ),
    )

    assert len(first) == len(second) == 1
    assert first[0].feature_key == second[0].feature_key
    assert first[0].value == 12.5
    assert second[0].value == 13.75
    assert first[0].entity_ids == (11, 22)


def test_dialog_scalar_evidence_keeps_opening_entity_context() -> None:
    ledger = GenericEvidenceLedger()
    opened = ledger.record_transition(
        step=1,
        context_id="ctx",
        pre_observation=observation(),
        action={"action": "TALK", "arg1": 33},
        post_observation=observation(
            dialog=(
                "Controller #2\n"
                "Current level: 500 units.\n"
                "Range 0 to 1000."
            )
        ),
    )
    changed = ledger.record_transition(
        step=2,
        context_id="ctx",
        pre_observation=observation(
            dialog="Controller #2\nCurrent level: 500 units."
        ),
        action={"chosen_dialog_option_int": 0},
        post_observation=observation(
            dialog="Controller #2\nCurrent level: 600 units."
        ),
    )

    assert any(item.value == 500 for item in opened)
    assert any(item.value == 600 for item in changed)
    assert all(item.entity_ids == (33,) for item in changed)


def test_generic_hypothesis_can_cross_policy_boundary() -> None:
    prediction = GenericNumericPrediction(
        subject_entity_id=20,
        control_entity_id=30,
        predicted_value=42.0,
        frozen_step=3,
    )
    hypothesis = GenericNumericHypothesis(
        hypothesis_id="h1",
        transfer_key="scalar-interaction-to-control",
        feature_key="feature-a",
        control_feature_key="feature-b",
        polynomial_degree=1,
        coefficients=(2.0, 3.0),
        source_evidence_ids=("e1", "e2"),
        predictions=(prediction,),
    )
    decision = PolicyDecision(
        action={"action": "WAIT"},
        generic_hypotheses=(hypothesis,),
        generic_validation_hypothesis_ids=("h1",),
    )
    context = ScientificContext(
        context_id="ctx",
        measurements=(),
        transfer_candidates=(),
        admitted_mechanisms=(),
        generic_evidence=(),
    )

    assert decision.generic_hypotheses == (hypothesis,)
    assert decision.generic_validation_hypothesis_ids == ("h1",)
    assert context.generic_evidence == ()
