from pathlib import Path

import pytest

from ecsa.benchmarks.discoveryworld.contracts import (
    MeasurementKind,
    ReactorMeasurement,
    ScientificContext,
)
from ecsa.benchmarks.discoveryworld.policies.openai_compatible import (
    OpenAICompatiblePolicy,
    create_policy,
)
from ecsa.mechanisms import (
    EpistemicStatus,
    MechanismKind,
    MechanismRecord,
    MechanismScope,
    TransferStatus,
)


def _candidate() -> MechanismRecord:
    return MechanismRecord(
        mechanism_id="source-law",
        version=1,
        kind=MechanismKind.SYMBOLIC_RULE,
        epistemic_status=EpistemicStatus.ADMITTED,
        representation_artifact="reactor-rule:sha256:" + "a" * 64,
        scope=MechanismScope(
            context_ids=(
                "discoveryworld:reactor-lab:normal:seed-0",
            ),
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
            ("measurement_kind", "density"),
            ("slope", 96.0),
            ("offset", 102.0),
        ),
        supporting_evidence=("source-evidence",),
    )


def _measurement() -> ReactorMeasurement:
    return ReactorMeasurement(
        evidence_id="target-evidence",
        step=0,
        context_id="discoveryworld:reactor-lab:normal:seed-1",
        crystal_uuid=20,
        crystal_name="quantum crystal",
        instrument_uuid=10,
        instrument_name="densitometer",
        kind=MeasurementKind.DENSITY,
        values=(52.62,),
        raw_message=(
            "The density is approximately 52.62 grams per cubic centimeter."
        ),
    )


def _observation() -> dict:
    return {
        "ui": {
            "taskProgress": [{"description": "Tune reactors"}],
            "lastActionMessage": "",
            "dialog_box": {},
            "inventoryObjects": [],
            "accessibleEnvironmentObjects": [
                {"uuid": 10, "name": "densitometer"},
                {"uuid": 20, "name": "quantum crystal"},
                {
                    "uuid": 30,
                    "name": "crystal reactor (uncalibrated)",
                },
            ],
            "nearbyObjects": {"objects": {}},
        }
    }


class StubPolicy(OpenAICompatiblePolicy):
    def __init__(self, responses):
        super().__init__(
            {
                "model": "stub",
                "max_attempts": 1,
            }
        )
        self.responses = list(responses)
        self.payloads = []

    def _complete_json(self, payload):
        self.payloads.append(payload)
        return self.responses.pop(0)


def test_model_policy_builds_typed_transfer_hypothesis() -> None:
    source = _candidate()
    policy = StubPolicy(
        [
            {
                "action": {
                    "action": "USE",
                    "arg1": 10,
                    "arg2": 20,
                },
                "reasoning": "measure target",
                "memory": "density measured",
                "hypotheses": [
                    {
                        "hypothesis_id": "h-target",
                        "measurement_kind": "density",
                        "slope": 96.0,
                        "offset": 102.0,
                        "source_evidence_ids": [
                            "source-evidence",
                            "target-evidence",
                        ],
                        "predictions": [
                            {
                                "target_crystal_uuid": 20,
                                "target_reactor_uuid": 30,
                                "predicted_frequency": 5153.52,
                            }
                        ],
                        "source_mechanism": {
                            "mechanism_id": "source-law",
                            "version": 1,
                        },
                    }
                ],
                "validation_hypothesis_ids": [],
            }
        ]
    )
    context = ScientificContext(
        context_id="discoveryworld:reactor-lab:normal:seed-1",
        measurements=(_measurement(),),
        transfer_candidates=(source,),
        admitted_mechanisms=(),
    )

    decision = policy.decide(
        _observation(),
        {"USE": {"args": ["arg1", "arg2"]}},
        {"quantum reactor lab": [1, 1]},
        context,
    )

    assert decision.action["action"] == "USE"
    [hypothesis] = decision.hypotheses
    assert hypothesis.source_mechanism == source.ref
    assert hypothesis.source_evidence_ids == (
        "source-evidence",
        "target-evidence",
    )
    assert hypothesis.predictions[0].frozen_step == 0
    assert policy.payloads[0]["scientific_context"][
        "transfer_candidates"
    ][0]["mechanism_id"] == "source-law"


def test_model_policy_can_validate_prior_hypothesis_on_later_turn() -> None:
    policy = StubPolicy(
        [
            {
                "action": {"action": "USE", "arg1": 10, "arg2": 20},
                "reasoning": None,
                "memory": "remember",
                "hypotheses": [
                    {
                        "hypothesis_id": "h1",
                        "measurement_kind": "density",
                        "slope": 1.0,
                        "offset": 0.0,
                        "source_evidence_ids": ["target-evidence"],
                        "predictions": [
                            {
                                "target_crystal_uuid": 20,
                                "target_reactor_uuid": 30,
                                "predicted_frequency": 52.62,
                            }
                        ],
                        "source_mechanism": None,
                    }
                ],
                "validation_hypothesis_ids": [],
            },
            {
                "action": {"chosen_dialog_option_int": 0},
                "reasoning": "test prediction",
                "memory": None,
                "hypotheses": [],
                "validation_hypothesis_ids": ["h1"],
            },
        ]
    )
    context = ScientificContext(
        context_id="ctx",
        measurements=(_measurement(),),
        transfer_candidates=(),
        admitted_mechanisms=(),
    )

    first = policy.decide(
        _observation(),
        {"USE": {}},
        {},
        context,
    )
    second = policy.decide(
        _observation(),
        {},
        {},
        context,
    )

    assert first.hypotheses[0].predictions[0].frozen_step == 0
    assert second.validation_hypothesis_ids == ("h1",)
    assert second.memory == "remember"


def test_model_policy_rejects_unavailable_action() -> None:
    policy = StubPolicy(
        [
            {
                "action": {"action": "FLY"},
                "reasoning": None,
                "memory": None,
                "hypotheses": [],
                "validation_hypothesis_ids": [],
            }
        ]
    )

    with pytest.raises(RuntimeError, match="unavailable action"):
        policy.decide(
            _observation(),
            {"USE": {}},
            {},
            ScientificContext("ctx", (), (), ()),
        )


def test_model_policy_rejects_fake_transfer_source() -> None:
    policy = StubPolicy(
        [
            {
                "action": {"action": "USE", "arg1": 10, "arg2": 20},
                "reasoning": None,
                "memory": None,
                "hypotheses": [
                    {
                        "hypothesis_id": "h1",
                        "measurement_kind": "density",
                        "slope": 1.0,
                        "offset": 0.0,
                        "source_evidence_ids": ["target-evidence"],
                        "predictions": [
                            {
                                "target_crystal_uuid": 20,
                                "target_reactor_uuid": 30,
                                "predicted_frequency": 52.62,
                            }
                        ],
                        "source_mechanism": {
                            "mechanism_id": "invented",
                            "version": 1,
                        },
                    }
                ],
                "validation_hypothesis_ids": [],
            }
        ]
    )

    with pytest.raises(RuntimeError, match="transfer candidate"):
        policy.decide(
            _observation(),
            {"USE": {}},
            {},
            ScientificContext(
                "ctx",
                (_measurement(),),
                (_candidate(),),
                (),
            ),
        )


def test_policy_factory_is_provider_neutral() -> None:
    policy = create_policy(
        {
            "model": "provider/model",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "EXAMPLE_API_KEY",
        }
    )
    assert isinstance(policy, OpenAICompatiblePolicy)
    assert policy.model == "provider/model"
    assert policy.base_url == "https://example.invalid/v1"
