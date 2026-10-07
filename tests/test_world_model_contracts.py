import pytest

from ecsa.world_model.contracts import (
    FrozenRawValue,
    GroundAction,
    InteractionTransition,
    RawActionOutcome,
    RawActionParameter,
    RawActionSchema,
    RawObservation,
    freeze_raw_value,
)


def _schema(*names: str) -> RawActionSchema:
    return RawActionSchema(
        schema_id="A17",
        parameters=tuple(
            RawActionParameter(
                name=name,
                public_candidates=(
                    freeze_raw_value("candidate"),
                ),
            )
            for name in names
        ),
        public_metadata=freeze_raw_value(
            {
                "description": "public action",
                "rank": 3,
            }
        ),
    )


def test_interaction_transition_is_immutable_and_lossless() -> None:
    before = RawObservation(
        observation_id="obs-1",
        step=3,
        payload=freeze_raw_value(
            {
                "entities": [
                    {"id": 7, "name": "object-a"},
                    {"id": 9, "name": "object-b"},
                ],
                "signal": 0.0,
            }
        ),
    )
    action = GroundAction(
        schema_id="A17",
        arguments=(
            freeze_raw_value(7),
            freeze_raw_value(9),
        ),
    )
    outcome = RawActionOutcome(
        success=True,
        payload=freeze_raw_value(
            {
                "message": "changed",
                "value": 12.5,
            }
        ),
    )
    after = RawObservation(
        observation_id="obs-2",
        step=4,
        payload=freeze_raw_value(
            {
                "entities": [
                    {"id": 7, "name": "object-a"},
                    {"id": 9, "name": "object-b"},
                ],
                "signal": 12.5,
            }
        ),
    )

    transition = InteractionTransition(
        transition_id="transition-1",
        before=before,
        action=action,
        outcome=outcome,
        after=after,
    )

    assert transition.before.payload.thaw()["signal"] == 0.0
    assert transition.outcome.payload.thaw() == {
        "message": "changed",
        "value": 12.5,
    }
    assert transition.after.payload.thaw()["signal"] == 12.5

    with pytest.raises(AttributeError):
        transition.transition_id = "changed"


def test_failed_action_is_a_valid_raw_outcome() -> None:
    outcome = RawActionOutcome(
        success=False,
        payload=freeze_raw_value(
            {
                "errors": ["argument combination rejected"],
                "code": 17,
            }
        ),
    )

    assert outcome.success is False
    assert outcome.payload.thaw()["errors"] == [
        "argument combination rejected"
    ]


def test_ground_action_rejects_wrong_arity() -> None:
    schema = _schema("arg0", "arg1")

    valid = GroundAction(
        schema_id="A17",
        arguments=(
            freeze_raw_value("x"),
            freeze_raw_value("y"),
        ),
    )
    valid.validate_against(schema)

    invalid = GroundAction(
        schema_id="A17",
        arguments=(freeze_raw_value("x"),),
    )
    with pytest.raises(ValueError, match="arity"):
        invalid.validate_against(schema)


def test_ground_action_rejects_wrong_schema_id() -> None:
    action = GroundAction(
        schema_id="OTHER",
        arguments=(freeze_raw_value("x"),),
    )

    with pytest.raises(ValueError, match="schema"):
        action.validate_against(_schema("arg0"))


def test_raw_action_schema_carries_public_candidates_without_semantic_roles() -> None:
    schema = _schema("left", "right")

    assert schema.schema_id == "A17"
    assert tuple(parameter.name for parameter in schema.parameters) == (
        "left",
        "right",
    )
    assert schema.parameters[0].public_candidates[0].thaw() == "candidate"
    assert schema.public_metadata.thaw() == {
        "description": "public action",
        "rank": 3,
    }
    assert not hasattr(schema, "role")


def test_freeze_raw_value_is_deterministic_for_mapping_order() -> None:
    left = freeze_raw_value({"b": 2, "a": [1, True, None]})
    right = freeze_raw_value({"a": [1, True, None], "b": 2})

    assert isinstance(left, FrozenRawValue)
    assert left == right
    assert left.thaw() == {
        "a": [1, True, None],
        "b": 2,
    }


def test_raw_observation_rejects_negative_step() -> None:
    with pytest.raises(ValueError, match="step"):
        RawObservation(
            observation_id="obs",
            step=-1,
            payload=freeze_raw_value({}),
        )
