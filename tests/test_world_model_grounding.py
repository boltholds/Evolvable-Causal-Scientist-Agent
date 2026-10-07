from ecsa.world_model.contracts import (
    GroundAction,
    InteractionTransition,
    RawActionOutcome,
    RawObservation,
    freeze_raw_value,
)
from ecsa.world_model.grounding import (
    InteractionGrounder,
    ObservedValueKind,
)
from ecsa.world_model.perception.base import (
    EntityObservation,
    ObservedFeature,
    PerceptualObservation,
)


def _raw(obs_id: str, step: int) -> RawObservation:
    return RawObservation(
        observation_id=obs_id,
        step=step,
        payload=freeze_raw_value({"frame": obs_id}),
    )


def _entity(
    *,
    local_ref: str,
    source_identity: str | None,
    value: object,
    feature_id: str = "public.sensor.value",
    provenance_id: str,
) -> EntityObservation:
    return EntityObservation(
        local_ref=local_ref,
        source_identity=source_identity,
        features=(
            ObservedFeature(
                feature_id=feature_id,
                value=freeze_raw_value(value),
                provenance_id=provenance_id,
            ),
        ),
        provenance_id=provenance_id,
    )


def _percept(
    obs_id: str,
    *entities: EntityObservation,
) -> PerceptualObservation:
    return PerceptualObservation(
        observation_id=obs_id,
        entities=tuple(entities),
        global_features=(),
    )


def _transition(
    *,
    transition_id: str = "t1",
    success: bool = True,
    arguments: tuple[object, ...] = ("left", "right"),
) -> InteractionTransition:
    return InteractionTransition(
        transition_id=transition_id,
        before=_raw("before", 1),
        action=GroundAction(
            schema_id="A17",
            arguments=tuple(
                freeze_raw_value(value)
                for value in arguments
            ),
        ),
        outcome=RawActionOutcome(
            success=success,
            payload=freeze_raw_value(
                {"message": "public outcome"}
            ),
        ),
        after=_raw("after", 2),
    )


def test_grounder_emits_anonymous_numeric_feature_delta() -> None:
    before = _percept(
        "before",
        _entity(
            local_ref="before:e0",
            source_identity="public-id:7",
            value=1.0,
            provenance_id="before",
        ),
    )
    after = _percept(
        "after",
        _entity(
            local_ref="after:e4",
            source_identity="public-id:7",
            value=2.5,
            provenance_id="after",
        ),
    )

    update = InteractionGrounder().observe(
        _transition(),
        before,
        after,
    )

    [delta] = update.transition.feature_deltas
    assert delta.entity_ref == "public-id:7"
    assert delta.value_kind is ObservedValueKind.NUMERIC
    assert delta.before.thaw() == 1.0
    assert delta.after.thaw() == 2.5
    assert delta.feature_key.startswith("feature:")
    assert "sensor" not in delta.feature_key
    assert delta.source_feature_id == "public.sensor.value"


def test_failed_action_updates_outcome_evidence_without_positive_effect() -> None:
    before_entity = _entity(
        local_ref="e0",
        source_identity="id:1",
        value=3,
        provenance_id="before",
    )
    after_entity = _entity(
        local_ref="e0",
        source_identity="id:1",
        value=3,
        provenance_id="after",
    )

    update = InteractionGrounder().observe(
        _transition(success=False),
        _percept("before", before_entity),
        _percept("after", after_entity),
    )

    assert update.transition.feature_deltas == ()
    assert update.transition.outcome_evidence.success is False
    assert update.transition.outcome_evidence.schema_id == "A17"
    assert len(update.transition.outcome_evidence.arguments) == 2


def test_binary_action_preserves_both_argument_role_hypotheses() -> None:
    grounder = InteractionGrounder()
    empty_before = _percept("before")
    empty_after = _percept("after")

    forward = grounder.observe(
        _transition(
            transition_id="forward",
            arguments=(11, 22),
        ),
        empty_before,
        empty_after,
    )
    reverse = grounder.observe(
        _transition(
            transition_id="reverse",
            arguments=(22, 11),
        ),
        empty_before,
        empty_after,
    )

    assert {
        (
            evidence.argument_index,
            evidence.argument_value.thaw(),
        )
        for evidence in forward.transition.participation
    } == {(0, 11), (1, 22)}
    assert {
        evidence.argument_value.thaw()
        for evidence in reverse.transition.participation
    } == {11, 22}
    assert all(
        not hasattr(evidence, "role")
        for evidence in forward.transition.participation
    )


def test_unchanged_identifier_like_numbers_are_not_promoted_as_effects() -> None:
    before = _percept(
        "before",
        _entity(
            local_ref="e0",
            source_identity="id:42",
            value=42,
            feature_id="public.object.identifier",
            provenance_id="before",
        ),
    )
    after = _percept(
        "after",
        _entity(
            local_ref="e9",
            source_identity="id:42",
            value=42,
            feature_id="public.object.identifier",
            provenance_id="after",
        ),
    )

    update = InteractionGrounder().observe(
        _transition(arguments=(42,)),
        before,
        after,
    )

    assert update.transition.feature_deltas == ()


def test_grounder_keeps_changed_text_as_anonymous_categorical_evidence() -> None:
    before = _percept(
        "before",
        _entity(
            local_ref="e0",
            source_identity="id:x",
            value="closed",
            feature_id="public.status",
            provenance_id="before",
        ),
    )
    after = _percept(
        "after",
        _entity(
            local_ref="e0",
            source_identity="id:x",
            value="open",
            feature_id="public.status",
            provenance_id="after",
        ),
    )

    update = InteractionGrounder().observe(
        _transition(arguments=("x",)),
        before,
        after,
    )

    [delta] = update.transition.feature_deltas
    assert delta.value_kind is ObservedValueKind.CATEGORICAL
    assert delta.before.thaw() == "closed"
    assert delta.after.thaw() == "open"
