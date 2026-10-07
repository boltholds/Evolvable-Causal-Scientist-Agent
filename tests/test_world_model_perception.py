from __future__ import annotations

from dataclasses import dataclass

from ecsa.world_model.contracts import (
    RawObservation,
    freeze_raw_value,
)
from ecsa.world_model.perception.base import (
    EntityIdentityRelation,
    EntityObservation,
    ObservedFeature,
    PerceptualObservation,
    propose_entity_identity_hypotheses,
)
from ecsa.world_model.perception.slot_attention import (
    PerceptualSlot,
    SlotAttentionFrontend,
)
from ecsa.world_model.perception.structured import (
    StructuredEntityRecord,
    StructuredObservationFrontend,
)


def _raw(observation_id: str = "obs-1") -> RawObservation:
    return RawObservation(
        observation_id=observation_id,
        step=1,
        payload=freeze_raw_value({"public": "payload"}),
    )


@dataclass
class FakeStructuredDecoder:
    with_identity: bool = True

    def decode_entities(
        self,
        raw_observation: RawObservation,
    ) -> tuple[StructuredEntityRecord, ...]:
        return (
            StructuredEntityRecord(
                local_ref="visible-0",
                source_identity=(
                    "public-id:7"
                    if self.with_identity
                    else None
                ),
                features=(
                    ObservedFeature(
                        feature_id="appearance.embedding",
                        value=freeze_raw_value([0.25, 0.75]),
                        provenance_id=raw_observation.observation_id,
                    ),
                ),
            ),
        )


class FakeSlotEncoder:
    def __init__(self, reverse: bool = False) -> None:
        self.reverse = reverse

    def encode(
        self,
        raw_observation: RawObservation,
    ) -> tuple[PerceptualSlot, ...]:
        slots = (
            PerceptualSlot(
                local_slot_id="s0",
                embedding=(0.25, 0.75),
                spatial_support=(0.1, 0.2, 0.3, 0.4),
                confidence=0.9,
                provenance_id=raw_observation.observation_id,
            ),
            PerceptualSlot(
                local_slot_id="s1",
                embedding=(0.9, 0.1),
                spatial_support=None,
                confidence=0.8,
                provenance_id=raw_observation.observation_id,
            ),
        )
        return tuple(reversed(slots)) if self.reverse else slots


def test_structured_frontend_preserves_public_id_as_evidence_not_required_identity() -> None:
    identified = StructuredObservationFrontend(
        FakeStructuredDecoder(with_identity=True)
    ).perceive(_raw("identified"))
    anonymous = StructuredObservationFrontend(
        FakeStructuredDecoder(with_identity=False)
    ).perceive(_raw("anonymous"))

    assert identified.entities[0].source_identity == "public-id:7"
    assert anonymous.entities[0].source_identity is None
    assert identified.entities[0].features[0].feature_id == (
        "appearance.embedding"
    )
    assert isinstance(identified.entities[0], EntityObservation)


def test_slot_frontend_accepts_permuted_slots() -> None:
    forward = SlotAttentionFrontend(
        FakeSlotEncoder(reverse=False)
    ).perceive(_raw("frame"))
    reversed_slots = SlotAttentionFrontend(
        FakeSlotEncoder(reverse=True)
    ).perceive(_raw("frame"))

    assert tuple(
        entity.local_ref for entity in forward.entities
    ) == ("slot:s0", "slot:s1")
    assert reversed_slots == forward
    assert all(
        entity.source_identity is None
        for entity in forward.entities
    )


def test_structured_and_slot_frontends_emit_compatible_entity_observations() -> None:
    structured = StructuredObservationFrontend(
        FakeStructuredDecoder()
    ).perceive(_raw("same-frame"))
    slotted = SlotAttentionFrontend(
        FakeSlotEncoder()
    ).perceive(_raw("same-frame"))

    assert isinstance(structured, PerceptualObservation)
    assert isinstance(slotted, PerceptualObservation)
    assert isinstance(structured.entities[0], EntityObservation)
    assert isinstance(slotted.entities[0], EntityObservation)

    structured_feature = structured.entities[0].features[0]
    slot_feature = next(
        feature
        for feature in slotted.entities[0].features
        if feature.feature_id == "appearance.embedding"
    )
    assert structured_feature.value.thaw() == (
        slot_feature.value.thaw()
    )


def test_missing_stable_id_preserves_competing_identity_hypotheses() -> None:
    left = EntityObservation(
        local_ref="frame-1:slot-0",
        source_identity=None,
        features=(),
        provenance_id="frame-1",
    )
    right = EntityObservation(
        local_ref="frame-2:slot-3",
        source_identity=None,
        features=(),
        provenance_id="frame-2",
    )

    hypotheses = propose_entity_identity_hypotheses(
        left,
        right,
        evidence_id="temporal-adjacency",
    )

    assert {
        hypothesis.relation
        for hypothesis in hypotheses
    } == {
        EntityIdentityRelation.SAME,
        EntityIdentityRelation.DIFFERENT,
    }
    assert sum(
        hypothesis.confidence
        for hypothesis in hypotheses
    ) == 1.0
    assert all(
        hypothesis.evidence_ids == ("temporal-adjacency",)
        for hypothesis in hypotheses
    )


def test_matching_public_identity_collapses_identity_alternative() -> None:
    left = EntityObservation(
        local_ref="left",
        source_identity="public-id:42",
        features=(),
        provenance_id="obs-a",
    )
    right = EntityObservation(
        local_ref="right",
        source_identity="public-id:42",
        features=(),
        provenance_id="obs-b",
    )

    hypotheses = propose_entity_identity_hypotheses(
        left,
        right,
        evidence_id="stable-id",
    )

    assert len(hypotheses) == 1
    assert hypotheses[0].relation is EntityIdentityRelation.SAME
    assert hypotheses[0].confidence == 1.0
