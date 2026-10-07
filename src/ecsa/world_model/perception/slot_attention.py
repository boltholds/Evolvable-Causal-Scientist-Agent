from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from ..contracts import RawObservation, freeze_raw_value
from .base import (
    EntityObservation,
    ObservedFeature,
    PerceptualObservation,
)


@dataclass(frozen=True)
class PerceptualSlot:
    local_slot_id: str
    embedding: tuple[float, ...]
    spatial_support: tuple[float, ...] | None
    confidence: float
    provenance_id: str

    def __post_init__(self) -> None:
        if not self.local_slot_id or not self.provenance_id:
            raise ValueError("slot id and provenance are required")
        if not isinstance(self.embedding, tuple) or not self.embedding:
            raise ValueError("slot embedding must be nonempty")
        if not all(isfinite(float(value)) for value in self.embedding):
            raise ValueError("slot embedding must be finite")
        if self.spatial_support is not None and (
            not isinstance(self.spatial_support, tuple)
            or not all(
                isfinite(float(value))
                for value in self.spatial_support
            )
        ):
            raise ValueError("slot spatial support must be finite")
        if (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("slot confidence must be finite in [0,1]")


class SlotEncoder(Protocol):
    def encode(
        self,
        raw_observation: RawObservation,
    ) -> tuple[PerceptualSlot, ...]: ...


class SlotAttentionFrontend:
    """Backend-neutral Slot Attention boundary.

    A real neural encoder is intentionally injected. This module does not
    import torch and does not claim slot identity across observations.
    """

    def __init__(self, encoder: SlotEncoder) -> None:
        self._encoder = encoder

    def perceive(
        self,
        raw_observation: RawObservation,
    ) -> PerceptualObservation:
        slots = tuple(self._encoder.encode(raw_observation))
        entities: list[EntityObservation] = []
        for slot in sorted(slots, key=lambda value: value.local_slot_id):
            features = [
                ObservedFeature(
                    feature_id="appearance.embedding",
                    value=freeze_raw_value(list(slot.embedding)),
                    provenance_id=slot.provenance_id,
                ),
                ObservedFeature(
                    feature_id="perception.confidence",
                    value=freeze_raw_value(slot.confidence),
                    provenance_id=slot.provenance_id,
                ),
            ]
            if slot.spatial_support is not None:
                features.append(
                    ObservedFeature(
                        feature_id="appearance.spatial_support",
                        value=freeze_raw_value(
                            list(slot.spatial_support)
                        ),
                        provenance_id=slot.provenance_id,
                    )
                )
            entities.append(
                EntityObservation(
                    local_ref=f"slot:{slot.local_slot_id}",
                    source_identity=None,
                    features=tuple(features),
                    provenance_id=slot.provenance_id,
                )
            )
        return PerceptualObservation(
            observation_id=raw_observation.observation_id,
            entities=tuple(entities),
            global_features=(),
        )
