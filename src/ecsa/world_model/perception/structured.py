from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..contracts import RawObservation
from .base import (
    EntityObservation,
    ObservedFeature,
    PerceptualObservation,
)


@dataclass(frozen=True)
class StructuredEntityRecord:
    local_ref: str
    source_identity: str | None
    features: tuple[ObservedFeature, ...]

    def __post_init__(self) -> None:
        if not self.local_ref:
            raise ValueError("structured local_ref is required")
        if self.source_identity is not None and not self.source_identity:
            raise ValueError("source_identity must be nonempty when present")
        if not isinstance(self.features, tuple) or not all(
            isinstance(feature, ObservedFeature)
            for feature in self.features
        ):
            raise ValueError("structured features must be typed tuples")


class StructuredEntityDecoder(Protocol):
    def decode_entities(
        self,
        raw_observation: RawObservation,
    ) -> tuple[StructuredEntityRecord, ...]: ...


class StructuredObservationFrontend:
    def __init__(self, decoder: StructuredEntityDecoder) -> None:
        self._decoder = decoder

    def perceive(
        self,
        raw_observation: RawObservation,
    ) -> PerceptualObservation:
        records = tuple(self._decoder.decode_entities(raw_observation))
        entities = tuple(
            EntityObservation(
                local_ref=record.local_ref,
                source_identity=record.source_identity,
                features=record.features,
                provenance_id=raw_observation.observation_id,
            )
            for record in sorted(
                records,
                key=lambda value: value.local_ref,
            )
        )
        return PerceptualObservation(
            observation_id=raw_observation.observation_id,
            entities=entities,
            global_features=(),
        )
