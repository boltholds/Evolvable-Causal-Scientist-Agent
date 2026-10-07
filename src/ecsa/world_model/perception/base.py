from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from typing import Protocol, runtime_checkable

from ..contracts import FrozenRawValue, RawObservation


@dataclass(frozen=True)
class ObservedFeature:
    feature_id: str
    value: FrozenRawValue
    provenance_id: str

    def __post_init__(self) -> None:
        if not self.feature_id or not self.provenance_id:
            raise ValueError("feature_id and provenance_id are required")
        if not isinstance(self.value, FrozenRawValue):
            raise ValueError("feature value must be frozen")


@dataclass(frozen=True)
class EntityObservation:
    local_ref: str
    source_identity: str | None
    features: tuple[ObservedFeature, ...]
    provenance_id: str

    def __post_init__(self) -> None:
        if not self.local_ref or not self.provenance_id:
            raise ValueError("entity local_ref and provenance_id are required")
        if self.source_identity is not None and not self.source_identity:
            raise ValueError("source_identity must be nonempty when present")
        if not isinstance(self.features, tuple) or not all(
            isinstance(feature, ObservedFeature)
            for feature in self.features
        ):
            raise ValueError("entity features must be immutable typed values")
        feature_ids = tuple(feature.feature_id for feature in self.features)
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("entity feature ids must be unique")


@dataclass(frozen=True)
class PerceptualObservation:
    observation_id: str
    entities: tuple[EntityObservation, ...]
    global_features: tuple[ObservedFeature, ...]

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("perceptual observation_id is required")
        if not isinstance(self.entities, tuple) or not all(
            isinstance(entity, EntityObservation)
            for entity in self.entities
        ):
            raise ValueError("entities must be immutable typed values")
        if not isinstance(self.global_features, tuple) or not all(
            isinstance(feature, ObservedFeature)
            for feature in self.global_features
        ):
            raise ValueError(
                "global features must be immutable typed values"
            )
        refs = tuple(entity.local_ref for entity in self.entities)
        if len(set(refs)) != len(refs):
            raise ValueError("entity local refs must be unique")


@runtime_checkable
class PerceptionFrontend(Protocol):
    def perceive(
        self,
        raw_observation: RawObservation,
    ) -> PerceptualObservation: ...


class EntityIdentityRelation(StrEnum):
    SAME = "same"
    DIFFERENT = "different"
    COMPONENT = "component"


@dataclass(frozen=True)
class EntityIdentityHypothesis:
    hypothesis_id: str
    left_ref: str
    right_ref: str
    relation: EntityIdentityRelation
    evidence_ids: tuple[str, ...]
    confidence: float

    def __post_init__(self) -> None:
        if not self.hypothesis_id or not self.left_ref or not self.right_ref:
            raise ValueError("identity hypothesis references are required")
        if not isinstance(self.relation, EntityIdentityRelation):
            raise ValueError("typed identity relation required")
        if (
            not isinstance(self.evidence_ids, tuple)
            or not self.evidence_ids
            or not all(
                isinstance(value, str) and value
                for value in self.evidence_ids
            )
        ):
            raise ValueError("identity evidence ids are required")
        if (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("identity confidence must be finite in [0,1]")


def _identity_id(
    left: EntityObservation,
    right: EntityObservation,
    relation: EntityIdentityRelation,
    evidence_id: str,
) -> str:
    material = (
        f"{left.local_ref}|{right.local_ref}|"
        f"{relation.value}|{evidence_id}"
    )
    return "entity-identity:" + sha256(material.encode()).hexdigest()


def propose_entity_identity_hypotheses(
    left: EntityObservation,
    right: EntityObservation,
    *,
    evidence_id: str,
) -> tuple[EntityIdentityHypothesis, ...]:
    if not isinstance(left, EntityObservation) or not isinstance(
        right, EntityObservation
    ):
        raise TypeError("identity proposals require entity observations")
    if not evidence_id:
        raise ValueError("evidence_id is required")

    if (
        left.source_identity is not None
        and right.source_identity is not None
    ):
        relation = (
            EntityIdentityRelation.SAME
            if left.source_identity == right.source_identity
            else EntityIdentityRelation.DIFFERENT
        )
        return (
            EntityIdentityHypothesis(
                hypothesis_id=_identity_id(
                    left,
                    right,
                    relation,
                    evidence_id,
                ),
                left_ref=left.local_ref,
                right_ref=right.local_ref,
                relation=relation,
                evidence_ids=(evidence_id,),
                confidence=1.0,
            ),
        )

    return tuple(
        EntityIdentityHypothesis(
            hypothesis_id=_identity_id(
                left,
                right,
                relation,
                evidence_id,
            ),
            left_ref=left.local_ref,
            right_ref=right.local_ref,
            relation=relation,
            evidence_ids=(evidence_id,),
            confidence=0.5,
        )
        for relation in (
            EntityIdentityRelation.SAME,
            EntityIdentityRelation.DIFFERENT,
        )
    )
