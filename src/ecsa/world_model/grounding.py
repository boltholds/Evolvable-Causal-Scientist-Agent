from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256

from .contracts import (
    FrozenRawValue,
    GroundAction,
    InteractionTransition,
)
from .perception.base import (
    EntityIdentityHypothesis,
    EntityObservation,
    PerceptualObservation,
    propose_entity_identity_hypotheses,
)


class ObservedValueKind(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    STRUCTURED = "structured"


@dataclass(frozen=True)
class FeatureDelta:
    evidence_id: str
    entity_ref: str
    feature_key: str
    source_feature_id: str
    value_kind: ObservedValueKind
    before: FrozenRawValue | None
    after: FrozenRawValue | None

    def __post_init__(self) -> None:
        if (
            not self.evidence_id
            or not self.entity_ref
            or not self.feature_key
            or not self.source_feature_id
        ):
            raise ValueError("feature delta identifiers are required")
        if not isinstance(self.value_kind, ObservedValueKind):
            raise ValueError("typed observed value kind required")


@dataclass(frozen=True)
class ActionParticipationEvidence:
    evidence_id: str
    schema_id: str
    argument_index: int
    argument_value: FrozenRawValue

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.schema_id:
            raise ValueError("participation identifiers are required")
        if type(self.argument_index) is not int or self.argument_index < 0:
            raise ValueError("argument_index must be nonnegative")
        if not isinstance(self.argument_value, FrozenRawValue):
            raise ValueError("argument value must be frozen")


@dataclass(frozen=True)
class ActionOutcomeEvidence:
    evidence_id: str
    schema_id: str
    success: bool
    arguments: tuple[FrozenRawValue, ...]

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.schema_id:
            raise ValueError("outcome evidence identifiers are required")
        if type(self.success) is not bool:
            raise ValueError("outcome success must be bool")
        if not isinstance(self.arguments, tuple) or not all(
            isinstance(value, FrozenRawValue)
            for value in self.arguments
        ):
            raise ValueError("outcome arguments must be frozen")


@dataclass(frozen=True)
class GroundedTransition:
    transition_id: str
    schema_id: str
    feature_deltas: tuple[FeatureDelta, ...]
    participation: tuple[ActionParticipationEvidence, ...]
    outcome_evidence: ActionOutcomeEvidence

    def __post_init__(self) -> None:
        if not self.transition_id or not self.schema_id:
            raise ValueError("grounded transition identifiers are required")


@dataclass(frozen=True)
class GroundingUpdate:
    transition: GroundedTransition
    identity_hypotheses: tuple[EntityIdentityHypothesis, ...]


def _evidence_id(*parts: object) -> str:
    material = "|".join(str(part) for part in parts)
    return "grounding:" + sha256(material.encode()).hexdigest()


def _anonymous_feature_key(source_feature_id: str) -> str:
    return "feature:" + sha256(source_feature_id.encode()).hexdigest()[:24]


def _observed_value_kind(value: FrozenRawValue | None) -> ObservedValueKind:
    if value is None:
        return ObservedValueKind.STRUCTURED
    thawed = value.thaw()
    if isinstance(thawed, (int, float)) and not isinstance(thawed, bool):
        return ObservedValueKind.NUMERIC
    if thawed is None or isinstance(thawed, (bool, str)):
        return ObservedValueKind.CATEGORICAL
    return ObservedValueKind.STRUCTURED


def _entity_ref(entity: EntityObservation) -> str:
    return entity.source_identity or entity.local_ref


def _feature_map(entity: EntityObservation) -> dict[str, FrozenRawValue]:
    return {
        feature.feature_id: feature.value
        for feature in entity.features
    }


class InteractionGrounder:
    def observe(
        self,
        transition: InteractionTransition,
        before: PerceptualObservation,
        after: PerceptualObservation,
    ) -> GroundingUpdate:
        if before.observation_id != transition.before.observation_id:
            raise ValueError("before perception does not match raw transition")
        if after.observation_id != transition.after.observation_id:
            raise ValueError("after perception does not match raw transition")

        identity_hypotheses = self._identity_hypotheses(
            transition,
            before,
            after,
        )
        matches = self._certain_matches(before, after)
        deltas: list[FeatureDelta] = []
        for left, right in matches:
            left_features = _feature_map(left)
            right_features = _feature_map(right)
            for feature_id in sorted(
                set(left_features) | set(right_features)
            ):
                old = left_features.get(feature_id)
                new = right_features.get(feature_id)
                if old == new:
                    continue
                evidence_id = _evidence_id(
                    transition.transition_id,
                    "feature",
                    _entity_ref(right),
                    feature_id,
                    old,
                    new,
                )
                deltas.append(
                    FeatureDelta(
                        evidence_id=evidence_id,
                        entity_ref=_entity_ref(right),
                        feature_key=_anonymous_feature_key(feature_id),
                        source_feature_id=feature_id,
                        value_kind=_observed_value_kind(
                            new if new is not None else old
                        ),
                        before=old,
                        after=new,
                    )
                )

        participation = tuple(
            ActionParticipationEvidence(
                evidence_id=_evidence_id(
                    transition.transition_id,
                    "argument",
                    index,
                    argument,
                ),
                schema_id=transition.action.schema_id,
                argument_index=index,
                argument_value=argument,
            )
            for index, argument in enumerate(
                transition.action.arguments
            )
        )
        outcome = ActionOutcomeEvidence(
            evidence_id=_evidence_id(
                transition.transition_id,
                "outcome",
                transition.action.schema_id,
                transition.outcome.success,
            ),
            schema_id=transition.action.schema_id,
            success=transition.outcome.success,
            arguments=transition.action.arguments,
        )
        return GroundingUpdate(
            transition=GroundedTransition(
                transition_id=transition.transition_id,
                schema_id=transition.action.schema_id,
                feature_deltas=tuple(deltas),
                participation=participation,
                outcome_evidence=outcome,
            ),
            identity_hypotheses=identity_hypotheses,
        )

    @staticmethod
    def _certain_matches(
        before: PerceptualObservation,
        after: PerceptualObservation,
    ) -> tuple[tuple[EntityObservation, EntityObservation], ...]:
        after_by_source = {
            entity.source_identity: entity
            for entity in after.entities
            if entity.source_identity is not None
        }
        after_by_local = {
            entity.local_ref: entity
            for entity in after.entities
        }
        found: list[
            tuple[EntityObservation, EntityObservation]
        ] = []
        seen_right: set[str] = set()
        for left in before.entities:
            right = None
            if left.source_identity is not None:
                right = after_by_source.get(left.source_identity)
            if right is None:
                right = after_by_local.get(left.local_ref)
            if right is None or right.local_ref in seen_right:
                continue
            seen_right.add(right.local_ref)
            found.append((left, right))
        return tuple(found)

    @staticmethod
    def _identity_hypotheses(
        transition: InteractionTransition,
        before: PerceptualObservation,
        after: PerceptualObservation,
    ) -> tuple[EntityIdentityHypothesis, ...]:
        found: list[EntityIdentityHypothesis] = []
        for left in before.entities:
            for right in after.entities:
                if (
                    left.source_identity is not None
                    and right.source_identity is not None
                    and left.source_identity != right.source_identity
                ):
                    continue
                evidence_id = _evidence_id(
                    transition.transition_id,
                    "identity",
                    left.local_ref,
                    right.local_ref,
                )
                found.extend(
                    propose_entity_identity_hypotheses(
                        left,
                        right,
                        evidence_id=evidence_id,
                    )
                )
        return tuple(found)
