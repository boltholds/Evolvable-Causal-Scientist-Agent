"""Typed, domain-neutral transition-to-relation acquisition.

Constructs empirical X/Y pairs from observable numeric features and grounded
actions. X is PRE-ACTION features + arguments; Y is a POST-ACTION observable.
No outcome is fabricated, entity identifiers do not become measurements, and
the module has no dependency on a neural framework or any benchmark vocabulary.

X -> encoder_X, Y -> encoder_Y, implicit relation head is supplied by an
optional learner downstream of this acquisition boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Protocol, runtime_checkable

from .contracts import FrozenRawValue, GroundAction, InteractionTransition, RawValueKind
from .perception.base import EntityObservation, ObservedFeature, PerceptualObservation


class RelationScope(StrEnum):
    GLOBAL = "global"
    ARGUMENT = "argument"


@dataclass(frozen=True)
class NumericInput:
    key: str
    value: float

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("numeric input key is required")
        if (
            not isinstance(self.value, (int, float))
            or type(self.value) is bool
            or not isfinite(float(self.value))
        ):
            raise ValueError("numeric input requires a finite number")


@dataclass(frozen=True)
class RelationTarget:
    schema_id: str
    scope: RelationScope
    feature_id: str
    argument_index: int | None

    def __post_init__(self) -> None:
        if not self.schema_id or not self.feature_id:
            raise ValueError("relation target requires a schema and feature")
        if not isinstance(self.scope, RelationScope):
            raise TypeError("typed relation scope required")
        if self.scope is RelationScope.GLOBAL and self.argument_index is not None:
            raise ValueError("global relation cannot use an argument index")
        if self.scope is RelationScope.ARGUMENT and (
            type(self.argument_index) is not int or self.argument_index < 0
        ):
            raise ValueError("argument relation requires a positional index")


@dataclass(frozen=True)
class RelationSignature:
    target: RelationTarget
    input_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target, RelationTarget)
            or not isinstance(self.input_keys, tuple)
            or not self.input_keys
            or not all(isinstance(key, str) and key for key in self.input_keys)
            or len(set(self.input_keys)) != len(self.input_keys)
        ):
            raise ValueError("relation signature requires distinct numeric keys")


@dataclass(frozen=True)
class NumericRelationSample:
    transition_id: str
    before_id: str
    after_id: str
    signature: RelationSignature
    x: tuple[NumericInput, ...]
    y: float
    success: bool
    changed: bool

    def __post_init__(self) -> None:
        if not self.transition_id or not self.before_id or not self.after_id:
            raise ValueError("relation sample requires provenance identifiers")
        if not isinstance(self.signature, RelationSignature):
            raise TypeError("typed relation signature required")
        if not isinstance(self.x, tuple) or not all(
            isinstance(feature, NumericInput) for feature in self.x
        ):
            raise TypeError("numeric inputs must be an immutable tuple")
        if tuple(feature.key for feature in self.x) != self.signature.input_keys:
            raise ValueError("input vector differs from declared signature")
        if (
            not isinstance(self.y, (int, float))
            or type(self.y) is bool
            or not isfinite(float(self.y))
        ):
            raise ValueError("numeric output requires a finite number")
        if type(self.success) is not bool or type(self.changed) is not bool:
            raise TypeError("outcome status and changed flag must be boolean")


@dataclass(frozen=True)
class RelationDataset:
    signature: RelationSignature
    samples: tuple[NumericRelationSample, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.signature, RelationSignature):
            raise TypeError("dataset signature required")
        if not isinstance(self.samples, tuple) or not self.samples:
            raise ValueError("nonempty immutable sample tuple required")
        if any(sample.signature != self.signature for sample in self.samples):
            raise ValueError("dataset mixes incompatible input/output variables")

    @property
    def x_matrix(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(feature.value) for feature in sample.x) for sample in self.samples)

    @property
    def y_vector(self) -> tuple[float, ...]:
        return tuple(float(sample.y) for sample in self.samples)


def _numeric(value: FrozenRawValue) -> float | None:
    if value.kind is not RawValueKind.SCALAR:
        return None
    raw = value.thaw()
    if type(raw) not in (float, int):
        return None
    result = float(raw)
    return result if isfinite(result) else None


def _feature_numeric(
    feature: ObservedFeature,
    entity: EntityObservation | None = None,
) -> float | None:
    value = _numeric(feature.value)
    if value is None:
        return None
    if entity is not None and (
        str(feature.value.thaw()) == entity.source_identity
        or str(feature.value.thaw()) == entity.local_ref
    ):
        # Source identity is evidence for matching, never a physical input.
        return None
    return value


def _entity_for_argument(
    action_value: FrozenRawValue,
    perception: PerceptualObservation,
) -> EntityObservation | None:
    if action_value.kind is not RawValueKind.SCALAR:
        return None
    raw = action_value.thaw()
    if type(raw) not in (str, int):
        return None
    ref = str(raw)
    matches = tuple(
        entity for entity in perception.entities
        if ref == entity.local_ref or ref == entity.source_identity
    )
    return matches[0] if len(matches) == 1 else None


def _match_after(
    source: EntityObservation,
    perception: PerceptualObservation,
) -> EntityObservation | None:
    if source.source_identity is not None:
        matches = tuple(
            entity for entity in perception.entities
            if entity.source_identity == source.source_identity
        )
        return matches[0] if len(matches) == 1 else None
    matches = tuple(
        entity for entity in perception.entities
        if entity.local_ref == source.local_ref
    )
    return matches[0] if len(matches) == 1 else None


def _numeric_features(
    features: tuple[ObservedFeature, ...],
    entity: EntityObservation | None = None,
) -> dict[str, float]:
    found: dict[str, float] = {}
    for feature in features:
        number = _feature_numeric(feature, entity)
        if number is not None:
            found[feature.feature_id] = number
    return found


class NumericRelationProjector:
    """Project observable action transitions into stable, role-relative X/Y."""

    def project(
        self,
        transition: InteractionTransition,
        before: PerceptualObservation,
        after: PerceptualObservation,
    ) -> tuple[NumericRelationSample, ...]:
        if not isinstance(transition, InteractionTransition):
            raise TypeError("typed interaction transition required")
        if not isinstance(before, PerceptualObservation) or not isinstance(
            after, PerceptualObservation
        ):
            raise TypeError("perceptual observations required")
        if (
            before.observation_id != transition.before.observation_id
            or after.observation_id != transition.after.observation_id
        ):
            raise ValueError("perception must match the raw transition")

        inputs: dict[str, float] = {}
        previous_global = _numeric_features(before.global_features)
        new_global = _numeric_features(after.global_features)
        for feature_id, value in previous_global.items():
            inputs[f"global.{feature_id}"] = value

        bound: list[tuple[int, EntityObservation, EntityObservation]] = []
        for index, argument in enumerate(transition.action.arguments):
            source = _entity_for_argument(argument, before)
            if source is None:
                number = _numeric(argument)
                if number is not None:
                    inputs[f"arg_value[{index}]"] = number
                continue
            for feature_id, value in _numeric_features(source.features, source).items():
                inputs[f"arg[{index}].{feature_id}"] = value
            target = _match_after(source, after)
            if target is not None:
                bound.append((index, source, target))

        if not inputs:
            # Fail closed: no stable numeric X; do not learn a bogus relation.
            return ()
        x = tuple(NumericInput(key, value) for key, value in sorted(inputs.items()))
        keys = tuple(item.key for item in x)

        def sample(
            target: RelationTarget,
            y: float,
            previous: float | None,
        ) -> NumericRelationSample:
            return NumericRelationSample(
                transition_id=transition.transition_id,
                before_id=before.observation_id,
                after_id=after.observation_id,
                signature=RelationSignature(target, keys),
                x=x,
                y=y,
                success=transition.outcome.success,
                changed=(previous is None or previous != y),
            )

        results: list[NumericRelationSample] = []
        for feature_id, y in sorted(new_global.items()):
            results.append(sample(
                RelationTarget(
                    transition.action.schema_id, RelationScope.GLOBAL,
                    feature_id, None,
                ),
                y, previous_global.get(feature_id),
            ))

        for index, source, target in bound:
            old = _numeric_features(source.features, source)
            new = _numeric_features(target.features, target)
            for feature_id, y in sorted(new.items()):
                results.append(sample(
                    RelationTarget(
                        transition.action.schema_id, RelationScope.ARGUMENT,
                        feature_id, index,
                    ),
                    y, old.get(feature_id),
                ))
        return tuple(results)


@runtime_checkable
class RelationSampleSink(Protocol):
    def observe_sample(self, sample: NumericRelationSample) -> None: ...


class NumericRelationAcquisition:
    """Bounded generic evidence store; optional sink may train a KAN backend."""

    def __init__(
        self,
        *,
        max_samples_per_signature: int = 512,
        sink: RelationSampleSink | None = None,
    ) -> None:
        if (
            type(max_samples_per_signature) is not int
            or max_samples_per_signature < 1
        ):
            raise ValueError("positive per-signature capacity required")
        if sink is not None and not isinstance(sink, RelationSampleSink):
            raise TypeError("sink must implement observe_sample")
        self._max_samples = max_samples_per_signature
        self._projector = NumericRelationProjector()
        self._samples: dict[RelationSignature, list[NumericRelationSample]] = {}
        self._observed_transition_ids: set[str] = set()
        self._sink = sink

    @property
    def samples(self) -> tuple[NumericRelationSample, ...]:
        return tuple(
            sample
            for group in self._samples.values()
            for sample in group
        )

    @property
    def total_samples(self) -> int:
        return sum(len(group) for group in self._samples.values())

    def observe_transition(
        self,
        transition: InteractionTransition,
        before: PerceptualObservation,
        after: PerceptualObservation,
    ) -> tuple[NumericRelationSample, ...]:
        if not isinstance(transition, InteractionTransition):
            raise TypeError("typed transition required")
        if transition.transition_id in self._observed_transition_ids:
            return ()
        results = self._projector.project(transition, before, after)
        self._observed_transition_ids.add(transition.transition_id)
        for item in results:
            group = self._samples.setdefault(item.signature, [])
            group.append(item)
            if len(group) > self._max_samples:
                del group[:len(group) - self._max_samples]
            if self._sink is not None:
                self._sink.observe_sample(item)
        return results

    def ready_datasets(
        self,
        *,
        min_examples: int = 24,
        min_unique_x: int = 8,
        success_only: bool = True,
    ) -> tuple[RelationDataset, ...]:
        if (
            type(min_examples) is not int or min_examples < 1
            or type(min_unique_x) is not int or min_unique_x < 1
            or type(success_only) is not bool
        ):
            raise ValueError("positive thresholds and boolean success_only required")
        found: list[RelationDataset] = []
        for signature, group in sorted(self._samples.items(), key=lambda item: repr(item[0])):
            selected = tuple(
                sample for sample in group
                if sample.success or not success_only
            )
            if len(selected) < min_examples:
                continue
            independent_x = {
                tuple(feature.value for feature in sample.x) for sample in selected
            }
            if len(independent_x) < min_unique_x:
                continue
            found.append(RelationDataset(signature, selected))
        return tuple(found)
