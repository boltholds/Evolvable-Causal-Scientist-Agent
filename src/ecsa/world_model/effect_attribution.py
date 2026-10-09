"""Domain-neutral comparable feature trials from the ECSA grounder.

Each trial represents an observed *candidate* change. It is not by itself
a causal effect or a complete observation of the hidden world.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .contracts import FrozenRawValue, InteractionTransition
from .grounding import GroundingUpdate
from .perception.base import PerceptualObservation, EntityObservation, ObservedFeature


@dataclass(frozen=True)
class FeatureTrial:
    evidence_id: str
    effect_key: str
    action_schema_id: str
    argument_role: str
    context_signature: tuple[str, ...]
    observed: bool
    changed: bool | None
    value_before: FrozenRawValue | None
    value_after: FrozenRawValue | None

    def __post_init__(self):
        if not all((self.evidence_id,self.effect_key,self.action_schema_id,self.argument_role)):
            raise ValueError("trial evidence identifiers required")
        if type(self.observed) is not bool:
            raise TypeError("observed must be bool")
        if self.observed and type(self.changed) is not bool:
            raise ValueError("observed trial has known change truth")
        if not self.observed and self.changed is not None:
            raise ValueError("unobserved feature cannot assert no change")


def _by_feature(features: tuple[ObservedFeature, ...]) -> dict[str, FrozenRawValue]:
    return {f.feature_id: f.value for f in features}


def _unique_identity_match(
    source: EntityObservation,
    possible: tuple[EntityObservation, ...],
) -> EntityObservation | None:
    if source.source_identity is not None:
        matches = tuple(p for p in possible if p.source_identity == source.source_identity)
    else:
        matches = tuple(p for p in possible if p.local_ref == source.local_ref)
    return matches[0] if len(matches) == 1 else None


def _role(
    before: EntityObservation,
    transition: InteractionTransition,
    all_before: tuple[EntityObservation,...],
) -> str:
    if before.source_identity is None:
        return "unbound"
    roles = []
    for i, value in enumerate(transition.action.arguments):
        v = value.thaw()
        if type(v) not in (int,str) or str(v) != before.source_identity:
            continue
        if sum(p.source_identity == before.source_identity for p in all_before) != 1:
            continue
        roles.append(f"arg[{i}]")
    return roles[0] if len(roles) == 1 else "unbound"


def _public_context(
    perception: PerceptualObservation,
    transition: InteractionTransition,
) -> tuple[str, ...]:
    items=[]
    for f in perception.global_features:
        v=f.value.thaw()
        if type(v) in (str, bool, int, float) and (
            type(v) is not float or isfinite(v)
        ):
            items.append(f"global:{f.feature_id}:{v!r}")
    for entity in perception.entities:
        role=_role(entity, transition, perception.entities)
        for f in entity.features:
            v=f.value.thaw()
            if type(v) in (str, bool, int, float) and (
                type(v) is not float or isfinite(v)
            ):
                items.append(f"{role}:{f.feature_id}:{v!r}")
    return tuple(sorted(set(items)))


class EffectTrialProjector:
    """Produce both changed and explicitly unchanged observations.

    Missing post-action identity or feature is an unknown trial, not a
    negative observation. Public UUIDs bind roles locally but never enter
    canonical effect keys.
    """
    def project(
        self,
        transition: InteractionTransition,
        grounding: GroundingUpdate,
        before: PerceptualObservation,
        after: PerceptualObservation,
    ) -> tuple[FeatureTrial, ...]:
        if not isinstance(transition, InteractionTransition):
            raise TypeError("typed transition required")
        if not isinstance(grounding, GroundingUpdate):
            raise TypeError("typed grounding required")
        if grounding.transition.transition_id != transition.transition_id:
            raise ValueError("grounding and transition provenance differ")
        if before.observation_id != transition.before.observation_id or after.observation_id != transition.after.observation_id:
            raise ValueError("perception and raw observation IDs differ")
        context=_public_context(before,transition)
        result: list[FeatureTrial]=[]

        def append(role: str, key: str, previous: FrozenRawValue,
                   current: FrozenRawValue | None, *, found: bool) -> None:
            result.append(FeatureTrial(
                evidence_id=f"{transition.transition_id}:{role}:{key}",
                effect_key=f"{role}:{key}",
                action_schema_id=transition.action.schema_id,
                argument_role=role,
                context_signature=context,
                observed=found,
                changed=(previous != current) if found else None,
                value_before=previous,
                value_after=current if found else None,
            ))

        after_globals=_by_feature(after.global_features)
        for key,value in _by_feature(before.global_features).items():
            append("global",key,value,after_globals.get(key),found=key in after_globals)

        for entity in before.entities:
            role=_role(entity,transition,before.entities)
            match=_unique_identity_match(entity,after.entities)
            now=_by_feature(match.features) if match is not None else {}
            for key,value in _by_feature(entity.features).items():
                append(role,key,value,now.get(key),found=key in now)
        # No unobserved novel post-only features are guessed to be changes.
        return tuple(result)
