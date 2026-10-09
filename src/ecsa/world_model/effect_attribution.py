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


from collections import deque
from enum import StrEnum


class CollectionMode(StrEnum):
    PASSIVE = "passive"
    CONTROLLED = "controlled"


class EffectExplanation(StrEnum):
    BACKGROUND = "background"
    STATE_DEPENDENT = "state_dependent"
    ACTION_DEPENDENT = "action_dependent"
    INTERACTION = "interaction"
    UNRESOLVED = "unresolved"


class EffectEvidenceStatus(StrEnum):
    PROPOSED = "proposed"
    OBSERVATIONALLY_SUPPORTED = "observationally_supported"
    INTERVENTIONALLY_SUPPORTED = "interventionally_supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class CohortRef:
    cohort_id: str
    episode_id: str
    protocol_id: str
    collection_mode: CollectionMode
    pre_registered_prediction_id: str | None
    state_match_group: str | None
    assigned_action: "GroundAction | None"
    assignment_scheme_id: str | None

    def __post_init__(self):
        if not self.cohort_id or not self.episode_id or not self.protocol_id:
            raise ValueError("cohort / episode / protocol identifiers required")
        if not isinstance(self.collection_mode, CollectionMode):
            raise TypeError("typed collection mode required")
        if self.assigned_action is not None:
            from .contracts import GroundAction
            if not isinstance(self.assigned_action, GroundAction):
                raise TypeError("public action assignment required")


@dataclass(frozen=True)
class EffectClaim:
    effect_key: str
    action_schema_id: str
    explanation: EffectExplanation
    status: EffectEvidenceStatus
    support_ids: tuple[str, ...]
    contradiction_ids: tuple[str, ...]
    assumptions: tuple[str, ...]
    context_signatures: tuple[tuple[str, ...], ...]

    @property
    def hypothesis_id(self) -> str:
        # Stable within the current public scope; never asserts global truth.
        from hashlib import sha256
        material=(self.effect_key,self.action_schema_id,self.explanation,
                  self.context_signatures)
        return "effect:" + sha256(repr(material).encode()).hexdigest()[:24]


class EffectAttributionLedger:
    """Bounded observational contrasts, not a full do-calculus identifier."""

    def __init__(self, *, min_support: int = 3, max_recent: int = 512,
                 min_effect_contrast: float = 0.35):
        if type(min_support) is not int or min_support < 1:
            raise ValueError("min_support must be positive")
        if type(max_recent) is not int or max_recent < 1:
            raise ValueError("max_recent must be positive")
        if not isfinite(min_effect_contrast) or not 0 < min_effect_contrast < 1:
            raise ValueError("invalid effect contrast")
        self.min_support=min_support
        self.max_recent=max_recent
        self.min_effect_contrast=min_effect_contrast
        self._items: deque[tuple[FeatureTrial,CohortRef]] = deque()
        self._ids: set[str]=set()
        self.intervention_actions_executed = 0
        self._verified: dict[tuple[str,str], EffectClaim] = {}

    @property
    def recent(self) -> tuple[FeatureTrial,...]:
        return tuple(row for row,_ in self._items)

    def observe(self, trials: tuple[FeatureTrial,...], *, cohort: CohortRef) -> None:
        if not isinstance(cohort,CohortRef):
            raise TypeError("typed evidence cohort required")
        if not isinstance(trials,tuple) or not trials or not all(
            isinstance(row,FeatureTrial) for row in trials
        ):
            raise ValueError("nonempty typed trials required")
        ids=tuple(row.evidence_id for row in trials)
        if len(set(ids))!=len(ids) or any(x in self._ids for x in ids):
            raise ValueError("duplicate trial evidence")
        for row in trials:
            if len(self._items)>=self.max_recent:
                previous,_=self._items.popleft()
                self._ids.remove(previous.evidence_id)
            self._items.append((row,cohort))
            self._ids.add(row.evidence_id)

    def confirm_intervention(
        self,
        effect_key: str,
        *,
        candidate: "GroundAction",
        evidence: tuple[FeatureTrial, ...],
        cohort: CohortRef,
    ) -> EffectClaim:
        """Accept a *protocol-attested* independent randomized contrast.

        The caller must supply actual public execution evidence. The claim is
        scoped to the observed context, never an unrestricted causal proof.
        """
        from .contracts import GroundAction
        if not isinstance(candidate, GroundAction):
            raise TypeError("typed candidate action required")
        if not isinstance(cohort, CohortRef) or cohort.collection_mode is not CollectionMode.CONTROLLED:
            raise ValueError("controlled independent intervention cohort required")
        if not (cohort.pre_registered_prediction_id and cohort.assignment_scheme_id
                and cohort.state_match_group):
            raise ValueError("controlled cohort needs registered predictions and action assignment")
        if not isinstance(evidence, tuple) or not evidence or not all(
            isinstance(e, FeatureTrial) for e in evidence
        ):
            raise ValueError("independent typed intervention evidence required")
        if cohort.episode_id in {existing.episode_id for _,existing in self._items}:
            raise ValueError("disjoint intervention episode required")
        ids=tuple(e.evidence_id for e in evidence)
        if len(set(ids)) != len(ids) or self._ids.intersection(ids):
            raise ValueError("disjoint intervention trial IDs required")
        if any(e.effect_key!=effect_key or not e.observed or e.changed is None for e in evidence):
            raise ValueError("all intervention outcomes must be observed for the target")
        observed_claims=self.hypotheses(effect_key,action_schema_id=candidate.schema_id)
        if not any(c.explanation in (EffectExplanation.ACTION_DEPENDENT,EffectExplanation.INTERACTION)
                   and c.status is EffectEvidenceStatus.OBSERVATIONALLY_SUPPORTED
                   for c in observed_claims):
            raise ValueError("independent intervention requires a previously supported proposal")
        treatment=[e for e in evidence if e.action_schema_id==candidate.schema_id]
        controls=[e for e in evidence if e.action_schema_id!=candidate.schema_id]
        if len(treatment)<self.min_support or len(controls)<self.min_support:
            raise ValueError("matched treatment and controls with sufficient trials required")
        matched={e.context_signature for e in treatment}&{e.context_signature for e in controls}
        if not matched:
            raise ValueError("matched public initial contexts required")
        treatment=[e for e in treatment if e.context_signature in matched]
        controls=[e for e in controls if e.context_signature in matched]
        if len(treatment)<self.min_support or len(controls)<self.min_support:
            raise ValueError("matched intervention context has insufficient controls")
        rate_t=(1+sum(bool(e.changed) for e in treatment))/(len(treatment)+2)
        rate_c=(1+sum(bool(e.changed) for e in controls))/(len(controls)+2)
        confirmed=(rate_t-rate_c)>=self.min_effect_contrast
        result=EffectClaim(
            effect_key, candidate.schema_id, EffectExplanation.ACTION_DEPENDENT,
            (EffectEvidenceStatus.INTERVENTIONALLY_SUPPORTED if confirmed
             else EffectEvidenceStatus.CONTRADICTED),
            tuple(e.evidence_id for e in treatment+controls) if confirmed else (),
            () if confirmed else tuple(e.evidence_id for e in treatment+controls),
            ("independent externally attested controlled assignment",
             f"protocol:{cohort.protocol_id}",
             f"state_match_group:{cohort.state_match_group}",
             "valid only for tested public contexts"),
            tuple(sorted(matched)),
        )
        # Both arms count toward the actual experiment budget.
        self.intervention_actions_executed += len(evidence)
        self._verified[(effect_key,candidate.schema_id)]=result
        return result

    def hypotheses(self, effect_key: str, *, action_schema_id: str) -> tuple[EffectClaim,...]:
        if not effect_key or not action_schema_id:
            raise ValueError("effect/action identifiers required")
        trials=[row for row,_ in self._items if row.effect_key==effect_key
                and row.observed and row.changed is not None]
        target=[row for row in trials if row.action_schema_id==action_schema_id]
        controls=[row for row in trials if row.action_schema_id!=action_schema_id]
        if not target:
            return ()
        comparable=set(r.context_signature for r in target)&set(r.context_signature for r in controls)
        support=tuple(row.evidence_id for row in target)
        assumption=("public observations only", "nonrandomized action selection",
                    "matched contexts may conceal latent state differences")

        def claim(kind:EffectExplanation,status:EffectEvidenceStatus,
                  support_ids:tuple[str,...]=support):
            return EffectClaim(effect_key,action_schema_id,kind,status,
                               support_ids,(),assumption,tuple(sorted(comparable)))

        if not comparable:
            return (claim(EffectExplanation.UNRESOLVED,EffectEvidenceStatus.INSUFFICIENT),)

        contrast=[]
        pooled_target=[]
        pooled_control=[]
        for context in sorted(comparable):
            positives=[r for r in target if r.context_signature==context]
            negatives=[r for r in controls if r.context_signature==context]
            if len(positives)<self.min_support or len(negatives)<self.min_support:
                continue
            # Beta-Bernoulli smoothed predictive event rates: these
            # identify observed association, not intervention effects.
            first=(1+sum(bool(r.changed) for r in positives))/(len(positives)+2)
            other=(1+sum(bool(r.changed) for r in negatives))/(len(negatives)+2)
            contrast.append(first-other)
            pooled_target.extend(positives)
            pooled_control.extend(negatives)
        if not contrast:
            return (claim(EffectExplanation.UNRESOLVED,EffectEvidenceStatus.INSUFFICIENT),)

        threshold=self.min_effect_contrast
        if (len(contrast)>1 and max(contrast)>=threshold
            and min(contrast)<threshold/2):
            return (claim(EffectExplanation.INTERACTION,
                          EffectEvidenceStatus.OBSERVATIONALLY_SUPPORTED),)
        if sum(contrast)/len(contrast)>=threshold:
            return (claim(EffectExplanation.ACTION_DEPENDENT,
                          EffectEvidenceStatus.OBSERVATIONALLY_SUPPORTED),)
        total=pooled_target+pooled_control
        changed=sum(bool(r.changed) for r in total)
        rate=changed/len(total)
        if rate>=0.8 or rate<=0.2:
            return (claim(EffectExplanation.BACKGROUND,
                          EffectEvidenceStatus.OBSERVATIONALLY_SUPPORTED,
                          tuple(r.evidence_id for r in total)),)
        return (claim(EffectExplanation.UNRESOLVED,EffectEvidenceStatus.PROPOSED),)
