"""One frozen, representation-independent frame of publicly observable outcomes.

All predictors receive exactly the same targets. Nothing from validation or
heldout evidence changes the training-derived vocabulary, normalization or
coordinate ordering. Missingness is not encoded as "unchanged".
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
from math import isfinite
from statistics import mean, pstdev

from .contracts import InteractionTransition
from .perception.base import PerceptualObservation
from .grounding import InteractionGrounder
from .effect_attribution import EffectTrialProjector, FeatureTrial


class TargetKind(StrEnum):
    NUMERIC_DELTA = "numeric_delta"
    BOOLEAN_CHANGE = "boolean_change"
    CATEGORICAL_CHANGE = "categorical_change"
    ACTION_SUCCESS = "action_success"


@dataclass(frozen=True)
class PublicOutcomeTransition:
    transition: InteractionTransition
    before: PerceptualObservation
    after: PerceptualObservation

    def __post_init__(self):
        if not isinstance(self.transition,InteractionTransition):
            raise TypeError("typed public transition required")
        if not isinstance(self.before,PerceptualObservation) or not isinstance(self.after,PerceptualObservation):
            raise TypeError("typed perception required")
        if (self.before.observation_id!=self.transition.before.observation_id or
            self.after.observation_id!=self.transition.after.observation_id):
            raise ValueError("public observation/provenance mismatch")


@dataclass(frozen=True)
class OutcomeCoordinate:
    key: str
    role: str
    kind: TargetKind
    train_center: float
    train_scale: float

    def __post_init__(self):
        if not self.key or not self.role or not isinstance(self.kind,TargetKind):
            raise ValueError("typed outcome coordinate required")
        if not (isfinite(self.train_center) and isfinite(self.train_scale) and self.train_scale>0):
            raise ValueError("finite positive training scale required")


@dataclass(frozen=True)
class SharedTarget:
    values: tuple[float,...]
    observed_mask: tuple[bool,...]
    unseen_feature_count: int
    coordinate_ids: tuple[str,...]

    def __post_init__(self):
        if not (len(self.values)==len(self.observed_mask)==len(self.coordinate_ids)):
            raise ValueError("coordinate/mask width mismatch")
        if any(not isfinite(value) for value in self.values):
            raise ValueError("finite target values required")
        if any(type(v) is not bool for v in self.observed_mask):
            raise TypeError("target mask must be boolean")
        if type(self.unseen_feature_count) is not int or self.unseen_feature_count<0:
            raise ValueError("nonnegative unseen count required")


def _trials(item:PublicOutcomeTransition)->tuple[FeatureTrial,...]:
    grounding=InteractionGrounder().observe(item.transition,item.before,item.after)
    return EffectTrialProjector().project(item.transition,grounding,item.before,item.after)


def _kind_value(trial:FeatureTrial)->tuple[TargetKind,float]|None:
    if not trial.observed or trial.value_before is None or trial.value_after is None:
        return None
    old=trial.value_before.thaw();new=trial.value_after.thaw()
    if type(old) is bool and type(new) is bool:
        return (TargetKind.BOOLEAN_CHANGE,float(old!=new))
    if type(old) in (int,float) and type(new) in (int,float):
        value=float(new)-float(old)
        if not isfinite(value):
            raise ValueError("nonfinite observed public delta")
        return (TargetKind.NUMERIC_DELTA,value)
    return (TargetKind.CATEGORICAL_CHANGE,float(old!=new))


def _row(item:PublicOutcomeTransition)->dict[str,tuple[TargetKind,float]]:
    grouped:dict[str,list[tuple[TargetKind,float]]]=defaultdict(list)
    for trial in _trials(item):
        sample=_kind_value(trial)
        if sample is not None:
            grouped[trial.effect_key].append(sample)
    summarized={}
    for key,values in grouped.items():
        kinds={kind for kind,_ in values}
        if len(kinds)!=1:
            continue  # incoherent multiplicity must not become a fabricated target
        kind=values[0][0]
        # Multiple unbound entities are compared as an order-independent
        # multiset mean/changed fraction; no UUID or list index is a feature.
        summarized[key]=(kind,mean(value for _,value in values))
    summarized["action:success"]=(TargetKind.ACTION_SUCCESS,float(item.transition.outcome.success))
    return summarized


class SharedOutcomeFrame:
    def __init__(self,coordinates:tuple[OutcomeCoordinate,...]):
        if not coordinates:
            raise ValueError("nonempty training outcome vocabulary required")
        keys=tuple(c.key for c in coordinates)
        if len(set(keys))!=len(keys):
            raise ValueError("unique coordinate keys required")
        self.coordinates=coordinates
        manifest=[(c.key,c.role,c.kind.value,c.train_center,c.train_scale) for c in coordinates]
        self.fingerprint="shared-public-outcomes:"+sha256(
            json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()
        ).hexdigest()

    @classmethod
    def fit(cls,train:tuple[PublicOutcomeTransition,...])->"SharedOutcomeFrame":
        if not isinstance(train,tuple) or not train:
            raise ValueError("nonempty immutable training transitions required")
        if any(not isinstance(item,PublicOutcomeTransition) for item in train):
            raise TypeError("typed public training transitions required")
        ids=[item.transition.transition_id for item in train]
        if len(set(ids))!=len(ids):
            raise ValueError("duplicate training transition IDs")
        data:dict[str,list[float]]=defaultdict(list)
        types:dict[str,TargetKind]={}
        for item in train:
            for key,(kind,value) in _row(item).items():
                if key in types and types[key] is not kind:
                    # Mixed types are unresolved in this feature contract.
                    continue
                types.setdefault(key,kind)
                data[key].append(value)
        coordinates=[]
        for key in sorted(types):
            kind=types[key]
            values=data[key]
            center=mean(values) if kind is TargetKind.NUMERIC_DELTA else 0.
            std=pstdev(values) if len(values)>1 else 0.
            scale=(std if std>1e-10 else 1.) if kind is TargetKind.NUMERIC_DELTA else 1.
            coordinates.append(OutcomeCoordinate(key,key.split(":",1)[0],kind,float(center),float(scale)))
        return cls(tuple(coordinates))

    def project(self,item:PublicOutcomeTransition)->SharedTarget:
        if not isinstance(item,PublicOutcomeTransition):
            raise TypeError("typed public transition required")
        target=_row(item)
        known={c.key for c in self.coordinates}
        # Also count post-only features absent from the train frame. They are
        # unknown, not backfilled into the immutable vocabulary.
        post_only={
            "global:"+f.feature_id for f in item.after.global_features
            if "global:"+f.feature_id not in known
        }
        unseen=len((set(target)-known)|post_only)
        values=[];mask=[]
        for coordinate in self.coordinates:
            datum=target.get(coordinate.key)
            if datum is None or datum[0] is not coordinate.kind:
                values.append(0.)
                mask.append(False)
            else:
                value=datum[1]
                if coordinate.kind is TargetKind.NUMERIC_DELTA:
                    value=(value-coordinate.train_center)/coordinate.train_scale
                values.append(float(value))
                mask.append(True)
        return SharedTarget(tuple(values),tuple(mask),unseen,
                            tuple(c.key for c in self.coordinates))
