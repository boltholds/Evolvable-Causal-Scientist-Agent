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


@dataclass(frozen=True)
class SharedForecast:
    values: tuple[float,...]
    mask: tuple[bool,...]
    coordinate_ids: tuple[str,...]
    frame_fingerprint: str

    def __post_init__(self):
        if not self.frame_fingerprint.startswith("shared-public-outcomes:"):
            raise ValueError("valid shared outcome fingerprint required")
        if not len(self.values)==len(self.mask)==len(self.coordinate_ids):
            raise ValueError("forecast dimensions differ")
        if any(not isfinite(v) for v in self.values):
            raise ValueError("nonfinite forecast")
        if len(set(self.coordinate_ids))!=len(self.coordinate_ids):
            raise ValueError("duplicate forecast coordinate")
        if any(c=="action:success" and not 0<=v<=1
               for c,v in zip(self.coordinate_ids,self.values)):
            raise ValueError("action success is a probability")



@dataclass(frozen=True)
class BackendScore:
    row_count: int
    scored_numeric_count: int
    scored_binary_count: int
    numeric_normalized_mse: float | None
    binary_brier: float | None
    binary_change_f1: float | None
    action_success_brier: float | None
    per_action: dict[str, float]
    per_feature: dict[str, float]
    coverage: float
    abstentions: int
    unseen_feature_count: int
    interventional_confirmation_count: int


class SharedEvaluation:
    """Evaluate heldout public targets with a fixed common observation mask.

    No forecast is allowed to adjust the truth denominator. Missing forecasts
    cause incomplete coverage and noncomparable global error, not a smaller
    evaluation set.
    """
    @staticmethod
    def score(
        truth: tuple[SharedTarget,...],
        forecasts: dict[str,tuple[SharedForecast,...]],
        *,
        actions: tuple[str,...],
        intervention_records: tuple[object,...]=(),
    ) -> dict[str,BackendScore]:
        if not truth or len(actions)!=len(truth) or not forecasts:
            raise ValueError("shared evaluation needs matched nonempty trials")
        frame_ids=truth[0].coordinate_ids
        if any(t.coordinate_ids!=frame_ids for t in truth):
            raise ValueError("common target frame dimensions required")
        frames={
            f.frame_fingerprint for collection in forecasts.values()
            for f in collection
        }
        if len(frames)!=1:
            raise ValueError("forecast frame mismatch between predictors")
        confirmed=sum(
            getattr(claim,"status",None).value=="interventionally_supported"
            for claim in intervention_records
            if getattr(claim,"status",None) is not None
        )
        output:dict[str,BackendScore]={}
        for name, predicted in forecasts.items():
            if len(predicted)!=len(truth):
                raise ValueError("all predictors require identical heldout rows")
            squared:dict[str,list[float]]={key:[] for key in frame_ids}
            by_action:dict[str,list[float]]={}
            binary_errors=[]
            binary_true_positive=0
            binary_false_positive=0
            binary_false_negative=0
            success_errors=[]
            missing=0
            available=0
            abstentions=0
            numeric_count=0
            binary_count=0
            # Determine binary data *only* by a target value stream and
            # label action outcome by its standardized schema coordinate.
            binary_feature=set()
            for j,key in enumerate(frame_ids):
                samples=[
                    target.values[j] for target in truth
                    if target.observed_mask[j]
                ]
                if key=="action:success":
                    continue
                if samples and all(v in (0.0,1.0) for v in samples):
                    binary_feature.add(key)
            for row,(target,estimate,action) in enumerate(zip(truth,predicted,actions)):
                if estimate.coordinate_ids!=frame_ids:
                    raise ValueError("forecast target frame mismatch")
                for j,key in enumerate(frame_ids):
                    if not target.observed_mask[j]:
                        continue
                    available+=1
                    if not estimate.mask[j]:
                        missing+=1
                        abstentions+=1
                        continue
                    loss=(target.values[j]-estimate.values[j])**2
                    squared[key].append(loss)
                    by_action.setdefault(str(action),[]).append(loss)
                    if key=="action:success":
                        success_errors.append(loss)
                    elif key in binary_feature:
                        binary_errors.append(loss)
                        binary_count+=1
                        positive=estimate.values[j]>=0.5
                        actual=target.values[j]>=0.5
                        binary_true_positive+=int(positive and actual)
                        binary_false_positive+=int(positive and not actual)
                        binary_false_negative+=int(not positive and actual)
                    else:
                        numeric_count+=1
            def average(values):
                return sum(values)/len(values) if values else None
            numeric=[v for key,values in squared.items()
                     if key!="action:success" and key not in binary_feature
                     for v in values]
            total_numeric=sum(
                target.observed_mask[j]
                for target in truth for j,key in enumerate(frame_ids)
                if key!="action:success" and key not in binary_feature
            )
            # Empty/missing predictions cannot silently improve the result.
            numeric_complete=total_numeric==numeric_count
            numeric_mse=average(numeric) if numeric_complete else None
            per_action={key:average(values) for key,values in by_action.items()}
            per_feature={key:average(values) for key,values in squared.items()
                         if values}
            denom=2*binary_true_positive+binary_false_positive+binary_false_negative
            binary_f1=(2*binary_true_positive/denom if denom else None)
            output[name]=BackendScore(
                row_count=len(truth),scored_numeric_count=numeric_count,
                scored_binary_count=binary_count,
                numeric_normalized_mse=numeric_mse,
                binary_brier=average(binary_errors),
                binary_change_f1=binary_f1,
                action_success_brier=average(success_errors),
                per_action=per_action,per_feature=per_feature,
                coverage=(available-missing)/available if available else 0.,
                abstentions=abstentions,
                unseen_feature_count=sum(t.unseen_feature_count for t in truth),
                interventional_confirmation_count=confirmed,
            )
        return output
