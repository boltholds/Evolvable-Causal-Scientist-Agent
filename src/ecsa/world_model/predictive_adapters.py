"""Fit-on-training prediction adapters for the same public target frame."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from math import isfinite
from typing import Protocol

import numpy as np

from .contracts import GroundAction
from .perception.base import PerceptualObservation
from .shared_evaluation import (
    SharedOutcomeFrame, SharedForecast, PublicOutcomeTransition, TargetKind,
)


class LatentPredictionPort(Protocol):
    def predict_latent(
        self, before: PerceptualObservation, action: GroundAction
    ) -> tuple[float,...]: ...


def _public_input(
    observation:PerceptualObservation,action:GroundAction
)->tuple[float,...]:
    values=[]
    # Domain-agnostic summary: no UUID, entity name, or benchmark semantics.
    values.append(float(len(observation.entities)))
    values.append(float(len(observation.global_features)))
    for feature in observation.global_features:
        v=feature.value.thaw()
        if type(v) in (int,float,bool):
            values.append(float(v))
    for entity in observation.entities:
        for feature in entity.features:
            v=feature.value.thaw()
            if type(v) in (int,float,bool):
                values.append(float(v))
    values.extend((float(len(action.arguments)),1.))
    # Stable feature projection. Action symbols are opaque IDs supplied by
    # the public transport, not semantically interpreted by core ECSA.
    result=np.zeros(128,dtype=np.float64)
    for j,v in enumerate(values):
        if isfinite(v):
            result[j%64]+=np.clip(v,-1e4,1e4)
    action_key=action.schema_id.encode("utf-8")
    digest=blake2b(action_key,digest_size=8,person=b"ecsa-act").digest()
    slot=64 + int.from_bytes(digest[:4],"little")%64
    result[slot]+=1.0 if digest[4]&1 else -1.0
    return tuple(float(x) for x in result)


def _target_rows(
    frame:SharedOutcomeFrame,train:tuple[PublicOutcomeTransition,...]
):
    y=[]
    for entry in train:
        target=frame.project(entry)
        y.append([v if valid else 0. for v,valid in zip(
            target.values,target.observed_mask)])
    return np.array(y,dtype=np.float64)


class PersistencePort:
    def __init__(self,frame:SharedOutcomeFrame):
        self.frame=frame
        self.frame_fingerprint=frame.fingerprint

    def predict_public(
        self,observation:PerceptualObservation,action:GroundAction
    )->SharedForecast:
        values=[]
        for coordinate in self.frame.coordinates:
            if coordinate.kind is TargetKind.NUMERIC_DELTA:
                values.append(-coordinate.train_center/coordinate.train_scale)
            elif coordinate.kind is TargetKind.ACTION_SUCCESS:
                values.append(.5)
            else:
                values.append(0.)
        return SharedForecast(tuple(values),
                              tuple(True for _ in values),
                              tuple(c.key for c in self.frame.coordinates),
                              self.frame.fingerprint)


class _FittedLinearPort:
    def __init__(self,frame:SharedOutcomeFrame,weights:np.ndarray):
        self.frame=frame
        self.frame_fingerprint=frame.fingerprint
        self._weights=weights

    def _forecast(self,values:tuple[float,...])->SharedForecast:
        x=np.asarray(values,dtype=np.float64)
        if x.shape!=(self._weights.shape[0],):
            raise ValueError("unexpected predictor width")
        output=x@self._weights
        coords=self.frame.coordinates
        p=[]
        for value,coordinate in zip(output,coords):
            if coordinate.kind is TargetKind.ACTION_SUCCESS or coordinate.kind in (
                TargetKind.CATEGORICAL_CHANGE,TargetKind.BOOLEAN_CHANGE
            ):
                p.append(float(np.clip(value,0,1)))
            else:p.append(float(value))
        return SharedForecast(tuple(p),tuple(True for _ in p),
                              tuple(c.key for c in coords),self.frame.fingerprint)


def _fit_ridge(x:np.ndarray,y:np.ndarray,lam:float)->np.ndarray:
    if not isfinite(lam) or lam<=0:
        raise ValueError("positive finite regularizer required")
    regularization=lam*np.eye(x.shape[1],dtype=np.float64)
    regularization[-1,-1]=1e-8
    return np.linalg.solve(x.T@x+regularization,x.T@y)


class RawRidgePort(_FittedLinearPort):
    @classmethod
    def fit(cls,frame:SharedOutcomeFrame,
            train:tuple[PublicOutcomeTransition,...],*,
            ridge_lambda:float=1.0):
        if not train:raise ValueError("training data required")
        x=np.array([_public_input(i.before,i.transition.action)
                   for i in train],dtype=np.float64)
        return cls(frame,_fit_ridge(x,_target_rows(frame,train),ridge_lambda))

    def predict_public(
        self,observation:PerceptualObservation,action:GroundAction
    )->SharedForecast:
        return self._forecast(_public_input(observation,action))


class LatentReadoutPort(_FittedLinearPort):
    def __init__(self,frame:SharedOutcomeFrame,weights:np.ndarray,
                 latent_port:LatentPredictionPort):
        super().__init__(frame,weights)
        self._latent_port=latent_port

    @classmethod
    def fit(cls,frame:SharedOutcomeFrame,
            train:tuple[PublicOutcomeTransition,...],
            latent_port:LatentPredictionPort,*,
            ridge_lambda:float=1.0):
        if not train:raise ValueError("training data required")
        features=[tuple(latent_port.predict_latent(
            i.before,i.transition.action)) for i in train]
        if not features or len({len(v) for v in features})!=1:
            raise ValueError("consistent latent width required")
        x=np.array([(*features[i],1.) for i in range(len(features))],dtype=np.float64)
        if not np.isfinite(x).all():raise ValueError("finite latent predictions required")
        return cls(frame,_fit_ridge(x,_target_rows(frame,train),ridge_lambda),
                   latent_port)

    def predict_public(
        self,observation:PerceptualObservation,action:GroundAction
    )->SharedForecast:
        return self._forecast((*self._latent_port.predict_latent(observation,action),1.))
