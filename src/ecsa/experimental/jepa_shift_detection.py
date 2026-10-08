"""Blinded transition-surprise monitors with prechange-only calibration.

Receives observed states and public actions. No simulator-law or hidden-time
imports. Calibration uses action-stratified null-bootstrap *maxima across the
whole horizon*, so repeated window inspection is explicitly accounted for.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from .action_jepa import ActionJEPA
from .jepa_world import TransitionDataset, _ACTIONS


@runtime_checkable
class ObservationEncoder(Protocol):
    def encode(self, observations: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class TransitionResiduals(Protocol):
    def errors(self, data: TransitionDataset) -> np.ndarray: ...


def _validate_one_hot(actions: np.ndarray, length: int) -> np.ndarray:
    matrix = np.asarray(actions, dtype=np.float64)
    if matrix.shape != (length, len(_ACTIONS)) or not np.all(np.isfinite(matrix)):
        raise ValueError("actions must be finite [N,3] one-hot rows")
    if not np.all((matrix == 0) | (matrix == 1)) or not np.all(matrix.sum(axis=1) == 1):
        raise ValueError("invalid action one-hot values")
    return matrix.argmax(axis=1)


class FrozenJEPAResiduals:
    """Per-transition latent squared prediction errors; model never updates."""

    def __init__(self, model: ActionJEPA) -> None:
        if not isinstance(model, ActionJEPA):
            raise TypeError("expected trained ActionJEPA")
        self._model = model.eval()

    def errors(self, data: TransitionDataset) -> np.ndarray:
        if not isinstance(data, TransitionDataset):
            raise TypeError("transition dataset required")
        device = next(self._model.parameters()).device
        with torch.inference_mode():
            before = torch.as_tensor(data.before, device=device)
            action = torch.as_tensor(data.actions, device=device)
            after = torch.as_tensor(data.after, device=device)
            z_before = self._model.encoder(before)
            prediction = self._model.predict_tensor(z_before, action)
            target = self._model.encoder(after)
            err = (prediction - target).square().mean(dim=1)
        result = err.double().cpu().numpy()
        if not np.all(np.isfinite(result)):
            raise ValueError("JEPA produced a nonfinite residual")
        return result


class RidgeDynamics:
    """Matched analytic observation-only linear dynamics baseline.

    Action interaction terms are included (not deliberately weakened). Both
    the feature projection and regression are fitted on additive training only.
    """

    def __init__(
        self, encoder: ObservationEncoder, weights: np.ndarray,
        input_dim: int, output_dim: int,
    ) -> None:
        self._encoder = encoder
        self._weights = weights
        self._input_dim = input_dim
        self._output_dim = output_dim

    @staticmethod
    def _features(encoded: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.concatenate((
            encoded,
            (encoded[:, :, None] * actions[:, None, :]).reshape(len(encoded), -1),
            actions,
            np.ones((len(encoded), 1), dtype=np.float64),
        ), axis=1)

    @classmethod
    def fit(
        cls, train: TransitionDataset, encoder: ObservationEncoder,
        *, regularization: float = 1e-2,
    ) -> RidgeDynamics:
        if not isinstance(encoder, ObservationEncoder):
            raise TypeError("observation encoder port required")
        if not isfinite(regularization) or regularization <= 0:
            raise ValueError("ridge regularization must be positive and finite")
        source = np.asarray(encoder.encode(train.before), dtype=np.float64)
        target = np.asarray(encoder.encode(train.after), dtype=np.float64)
        if source.ndim != 2 or target.shape != source.shape or not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("encoder returned invalid observation matrix")
        one_hot = np.asarray(train.actions, dtype=np.float64)
        design = cls._features(source, one_hot)
        gram = design.T @ design
        penalties = np.eye(design.shape[1]) * regularization
        penalties[-1, -1] = 0.0
        weights = np.linalg.solve(gram + penalties, design.T @ target)
        return cls(encoder, weights, train.before.shape[1], source.shape[1])

    def errors(self, data: TransitionDataset) -> np.ndarray:
        if data.before.shape[1] != self._input_dim:
            raise ValueError("observation dimension changed")
        source = np.asarray(self._encoder.encode(data.before), dtype=np.float64)
        target = np.asarray(self._encoder.encode(data.after), dtype=np.float64)
        if source.shape != target.shape or source.shape != (len(data.before),self._output_dim):
            raise ValueError("encoded dimension changed")
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("nonfinite encoded observations")
        prediction = self._features(source, np.asarray(data.actions,dtype=np.float64)) @ self._weights
        residuals = np.square(prediction-target).mean(axis=1)
        if not np.isfinite(residuals).all():
            raise ValueError("nonfinite ridge residual")
        return residuals


@dataclass(frozen=True)
class ChangeAlarm:
    alarm_index: int | None  # zero-based observation index; not mechanism time
    threshold: float
    peak_score: float  # stopping-time score, or maximum if no alarm
    global_null_p: float  # max-statistic bootstrap tail probability
    window: int


class CalibratedShiftMonitor:
    """Fixed action-stratified baseline and whole-horizon max-statistic test.

    Not a distribution-free anytime guarantee: the null approximation assumes
    calibrated transition residuals are exchangeable conditional on actions.
    """

    def __init__(
        self, calibration_errors: np.ndarray, calibration_actions: np.ndarray,
        planned_actions: np.ndarray, *, horizon: int, window: int = 12,
        bootstrap: int = 512, alpha: float = .01, seed: int = 0,
    ) -> None:
        values = np.asarray(calibration_errors, dtype=np.float64)
        if values.ndim != 1 or len(values) < 30 or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("calibration must contain >=30 nonnegative finite residuals")
        if type(horizon) is not int or horizon < 16 or type(window) is not int or not 4 <= window <= horizon:
            raise ValueError("invalid horizon/window")
        if type(bootstrap) is not int or bootstrap < 100:
            raise ValueError("bootstrap must have >=100 null trials")
        if not isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        cal_indices = _validate_one_hot(calibration_actions,len(values))
        planned_indices = _validate_one_hot(planned_actions,horizon)
        self.horizon = horizon
        self.window = window
        self._alpha = alpha
        self._baseline_mean = np.zeros(len(_ACTIONS), dtype=np.float64)
        self._baseline_std = np.zeros(len(_ACTIONS), dtype=np.float64)
        pool = []
        for action in range(len(_ACTIONS)):
            sample = values[cal_indices == action]
            if len(sample) < 8:
                raise ValueError("each action needs >=8 independent calibration residuals")
            self._baseline_mean[action] = sample.mean()
            self._baseline_std[action] = max(float(sample.std(ddof=1)), 1e-12)
            pool.append((sample - self._baseline_mean[action]) / self._baseline_std[action])

        rng = np.random.default_rng(seed)
        nulls = np.empty((bootstrap,horizon),dtype=np.float64)
        for action in range(len(_ACTIONS)):
            mask = planned_indices == action
            nulls[:,mask] = rng.choice(pool[action],size=(bootstrap,int(mask.sum())),replace=True)
        cumulative = np.concatenate((np.zeros((bootstrap,1)),np.cumsum(nulls,axis=1)),axis=1)
        rolling = (cumulative[:,window:] - cumulative[:,:-window]) / window
        self._null_max = rolling.max(axis=1)
        self.threshold = float(np.quantile(self._null_max, 1-alpha))

    def detect(self, errors: np.ndarray, actions: np.ndarray) -> ChangeAlarm:
        values=np.asarray(errors,dtype=np.float64)
        if values.ndim != 1 or len(values)!=self.horizon:
            raise ValueError("residual horizon mismatch")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("monitor residuals must be nonnegative and finite")
        ids = _validate_one_hot(actions,self.horizon)
        normalized=(values-self._baseline_mean[ids])/self._baseline_std[ids]
        scores=np.convolve(normalized,np.ones(self.window)/self.window,mode="valid")
        above=np.flatnonzero(scores > self.threshold)
        alarm=(int(above[0])+self.window-1) if len(above) else None
        # Once an alert fires, freeze all reported evidence at its stopping
        # time: future residuals cannot strengthen a past scientific claim.
        peak=max(float(scores[int(above[0])]) if len(above) else float(scores.max()),0.0)
        global_p=float((1+int((self._null_max>=peak).sum()))/(len(self._null_max)+1))
        return ChangeAlarm(alarm, self.threshold, peak, global_p, self.window)

class ObservedActionDeltaTemplate:
    """Rule-based finite-difference baseline on opaque observed coordinates.

    Computes the mean observed delta for each public action; no neural model,
    hidden physical state, regime label, or simulator access. A simple symbolic
    *template*, deliberately reported alongside stronger ridge baselines.
    """

    def __init__(self, expected_deltas: np.ndarray) -> None:
        self._deltas = expected_deltas

    @classmethod
    def fit(cls, train: TransitionDataset) -> ObservedActionDeltaTemplate:
        ids = _validate_one_hot(train.actions, len(train.before))
        delta = np.asarray(train.after - train.before,dtype=np.float64)
        expected = []
        for action in range(len(_ACTIONS)):
            chosen=delta[ids==action]
            if len(chosen)<8:
                raise ValueError("symbolic action templates need >=8 transitions per action")
            expected.append(chosen.mean(axis=0))
        return cls(np.asarray(expected,dtype=np.float64))

    def errors(self, data: TransitionDataset) -> np.ndarray:
        if data.before.shape[1] != self._deltas.shape[1]:
            raise ValueError("observation dimension changed")
        ids=_validate_one_hot(data.actions,len(data.before))
        delta=np.asarray(data.after-data.before,dtype=np.float64)
        err=np.square(delta-self._deltas[ids]).mean(axis=1)
        if not np.isfinite(err).all():
            raise ValueError("nonfinite symbolic template residual")
        return err
