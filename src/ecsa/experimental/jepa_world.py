"""Opaque synthetic switch world, observable actions and transition samples.

Only this evaluator-side module knows the simulator mechanism. Training data
contains observation/action/next observation, never a ground-truth state.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

import numpy as np

class Action(StrEnum):
    PULSE = "pulse"
    FLIP = "flip"
    WAIT = "wait"


class Mechanism(StrEnum):
    """Simulator-only ground truth. Never an encoder/scientist input."""
    ADDITIVE = "additive"
    GATED = "gated"


_ACTIONS = tuple(Action)


def action_vector(action: Action) -> np.ndarray:
    if not isinstance(action, Action):
        raise TypeError("action must be a typed Action")
    values = np.zeros(len(_ACTIONS), dtype=np.float32)
    values[_ACTIONS.index(action)] = 1
    return values


class OpaqueSwitchWorld:
    """Two intervention mechanisms sharing the same opaque sensor surface.

    The caller can reset by a public trial ID without learning the sampled
    underlying state. The ``level/gate`` override is for fixture tests only.
    ``seed`` changes observation noise, ``render_seed`` fixes sensor geometry.
    """

    def __init__(
        self, *, mechanism: Mechanism, observation_dim: int = 24,
        seed: int = 0, noise_std: float = .012, render_seed: int = 2026,
    ) -> None:
        if not isinstance(mechanism, Mechanism):
            raise TypeError("mechanism must be a typed simulator Mechanism")
        if type(observation_dim) is not int or observation_dim < 8:
            raise ValueError("observation_dim must be >= 8")
        if not isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        self._mechanism = mechanism
        self.observation_dim = observation_dim
        self._noise_std = noise_std
        self._sensor_rng = np.random.default_rng(seed)
        rng = np.random.default_rng(render_seed)
        self._mixing = rng.normal(0, 1.0, (4, observation_dim)).astype(np.float32)
        self._level = 0.0
        self._gate = False

    def reset(
        self, *, trial_seed: int | None = None, level: float | None = None,
        gate: bool | None = None,
    ) -> np.ndarray:
        if trial_seed is not None:
            if level is not None or gate is not None:
                raise ValueError("trial_seed and privileged overrides are exclusive")
            rng = np.random.default_rng(trial_seed)
            self._level = float(rng.uniform(-.6, .6))
            self._gate = bool(rng.integers(0, 2))
        elif level is not None and gate is not None:
            if not isfinite(level) or type(gate) is not bool:
                raise ValueError("invalid explicit fixture state")
            self._level = float(level)
            self._gate = gate
        else:
            raise ValueError("provide trial_seed or fixture level and gate")
        return self._observe()

    def step(self, action: Action) -> np.ndarray:
        if not isinstance(action, Action):
            raise TypeError("step requires typed Action")
        if action is Action.PULSE:
            if self._mechanism is Mechanism.ADDITIVE or self._gate:
                self._level += .36
        elif action is Action.FLIP:
            self._gate = not self._gate
        return self._observe()

    def _observe(self) -> np.ndarray:
        signed_gate = 1.0 if self._gate else -1.0
        # No coordinate represents a labelled variable; nonlinear sensor map
        # and independent observation nuisance obscure the latent mechanism.
        hidden = np.array([
            self._level, signed_gate, self._level * signed_gate,
            np.sin(1.7 * self._level),
        ], dtype=np.float32)
        surface = np.tanh(hidden @ self._mixing * .72)
        noise = self._sensor_rng.normal(0, self._noise_std, self.observation_dim)
        return (surface + noise).astype(np.float32)


@dataclass(frozen=True)
class TransitionDataset:
    before: np.ndarray
    actions: np.ndarray
    after: np.ndarray

    def __post_init__(self) -> None:
        if self.before.ndim != 2 or self.after.shape != self.before.shape:
            raise ValueError("observations must have equal [N, D] dimensions")
        if self.actions.shape != (len(self.before), len(_ACTIONS)):
            raise ValueError("actions must be typed one-hot vectors")
        if not all(np.isfinite(x).all() for x in (self.before,self.actions,self.after)):
            raise ValueError("transition arrays must be finite")
        if not np.all((self.actions == 0) | (self.actions == 1)) or not np.all(
            self.actions.sum(axis=1) == 1
        ):
            raise ValueError("actions must be one-hot")


def collect_transitions(
    *, mechanism: Mechanism, count: int, observation_dim: int = 24,
    seed: int = 0, noise_std: float = .012,
) -> TransitionDataset:
    """Generate (observation, public action, next observation) only."""
    if type(count) is not int or count < 1:
        raise ValueError("count must be positive")
    world = OpaqueSwitchWorld(
        mechanism=mechanism, observation_dim=observation_dim,
        noise_std=noise_std, seed=seed,
    )
    rng = np.random.default_rng(seed + 100_019)
    before, actions, after = [], [], []
    for i in range(count):
        observation = world.reset(trial_seed=seed * 1_000_003 + i)
        action = _ACTIONS[int(rng.integers(len(_ACTIONS)))]
        before.append(observation)
        actions.append(action_vector(action))
        after.append(world.step(action))
    return TransitionDataset(
        before=np.asarray(before, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        after=np.asarray(after, dtype=np.float32),
    )


