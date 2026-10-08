"""Evaluator-only, unannounced mechanism switching over a fixed sensor map.

A scientist sees only reset / step and opaque observations. `switch_after`
never travels in a transition or through an action interface to a learner.
"""
from __future__ import annotations

from .jepa_world import (
    Action, Mechanism, OpaqueSwitchWorld, TransitionDataset,
    _ACTIONS, action_vector,
)

import numpy as np


class HiddenShiftWorld(OpaqueSwitchWorld):
    """Change additive -> gated after N actions, preserving physical state.

    `switch_after=None` is the stationary no-change control. Resets never
    reset the regime or its clock. Only the evaluator should construct this.
    """

    def __init__(
        self, *, switch_after: int | None, seed: int = 0,
        noise_std: float = .16, observation_dim: int = 24,
        render_seed: int = 2026,
    ) -> None:
        if switch_after is not None and (type(switch_after) is not int or switch_after < 0):
            raise ValueError("switch_after must be nonnegative or None")
        super().__init__(
            mechanism=Mechanism.ADDITIVE, observation_dim=observation_dim,
            seed=seed, noise_std=noise_std, render_seed=render_seed,
        )
        self._switch_after = switch_after
        self._actions_seen = 0

    def step(self, action: Action) -> np.ndarray:
        if self._switch_after is not None and self._actions_seen == self._switch_after:
            self._mechanism = Mechanism.GATED
        result = super().step(action)
        self._actions_seen += 1
        return result


def observe_stream(
    world: HiddenShiftWorld, *, count: int, seed: int,
) -> TransitionDataset:
    """Produce pre-/post-action observations without regime or step labels."""
    if not isinstance(world, HiddenShiftWorld):
        raise TypeError("observation stream requires a hidden-shift environment")
    if type(count) is not int or count <= 0:
        raise ValueError("stream count must be positive")
    rng = np.random.default_rng(seed * 43 + 997)
    schedule = tuple(_ACTIONS[int(i)] for i in rng.permutation(np.tile(np.arange(3), (count+2)//3))[:count])
    before, actions, after = [], [], []
    for i, action in enumerate(schedule):
        start = world.reset(trial_seed=seed * 1_000_019 + i)
        before.append(start)
        actions.append(action_vector(action))
        after.append(world.step(action))
    return TransitionDataset(
        before=np.asarray(before, np.float32),
        actions=np.asarray(actions, np.float32),
        after=np.asarray(after, np.float32),
    )
