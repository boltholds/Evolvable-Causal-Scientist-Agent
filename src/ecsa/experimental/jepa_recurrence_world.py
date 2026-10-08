"""Evaluator-only recurrent synthetic physics; no regime metadata in observations.

One opaque sensor geometry and physical state across unannounced transitions:
additive -> gated -> additive. Only the evaluator constructs this fixture.
"""
from __future__ import annotations

import numpy as np

from .jepa_world import (
    Action, Mechanism, OpaqueSwitchWorld, TransitionDataset,
    _ACTIONS, action_vector,
)


class RecurrenceWorld(OpaqueSwitchWorld):
    """Change rules twice at action indices. Public reset does not reset epoch."""

    def __init__(
        self, *, switch_at: tuple[int, int], seed: int = 0,
        noise_std: float = .16, observation_dim: int = 24,
        render_seed: int = 2026,
    ) -> None:
        if (not isinstance(switch_at, tuple) or len(switch_at) != 2
            or any(type(x) is not int for x in switch_at)
            or not 0 < switch_at[0] < switch_at[1]):
            raise ValueError("strictly increasing positive switch boundaries required")
        super().__init__(
            mechanism=Mechanism.ADDITIVE, seed=seed,
            noise_std=noise_std, observation_dim=observation_dim,
            render_seed=render_seed,
        )
        self._switch_at = switch_at
        self._actions_seen = 0

    def step(self, action: Action) -> np.ndarray:
        if self._actions_seen == self._switch_at[0]:
            self._mechanism = Mechanism.GATED
        elif self._actions_seen == self._switch_at[1]:
            self._mechanism = Mechanism.ADDITIVE
        result = super().step(action)
        self._actions_seen += 1
        return result


def observe_recurrence_stream(
    world: RecurrenceWorld, *, count: int, seed: int,
) -> TransitionDataset:
    """Only public observations/actions. No mechanism, changepoint or state ID."""
    if not isinstance(world, RecurrenceWorld):
        raise TypeError("requires recurrence world")
    if type(count) is not int or count < 1:
        raise ValueError("positive observation count required")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative seed required")
    rng = np.random.default_rng(seed * 43 + 997)
    schedule = tuple(_ACTIONS[int(i)] for i in rng.permutation(
        np.tile(np.arange(len(_ACTIONS)), (count + len(_ACTIONS) - 1) // len(_ACTIONS))
    )[:count])
    before, action_ids, after = [], [], []
    for index, action in enumerate(schedule):
        before.append(world.reset(trial_seed=seed * 1_000_019 + index))
        action_ids.append(action_vector(action))
        after.append(world.step(action))
    return TransitionDataset(
        np.asarray(before, dtype=np.float32),
        np.asarray(action_ids, dtype=np.float32),
        np.asarray(after, dtype=np.float32),
    )


def segment(data: TransitionDataset, start: int, stop: int) -> TransitionDataset:
    """Evaluator-only slicing, never a learner-side mechanism decoder."""
    if not isinstance(data, TransitionDataset):
        raise TypeError("transition data required")
    if (type(start) is not int or type(stop) is not int
        or not 0 <= start < stop <= len(data.before)):
        raise ValueError("invalid transition slice")
    return TransitionDataset(data.before[start:stop], data.actions[start:stop], data.after[start:stop])
