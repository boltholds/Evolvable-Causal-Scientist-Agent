"""Intervention-based discovery of action noncommutativity in latent space.

No import of the simulator mechanism or hidden state: only public action IDs,
resettable observations, and an encoder/predictor port are available here.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from .action_jepa import ActionJEPA
from .jepa_world import Action, _ACTIONS, action_vector

@runtime_checkable
class ObservationEncoder(Protocol):
    def encode(self, observations: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class ResettableInterventionPort(Protocol):
    def reset(self, *, trial_seed: int) -> np.ndarray: ...
    def step(self, action: Action) -> np.ndarray: ...


@dataclass(frozen=True)
class CommutatorEvidence:
    first_action: Action
    second_action: Action
    effect_distance: float
    null_distance: float
    representation_spread: float
    prediction: bool | None
    evidence_ids: tuple[str, ...]


def _run_sequence(
    world: ResettableInterventionPort, trial_id: int,
    actions: tuple[Action, Action],
) -> tuple[np.ndarray, np.ndarray]:
    before = world.reset(trial_seed=trial_id)
    for action in actions:
        after = world.step(action)
    return before, after


@dataclass(frozen=True)
class CommutatorBatch:
    """The same measured intervention outcomes can be scored by every port."""
    initial: np.ndarray
    forward: np.ndarray
    backward: np.ndarray
    repeated: np.ndarray
    evidence_ids: tuple[str, ...]


def collect_commutator_observations(
    world: ResettableInterventionPort,
    first_action: Action,
    second_action: Action,
    *, trials: int = 24, seed: int = 0,
) -> CommutatorBatch:
    """Acquire paired, independently noisy observations from matched resets."""
    if not isinstance(first_action, Action) or not isinstance(second_action, Action):
        raise TypeError("typed intervention actions required")
    if first_action is second_action:
        raise ValueError("intervention actions must be distinct")
    if type(trials) is not int or trials < 4:
        raise ValueError("trials must be >= 4")
    initial, forward, backward, repeated = [], [], [], []
    for trial in range(trials):
        trial_id = 2_000_033 + seed * 1009 + trial
        start, ab = _run_sequence(world, trial_id, (first_action,second_action))
        _, ba = _run_sequence(world, trial_id, (second_action,first_action))
        _, ab_again = _run_sequence(world, trial_id, (first_action,second_action))
        initial.append(start)
        forward.append(ab)
        backward.append(ba)
        repeated.append(ab_again)
    return CommutatorBatch(
        initial=np.asarray(initial), forward=np.asarray(forward),
        backward=np.asarray(backward), repeated=np.asarray(repeated),
        evidence_ids=tuple(f"commutator:{seed}:{trial}" for trial in range(trials)),
    )


def score_commutator(
    encoder: ObservationEncoder, batch: CommutatorBatch,
    first_action: Action, second_action: Action,
) -> CommutatorEvidence:
    """Score an acquired dataset without access to the simulator or law."""
    initial = np.asarray(encoder.encode(batch.initial))
    forward = np.asarray(encoder.encode(batch.forward))
    backward = np.asarray(encoder.encode(batch.backward))
    repeated = np.asarray(encoder.encode(batch.repeated))
    if (initial.ndim != 2 or initial.shape[0] != len(batch.evidence_ids)
            or forward.shape != backward.shape or forward.shape != repeated.shape
            or forward.shape[0] != len(batch.evidence_ids)
            or not all(np.isfinite(v).all() for v in (
            initial, forward, backward, repeated))):
        raise ValueError("encoder returned invalid latent observations")
    gap = float(np.linalg.norm(forward - backward, axis=1).mean())
    null_gap = float(np.linalg.norm(forward - repeated, axis=1).mean())
    spread = float(initial.std(axis=0).mean())
    prediction: bool | None = None
    if spread > 1e-5:
        prediction = (gap > 2.5 * null_gap and gap - null_gap > .05 * spread)
    return CommutatorEvidence(
        first_action=first_action, second_action=second_action,
        effect_distance=gap, null_distance=null_gap,
        representation_spread=spread, prediction=prediction,
        evidence_ids=batch.evidence_ids,
    )


def examine_commutator(
    encoder: ObservationEncoder,
    world: ResettableInterventionPort,
    first_action: Action,
    second_action: Action,
    *, trials: int = 24, seed: int = 0,
) -> CommutatorEvidence:
    """Test two intervention orders using observations, never simulator truth."""
    batch = collect_commutator_observations(
        world, first_action, second_action, trials=trials, seed=seed,
    )
    return score_commutator(encoder, batch, first_action, second_action)


def select_intervention_pair(
    model: ActionJEPA, observation: np.ndarray,
) -> tuple[tuple[Action, Action], dict[str, float]]:
    """Pick which action pair to experimentally challenge using model rollouts.

    Deliberately never calls or accesses any simulator or hidden mechanism.
    Ranking is a proposal only; experiment evidence is evaluated separately.
    """
    device = next(model.parameters()).device
    with torch.inference_mode():
        z = model.encoder(torch.as_tensor(observation[None, :], device=device))
        def roll(first: Action, second: Action) -> Tensor:
            a = torch.as_tensor(action_vector(first)[None, :], device=device)
            b = torch.as_tensor(action_vector(second)[None, :], device=device)
            return model.predict_tensor(model.predict_tensor(z,a),b)
        estimates: dict[str,float] = {}
        indexed: list[tuple[float, tuple[Action, Action]]] = []
        for first,second in combinations(_ACTIONS,2):
            label = ":".join(sorted((first.value,second.value)))
            value = float(torch.linalg.vector_norm(roll(first,second)-roll(second,first)))
            estimates[label] = value
            indexed.append((value,(first,second)))
    selection = max(indexed, key=lambda item: (item[0], item[1][0].value, item[1][1].value))
    return selection[1], estimates


def _scientist_dict(evidence: CommutatorEvidence) -> dict[str, object]:
    return {
        "first_action": evidence.first_action.value,
        "second_action": evidence.second_action.value,
        "effect_distance": evidence.effect_distance,
        "null_distance": evidence.null_distance,
        "representation_spread": evidence.representation_spread,
        "predicted_noncommutative": evidence.prediction,
        "evidence_ids": list(evidence.evidence_ids),
    }


