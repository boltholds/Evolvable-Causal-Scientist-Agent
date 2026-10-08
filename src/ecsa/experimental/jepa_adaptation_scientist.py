"""Compare independent action-order intervention cohorts across unknown epochs.

Only public action IDs, observable outcome batches, and encoder projections.
No regime type or changepoint is used. Non-rejection cannot prove equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .jepa_world import Action
from .jepa_causal_scientist import CommutatorBatch, ObservationEncoder


@dataclass(frozen=True)
class OrderEffectContrast:
    pair: tuple[Action,Action]
    anchor_effect: float
    return_effect: float
    decline: float
    decline_p: float
    effect_decrease_confirmed: bool
    anchor_evidence_ids: tuple[str,...]
    return_evidence_ids: tuple[str,...]
    permutations: int


def _excess(encoder: ObservationEncoder,batch: CommutatorBatch) -> np.ndarray:
    forward=np.asarray(encoder.encode(batch.forward),dtype=np.float64)
    backward=np.asarray(encoder.encode(batch.backward),dtype=np.float64)
    repeated=np.asarray(encoder.encode(batch.repeated),dtype=np.float64)
    if (forward.ndim!=2 or backward.shape!=forward.shape
        or repeated.shape!=forward.shape or len(forward)!=len(batch.evidence_ids)
        or not np.all(np.isfinite(forward)) or not np.all(np.isfinite(backward))
        or not np.all(np.isfinite(repeated))):
        raise ValueError('invalid commutator representations')
    return np.square(forward-backward).sum(axis=1)-np.square(forward-repeated).sum(axis=1)


def compare_order_effect(
    encoder: ObservationEncoder, anchor: CommutatorBatch,
    returned: CommutatorBatch, *, pair: tuple[Action,Action],
    permutations: int = 4095, alpha: float = .01, seed: int = 0,
) -> OrderEffectContrast:
    """One-sided two-sample permutation test for *decline* in order effect.

    The action pair must be frozen from a prior independently confirmed
    hypothesis. A significant decline supports change, not full causal
    equivalence or proof of a prior regime's exact restoration.
    """
    if (not isinstance(pair,tuple) or len(pair)!=2
        or any(not isinstance(a,Action) for a in pair) or pair[0]==pair[1]):
        raise ValueError('distinct typed action pair required')
    if not isinstance(anchor,CommutatorBatch) or not isinstance(returned,CommutatorBatch):
        raise TypeError('paired commutator batches required')
    if type(permutations) is not int or permutations<99:
        raise ValueError('at least 99 permutation draws required')
    if not isfinite(alpha) or not 0<alpha<1:
        raise ValueError('alpha must be in (0,1)')
    if set(anchor.evidence_ids)&set(returned.evidence_ids):
        raise ValueError('anchor and return evidence must not overlap')
    left=_excess(encoder,anchor)
    right=_excess(encoder,returned)
    if len(left)<8 or len(right)<8:
        raise ValueError('each effect cohort needs >=8 independent trials')
    decrease=float(left.mean()-right.mean())
    combined=np.concatenate((left,right))
    rng=np.random.default_rng(seed)
    exceed=0
    for _ in range(permutations):
        ordering=rng.permutation(len(combined))
        sampled=float(combined[ordering[:len(left)]].mean()-combined[ordering[len(left):]].mean())
        exceed+=sampled>=decrease
    p=(1+exceed)/(permutations+1)
    return OrderEffectContrast(
        pair=pair,anchor_effect=float(left.mean()),return_effect=float(right.mean()),
        decline=decrease,decline_p=float(p),
        effect_decrease_confirmed=bool(decrease>0 and p<alpha),
        anchor_evidence_ids=tuple(anchor.evidence_ids),
        return_evidence_ids=tuple(returned.evidence_ids),permutations=permutations,
    )
