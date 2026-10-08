"""Model-agnostic scientist for active, heldout action-order interventions.

The experiment selector sees only public action IDs, opaque observations and
an encoder port. It does not inspect the simulator or learn law labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Protocol, runtime_checkable

import numpy as np

from .jepa_world import Action
from .jepa_causal_scientist import (
    CommutatorBatch, ObservationEncoder, ResettableInterventionPort,
    collect_commutator_observations,
)


@dataclass(frozen=True)
class OrderEffectEvidence:
    pair: tuple[Action, Action]
    p_value: float
    positive_effect: float  # excess forward/backward squared distance
    null_distance: float    # mean forward/repeat squared distance
    supports_order_effect: bool
    evidence_ids: tuple[str, ...]
    permutations: int
    alpha: float


def score_order_effect(
    encoder: ObservationEncoder, batch: CommutatorBatch,
    pair: tuple[Action, Action], *, permutations: int = 2047,
    alpha: float = .01, seed: int = 0,
) -> OrderEffectEvidence:
    """One-sided paired sign-flip test with independent repeated-order null.

    At a commutative law, forward/backward and forward/repeat order are
    exchangeable conditional on their common forward measurement. Each trial
    may flip its assignment; **not** independent unpaired reshuffling.
    """
    if len(pair)!=2 or any(not isinstance(action,Action) for action in pair) or pair[0] == pair[1]:
        raise ValueError("distinct typed action pair required")
    if type(permutations) is not int or permutations < 99:
        raise ValueError("at least 99 permutation draws required")
    if not isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    encoded = tuple(np.asarray(encoder.encode(values),dtype=np.float64)
                    for values in (batch.initial,batch.forward,batch.backward,batch.repeated))
    initial,forward,backward,repeated = encoded
    count=len(batch.evidence_ids)
    if any(v.ndim != 2 or len(v)!=count or v.shape != forward.shape or not np.isfinite(v).all()
           for v in encoded):
        raise ValueError("incompatible or nonfinite representation batches")
    gap_sq=np.sum((forward-backward)**2,axis=1)
    null_sq=np.sum((forward-repeated)**2,axis=1)
    difference=gap_sq-null_sq
    observed=float(difference.mean())
    rng=np.random.default_rng(seed)
    signs=rng.choice(np.asarray((-1.,1.)),size=(permutations,count))
    null_scores=(signs*difference[None,:]).mean(axis=1)
    p_value=float((1+int((null_scores>=observed).sum()))/(permutations+1))
    return OrderEffectEvidence(
        pair=pair,p_value=p_value,positive_effect=observed,
        null_distance=float(null_sq.mean()),
        supports_order_effect=(observed>0 and p_value<alpha),
        evidence_ids=batch.evidence_ids,permutations=permutations,alpha=alpha,
    )


def confirm_order_effect(
    encoder: ObservationEncoder, world: ResettableInterventionPort,
    pair: tuple[Action,Action], *, trials: int = 64, seed: int = 0,
    permutations: int = 2047, alpha: float = .01,
) -> OrderEffectEvidence:
    batch=collect_commutator_observations(world,*pair,trials=trials,seed=seed)
    return score_order_effect(encoder,batch,pair,permutations=permutations,
                              alpha=alpha,seed=seed+991)


@dataclass(frozen=True)
class HypothesisDiscovery:
    hypotheses: tuple[str,str]
    pilot_scores: tuple[tuple[str,float],...]
    pilot_evidence_ids: tuple[str,...]
    selected_pair: tuple[Action,Action]
    confirmation: OrderEffectEvidence


def discover_order_hypothesis(
    encoder: ObservationEncoder, world: ResettableInterventionPort,
    *, actions: tuple[Action,...] = tuple(Action), pilot_trials: int = 12,
    confirmation_trials: int = 64, permutations: int = 2047,
    alpha: float = .01, seed: int = 0,
) -> HypothesisDiscovery:
    """Select pair with strongest pilot excess; confirm on distinct trials.

    Multiple pilot comparisons are used for *selection only*. Only one pair
    is judged against alpha using a fresh confirmatory experiment.
    """
    if not isinstance(actions,tuple) or len(actions)<2 or len(set(actions))!=len(actions) or any(
        not isinstance(action,Action) for action in actions
    ):
        raise ValueError("distinct observable action IDs required")
    indexed = []
    ids: list[str] = []
    scores: list[tuple[str,float]] = []
    for index,pair in enumerate(combinations(actions,2)):
        # Even for disjoint action pairs, independent pilot cohorts are used.
        candidate_seed=seed * 1_000 + 400 + index * 37
        batch=collect_commutator_observations(world,*pair,trials=pilot_trials,
                                              seed=candidate_seed)
        pilot=score_order_effect(encoder,batch,pair,permutations=permutations,
                                 alpha=alpha,seed=candidate_seed+317)
        signal=pilot.positive_effect
        label=":".join(action.value for action in pair)
        scores.append((label,signal))
        ids.extend(batch.evidence_ids)
        indexed.append((signal,pair))
    selection=max(indexed,key=lambda row:(row[0],row[1][0].value,row[1][1].value))[1]
    confirm=confirm_order_effect(encoder,world,selection,trials=confirmation_trials,
                                 seed=seed*1_000+999_931,permutations=permutations,
                                 alpha=alpha)
    if set(ids)&set(confirm.evidence_ids):
        raise AssertionError("pilot/confirmation evidence may not overlap")
    return HypothesisDiscovery(
        hypotheses=("order_independence","order_dependence"),
        pilot_scores=tuple(scores),pilot_evidence_ids=tuple(ids),
        selected_pair=selection,confirmation=confirm,
    )


def discover_from_pilot_batches(
    encoder: ObservationEncoder,
    pilots: dict[tuple[Action,Action],CommutatorBatch],
    confirmations: dict[tuple[Action,Action],CommutatorBatch],
    *, permutations: int = 2047, alpha: float = .01, seed: int = 0,
) -> HypothesisDiscovery:
    """Active selection over matched common pilot banks; independent confirm.

    The evaluator acquires all batches beforehand. Only pilot outcomes affect
    the chosen pair. The same acquired observations can be scored fairly by
    multiple candidate representations without simulator leakage.
    """
    if len(pilots)<1 or set(pilots)!=set(confirmations):
        raise ValueError("paired pilot and confirmation banks are required")
    scored=[]
    scores=[]
    ids=[]
    for index,pair in enumerate(sorted(pilots,key=lambda pair:(pair[0].value,pair[1].value))):
        pilot=score_order_effect(encoder,pilots[pair],pair,permutations=permutations,
                                 alpha=alpha,seed=seed+100+index)
        scored.append((pilot.positive_effect,pair))
        scores.append((":".join(a.value for a in pair), pilot.positive_effect))
        ids.extend(pilot.evidence_ids)
    pair=max(scored,key=lambda row:(row[0],row[1][0].value,row[1][1].value))[1]
    confirmation=score_order_effect(encoder,confirmations[pair],pair,
                                    permutations=permutations,alpha=alpha,
                                    seed=seed+99_999)
    if set(ids)&set(confirmation.evidence_ids):
        raise ValueError("confirmation must not reuse pilot observations")
    if len(set(ids)) != len(ids):
        raise ValueError("pilot evidence identifiers must be disjoint")
    return HypothesisDiscovery(("order_independence","order_dependence"),
                                tuple(scores),tuple(ids),pair,confirmation)
