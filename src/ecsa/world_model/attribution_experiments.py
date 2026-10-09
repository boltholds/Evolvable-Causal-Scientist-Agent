"""Rank public interventions to distinguish competing effect explanations."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from ecsa.contracts import PredictiveDistribution, TheoryPosterior, TheoryRef
from ecsa.science import ScienceKernel

from .contracts import GroundAction
from .effect_attribution import (
    EffectAttributionLedger, EffectClaim, EffectExplanation,
)


@dataclass(frozen=True)
class AttributionExperiment:
    action: GroundAction
    information_gain_bits: float
    hypothesis_ids: tuple[str, ...]
    pre_registered_predictions: tuple[PredictiveDistribution, ...]


class AttributionExperimentSelector:
    def __init__(self, ledger: EffectAttributionLedger,
                 science: ScienceKernel | None = None):
        if not isinstance(ledger, EffectAttributionLedger):
            raise TypeError("typed attribution ledger required")
        self.ledger = ledger
        self.science = science or ScienceKernel()

    def _probability(
        self, claim: EffectClaim, action: GroundAction,
        context: tuple[str, ...],
    ) -> float:
        # Learn action-conditional rates from *prior* observations. Unseen
        # actions fall back to the pooled rate instead of fabricated effects.
        rows=[row for row,_ in self.ledger._items
              if row.effect_key==claim.effect_key and
              row.context_signature==context and row.observed
              and row.changed is not None]
        pooled=(1+sum(bool(row.changed) for row in rows))/(len(rows)+2) if rows else 0.5
        if claim.explanation in (
            EffectExplanation.BACKGROUND, EffectExplanation.STATE_DEPENDENT,
            EffectExplanation.UNRESOLVED,
        ):
            return pooled
        if action.schema_id != claim.action_schema_id:
            return pooled
        matching=[r for r in rows if r.action_schema_id==action.schema_id]
        return (1+sum(bool(row.changed) for row in matching))/(len(matching)+2) if matching else pooled

    def select(
        self,
        claims: tuple[EffectClaim, ...],
        candidate_actions: tuple[GroundAction, ...],
        *,
        context_signature: tuple[str, ...],
    ) -> AttributionExperiment:
        if not isinstance(claims,tuple) or len(claims)<2 or not all(
            isinstance(c,EffectClaim) for c in claims
        ):
            raise ValueError("at least two typed competing claims required")
        if not isinstance(candidate_actions,tuple) or not candidate_actions or not all(
            isinstance(a,GroundAction) for a in candidate_actions
        ):
            raise ValueError("public candidate actions required")
        if len(set(c.hypothesis_id for c in claims)) != len(claims):
            raise ValueError("distinct scientific hypotheses required")
        posterior=TheoryPosterior.uniform(tuple(
            TheoryRef(c.hypothesis_id,c.hypothesis_id) for c in claims))
        available=[]
        for action in candidate_actions:
            experiment_id="effect-intervention:"+sha256(repr(action).encode()).hexdigest()[:20]
            predictions=[]
            for claim in claims:
                p=self._probability(claim,action,context_signature)
                predictions.append(PredictiveDistribution(
                    theory_id=claim.hypothesis_id,experiment_id=experiment_id,
                    probabilities=((("changed",),p),(("unchanged",),1-p)),
                ))
            score=self.science.score_experiment(posterior,tuple(predictions))
            available.append(AttributionExperiment(
                action,score.information_gain_bits,
                tuple(c.hypothesis_id for c in claims),tuple(predictions)))
        return min(available,key=lambda e:(-e.information_gain_bits,repr(e.action)))
