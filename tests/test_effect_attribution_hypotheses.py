"""Falsifiable competing background, action and state claims."""
import pytest

from ecsa.world_model.contracts import GroundAction, freeze_raw_value
from ecsa.world_model.effect_attribution import (
    EffectAttributionLedger, FeatureTrial, CohortRef,
    CollectionMode, EffectExplanation, EffectEvidenceStatus,
)


def row(index, schema, changed, context=("ambient:stable",)):
    return FeatureTrial(
        evidence_id=f"evidence-{index}",
        effect_key="global:reading", action_schema_id=schema,
        argument_role="global", context_signature=context,
        observed=True, changed=changed,
        value_before=freeze_raw_value(0),
        value_after=freeze_raw_value(int(changed)),
    )


def observe(ledger, schema, start, n, changed, context=("ambient:stable",),episode="train"):
    cohort=CohortRef(f"{episode}-{schema}-{start}",episode,"trace-v1",
                     CollectionMode.PASSIVE,None,None,GroundAction(schema,()),None)
    ledger.observe(tuple(row(i,schema,changed,context) for i in range(start,start+n)),
                   cohort=cohort)


def claims(ledger,schema):
    return ledger.hypotheses("global:reading",action_schema_id=schema)


def test_cross_action_counter_prefers_background():
    ledger=EffectAttributionLedger()
    for index,name in enumerate(("arbitrary-A","arbitrary-B","arbitrary-C")):
        observe(ledger,name,index*20,10,True)
    explanations={claim.explanation for claim in claims(ledger,"arbitrary-A")
                  if claim.status is EffectEvidenceStatus.OBSERVATIONALLY_SUPPORTED}
    assert EffectExplanation.BACKGROUND in explanations
    assert EffectExplanation.ACTION_DEPENDENT not in explanations


def test_state_confounding_blocks_action_claim():
    ledger=EffectAttributionLedger()
    observe(ledger,"A",0,12,True,("flag:1",))
    observe(ledger,"B",100,12,False,("flag:0",))
    candidates=claims(ledger,"A")
    assert any(item.explanation in (EffectExplanation.UNRESOLVED,
                                     EffectExplanation.STATE_DEPENDENT) for item in candidates)
    assert all(item.status is not EffectEvidenceStatus.INTERVENTIONALLY_SUPPORTED for item in candidates)
    assert all(item.explanation is not EffectExplanation.ACTION_DEPENDENT for item in candidates)


def test_matched_actions_support_action_dependent_proposal():
    ledger=EffectAttributionLedger()
    observe(ledger,"A",0,12,True)
    observe(ledger,"B",20,12,False)
    supported=[item for item in claims(ledger,"A")
               if item.explanation is EffectExplanation.ACTION_DEPENDENT]
    assert supported and supported[0].status is EffectEvidenceStatus.OBSERVATIONALLY_SUPPORTED
    assert supported[0].support_ids
    assert supported[0].assumptions


def test_action_state_interaction_distinguished():
    ledger=EffectAttributionLedger()
    observe(ledger,"A",0,10,True,("flag:on",))
    observe(ledger,"B",20,10,False,("flag:on",))
    observe(ledger,"A",40,10,False,("flag:off",))
    observe(ledger,"B",60,10,False,("flag:off",))
    assert any(v.explanation is EffectExplanation.INTERACTION for v in claims(ledger,"A"))


def test_no_overlap_yields_unresolved():
    ledger=EffectAttributionLedger()
    observe(ledger,"solo",0,5,True)
    outcome=claims(ledger,"solo")
    assert outcome and outcome[0].explanation is EffectExplanation.UNRESOLVED
    assert outcome[0].status in (EffectEvidenceStatus.INSUFFICIENT,
                                 EffectEvidenceStatus.PROPOSED)


def test_finite_memory_and_duplicate_trials_rejected():
    ledger=EffectAttributionLedger(max_recent=8)
    observe(ledger,"X",0,5,True)
    with pytest.raises(ValueError,match="duplicate"):
        observe(ledger,"X",0,5,True)
    observe(ledger,"X",10,9,False)
    assert len(ledger.recent)<=8


def test_offline_reports_never_claim_proven_causality():
    ledger=EffectAttributionLedger()
    observe(ledger,"X",0,9,True)
    observe(ledger,"Y",20,9,False)
    assert all(v.status is not EffectEvidenceStatus.INTERVENTIONALLY_SUPPORTED
               for v in claims(ledger,"X"))
