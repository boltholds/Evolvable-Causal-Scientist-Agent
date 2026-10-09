"""ECSA world-model remains authoritative while effect claims stay provisional."""
from ecsa.autonomy.scientist import AutonomousScientist
from ecsa.world_model.contracts import (
    GroundAction, RawObservation, RawActionOutcome,
    InteractionTransition, freeze_raw_value,
)
from ecsa.world_model.kernel import WorldModelAcquisitionKernel
from ecsa.world_model.experiments import ContractExperimentCoordinator
from ecsa.world_model.effect_attribution import (
    EffectAttributionLedger, EffectEvidenceStatus,
)
from ecsa.world_model.perception.base import PerceptualObservation,ObservedFeature


class NumericView:
    def perceive(self,raw):
        item=raw.payload.thaw()
        return PerceptualObservation(
            raw.observation_id,(),
            (ObservedFeature("uninterpreted",freeze_raw_value(item["sensor"]),raw.observation_id),),
        )


def make_transition(index,schema,old,new,success=True):
    return InteractionTransition(
        f"public-effect-{index}",
        RawObservation(f"before-{index}",index,freeze_raw_value({"sensor":old})),
        GroundAction(schema,()),RawActionOutcome(success,freeze_raw_value({})),
        RawObservation(f"after-{index}",index+1,freeze_raw_value({"sensor":new})),
    )


def build(*,attribution=None):
    return AutonomousScientist(
        world_model=WorldModelAcquisitionKernel(learners=()),
        experiments=ContractExperimentCoordinator(),
        perception=NumericView(),
        effect_attribution=attribution,
    )


def test_only_real_observed_transitions_reach_attribution():
    ledger=EffectAttributionLedger(min_support=2)
    sci=build(attribution=ledger)
    assert not ledger.recent
    result=sci.observe_transition(make_transition(1,"uninterpreted-a",0,1))
    assert len(ledger.recent)==1
    assert ledger.recent[0].effect_key=="global:uninterpreted"
    assert ledger.recent[0].evidence_id.startswith("public-effect-1:")
    assert result.attribution_claim_ids==()


def test_observational_effect_does_not_admit_causal_contract():
    ledger=EffectAttributionLedger(min_support=2)
    sci=build(attribution=ledger)
    for i in range(16):
        sci.observe_transition(make_transition(i,"A" if i%2 else "B",
                            0,1 if i%2 else 0))
    claims=ledger.hypotheses("global:uninterpreted",action_schema_id="A")
    assert claims
    assert all(x.status is not EffectEvidenceStatus.INTERVENTIONALLY_SUPPORTED
               for x in claims)
    assert not sci.world_model.contract_hypotheses()


def test_claim_evidence_linked_to_original_grounding():
    ledger=EffectAttributionLedger()
    sci=build(attribution=ledger)
    outcome=sci.observe_transition(make_transition(12,"X",7,8))
    assert ledger.recent[0].evidence_id.endswith("global:uninterpreted")
    assert len(outcome.new_evidence_ids)>0


def test_default_kernel_behavior_unchanged_without_attribution():
    sci=AutonomousScientist(
        world_model=WorldModelAcquisitionKernel(),
        experiments=ContractExperimentCoordinator(),
        perception=NumericView(),
    )
    update=sci.observe_transition(make_transition(3,"Y",0,0))
    assert update.attribution_claim_ids==()
    assert update.new_evidence_ids


def test_action_budget_counts_controls():
    from ecsa.world_model.effect_attribution import FeatureTrial, CollectionMode, CohortRef
    ledger=EffectAttributionLedger(min_support=2)
    def rows(offset,schema,change):
        return tuple(FeatureTrial(f"bound-{i}","global:uninterpreted",schema,
                     "global",("matched",),True,change,
                     freeze_raw_value(0),freeze_raw_value(int(change)))
                     for i in range(offset,offset+4))
    ledger.observe(rows(0,"A",True),cohort=CohortRef(
        "source-a","source","passive",CollectionMode.PASSIVE,None,None,None,None))
    ledger.observe(rows(10,"B",False),cohort=CohortRef(
        "source-b","source","passive",CollectionMode.PASSIVE,None,None,None,None))
    confirmation=CohortRef("heldout","validation","registered",CollectionMode.CONTROLLED,
                     "predeclared","matched","", "rnd")
    # Malformed assignment is not an admissible shortcut to an intervention.
    import pytest
    with pytest.raises((TypeError,ValueError)):
        ledger.confirm_intervention("global:uninterpreted",candidate=GroundAction("A",()),
                                    evidence=rows(100,"A",True)+rows(200,"B",False),
                                    cohort=confirmation)
    assert ledger.intervention_actions_executed==0
