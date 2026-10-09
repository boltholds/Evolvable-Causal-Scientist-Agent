"""Prospective tests distinguish intervention claims from observational fit."""
import pytest
from ecsa.world_model.contracts import GroundAction,freeze_raw_value
from ecsa.world_model.effect_attribution import (
    EffectAttributionLedger, EffectExplanation, EffectEvidenceStatus,
    EffectClaim, FeatureTrial, CohortRef, CollectionMode,
)
from ecsa.world_model.attribution_experiments import AttributionExperimentSelector


def a(name):
    return GroundAction(name,())


def trial(i, name, change, ctx=("matched",)):
    return FeatureTrial(f"trial-{i}","global:signal",name,"global",ctx,True,change,
                        freeze_raw_value(0),freeze_raw_value(int(change)))


def cohort(name, episode="training", mode=CollectionMode.PASSIVE, protocol="log",
           prereg=None, matched=None, scheme=None):
    return CohortRef(name,episode,protocol,mode,prereg,matched,None,scheme)


def make_ledger():
    ledger=EffectAttributionLedger()
    ledger.observe(tuple(trial(i,"probe",True) for i in range(8)),
                   cohort=cohort("train-probe"))
    ledger.observe(tuple(trial(20+i,"comparison",False) for i in range(8)),
                   cohort=cohort("train-comparison"))
    return ledger


def registered(episode="validation",mode=CollectionMode.CONTROLLED):
    return cohort("validation-cohort",episode,mode,protocol="randomized-pair-v1",
                  prereg="prediction-1",matched="cohort-matched",scheme="registered-assignment-1")


def evidence(probe_true=True, episode_offset=100):
    return tuple(trial(episode_offset+i,"probe",probe_true) for i in range(8))+tuple(
        trial(episode_offset+10+i,"comparison",False) for i in range(8))


def test_eig_selects_discriminating_public_action():
    ledger=make_ledger()
    selected=AttributionExperimentSelector(ledger).select(
        claims=(
            EffectClaim("global:signal","probe",EffectExplanation.ACTION_DEPENDENT,
                        EffectEvidenceStatus.OBSERVATIONALLY_SUPPORTED,("t1",),(),(),(("matched",),)),
            EffectClaim("global:signal","probe",EffectExplanation.BACKGROUND,
                        EffectEvidenceStatus.PROPOSED,("t2",),(),(),(("matched",),)),
        ),
        candidate_actions=(a("uninvolved"),a("probe")),
        context_signature=("matched",),
    )
    assert selected.action==a("probe")
    assert selected.information_gain_bits>0
    assert len(selected.pre_registered_predictions)==2


def test_offline_only_evidence_cannot_gain_interventional_status():
    ledger=make_ledger()
    with pytest.raises(ValueError,match="controlled|independent"):
        ledger.confirm_intervention("global:signal",candidate=a("probe"),
                                    evidence=evidence(),cohort=registered(mode=CollectionMode.PASSIVE))


def test_confirmation_requires_disjoint_trial_and_episode_ids():
    ledger=make_ledger()
    with pytest.raises(ValueError,match="disjoint"):
        ledger.confirm_intervention("global:signal",candidate=a("probe"),
                                    evidence=evidence(),cohort=registered(episode="training"))
    with pytest.raises(ValueError,match="disjoint|duplicate"):
        ledger.confirm_intervention("global:signal",candidate=a("probe"),
                                    evidence=evidence(episode_offset=0),cohort=registered())


def test_matched_independent_intervention_promotes_only_tested_scope():
    ledger=make_ledger()
    claim=ledger.confirm_intervention("global:signal",candidate=a("probe"),
                                      evidence=evidence(),cohort=registered())
    assert claim.explanation is EffectExplanation.ACTION_DEPENDENT
    assert claim.status is EffectEvidenceStatus.INTERVENTIONALLY_SUPPORTED
    assert claim.context_signatures == (("matched",),)
    assert ledger.intervention_actions_executed==16
    assert len(claim.support_ids)==16


def test_failed_matched_intervention_contradicts_claim():
    ledger=make_ledger()
    claim=ledger.confirm_intervention("global:signal",candidate=a("probe"),
                                      evidence=evidence(probe_true=False),cohort=registered())
    assert claim.status is EffectEvidenceStatus.CONTRADICTED
    assert claim.contradiction_ids


def test_nonresettable_world_abstains_instead_of_simulating_controls():
    ledger=make_ledger()
    with pytest.raises(ValueError,match="matched|controls"):
        ledger.confirm_intervention("global:signal",candidate=a("probe"),
                                    evidence=tuple(trial(100+i,"probe",True) for i in range(8)),
                                    cohort=registered())


def test_invalid_claim_population_is_rejected():
    with pytest.raises(ValueError):
        AttributionExperimentSelector(EffectAttributionLedger()).select(
            claims=(),candidate_actions=(a("anything"),),context_signature=())
