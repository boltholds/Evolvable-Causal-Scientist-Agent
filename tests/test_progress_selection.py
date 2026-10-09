"""Evidence-based experiment utility without world-specific action names."""
from dataclasses import fields

from ecsa.world_model.contracts import GroundAction, freeze_raw_value
from ecsa.world_model.experiments import ContractExperiment
from ecsa.world_model.learning_progress import ProgressEvidence, LearningProgressLedger
from ecsa.world_model.progress_selection import ProgressAwareSelection


def choice(label: str) -> ContractExperiment:
    return ContractExperiment(label, GroundAction(label, ()), ())


def record(ledger: LearningProgressLedger, n: int, action: GroundAction,
           *, gain=None, confirmed=(), group=None):
    ledger.observe(ProgressEvidence(
        transition_id=f"t-{n}", action=action,
        context_signature=("nearby",),
        before_signature=("same",), after_signature=("same",),
        predictive_gain=gain, uncertainty_reduction=0,
        confirmed_hypothesis_ids=confirmed, contradicted_hypothesis_ids=(),
        action_cost=1, independent_trial_group=group,
    ))


def test_unproductive_repeat_eventually_loses_to_alternative():
    a, b = choice("opaque-A"), choice("opaque-B")
    ledger = LearningProgressLedger(stagnation_horizon=3)
    for n in range(8):
        record(ledger,n,a.action)
    selector = ProgressAwareSelection(ledger)
    best, detail = selector.select((a,b), eig={"opaque-A":0.0, "opaque-B":0.0},
                                   context_signature=("nearby",))
    assert best == b
    assert detail.stagnation_penalty == 0
    assert detail.total_score > 0


def test_independent_replications_retain_priority():
    a, b = choice("gamma"), choice("delta")
    ledger = LearningProgressLedger(stagnation_horizon=3)
    for n in range(6):
        record(ledger,n,a.action,confirmed=("theory-1",),group=f"cohort-{n}")
    best, detail = ProgressAwareSelection(ledger).select(
        (a,b), eig={"gamma":0, "delta":0}, context_signature=("nearby",))
    assert best == a
    assert detail.progress_estimate > 0


def test_only_feasible_action_never_disappears():
    a=choice("only-legal")
    ledger=LearningProgressLedger(stagnation_horizon=2)
    for n in range(10):record(ledger,n,a.action)
    best,_=ProgressAwareSelection(ledger).select(
        (a,),eig={"only-legal":0.0},context_signature=("nearby",))
    assert best==a


def test_selection_exposes_complete_score_breakdown():
    a=choice("S")
    selected,score=ProgressAwareSelection(LearningProgressLedger()).select(
        (a,),eig={"S":0.03},context_signature=("nearby",))
    assert selected==a
    assert {f.name for f in fields(score)}=={
        "experiment_id","information_gain_bits","uncertainty",
        "progress_estimate","stagnation_penalty","cost","total_score"}
    assert score.information_gain_bits==0.03


def test_no_zero_division_with_unknown_gain():
    a=choice("no-data")
    score=ProgressAwareSelection(LearningProgressLedger()).select(
        (a,),eig={},context_signature=("nearby",))[1]
    assert score.progress_estimate==0.0
    assert score.uncertainty>0


def test_nonzero_science_eig_remains_primary():
    a,b=choice("sci-A"),choice("sci-B")
    ledger=LearningProgressLedger(stagnation_horizon=2)
    for n in range(7):record(ledger,n,a.action)
    best,detail=ProgressAwareSelection(ledger).select(
        (a,b),eig={"sci-A":0.12,"sci-B":0.10},context_signature=("nearby",))
    assert best==a
    assert detail.information_gain_bits==0.12


def test_invalid_eig_and_candidate_budget():
    selector=ProgressAwareSelection(LearningProgressLedger())
    a=choice("legal")
    import pytest
    with pytest.raises(ValueError):
        selector.select((),eig={},context_signature=("nearby",))
    with pytest.raises(ValueError):
        selector.select((a,),eig={"legal":float("nan")},context_signature=("nearby",))
