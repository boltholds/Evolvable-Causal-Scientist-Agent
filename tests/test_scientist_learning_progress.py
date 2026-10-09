"""Opt-in scientist integration; existing scientific hypotheses remain authoritative."""
from ecsa.contracts import TheoryPosterior
from ecsa.autonomy.scientist import AutonomousScientist
from ecsa.world_model.contracts import (
    GroundAction, RawObservation, RawActionOutcome, InteractionTransition,
    freeze_raw_value,
)
from ecsa.world_model.experiments import (
    ContractExperiment, ContractExperimentCoordinator, ContractOutcomePrediction,
)
from ecsa.world_model.learning_progress import LearningProgressLedger, ProgressEvidence
from ecsa.world_model.kernel import WorldModelAcquisitionKernel, WorldModelUpdate
from ecsa.world_model.perception.base import PerceptualObservation


def choice(label, a=()):
    return ContractExperiment(label, GroundAction(label, tuple(freeze_raw_value(x) for x in a)), ())


def observed(i, name="again", gain=None):
    return ProgressEvidence(
        transition_id=f"past-{i}", action=GroundAction(name, ()),
        context_signature=(), before_signature=(), after_signature=(),
        predictive_gain=gain, uncertainty_reduction=0.,
        confirmed_hypothesis_ids=(), contradicted_hypothesis_ids=(),
        action_cost=1., independent_trial_group=None,
    )


class EmptyPublicPerception:
    def perceive(self, raw):
        return PerceptualObservation(raw.observation_id, (), ())


class ProposingWorld(WorldModelAcquisitionKernel):
    def observe_grounding(self, update):
        super().observe_grounding(update)
        return WorldModelUpdate(new_evidence_ids=("fresh-sample",),
                                added_contract_ids=("proposed-only",))


def raw_transition(i=0, success=False):
    return InteractionTransition(
        f"transition-{i}",
        RawObservation(f"before-{i}", i, freeze_raw_value({"status":"stable"})),
        GroundAction("again", ()),
        RawActionOutcome(success,freeze_raw_value({"reason":"public"})),
        RawObservation(f"after-{i}",i+1,freeze_raw_value({"status":"stable"})),
    )


def test_default_coordinator_ranking_unchanged():
    a,b=choice("opaque-A"),choice("opaque-B")
    coordinator=ContractExperimentCoordinator()
    expected=coordinator.select_bootstrap((a,b))
    assert coordinator.last_selection_breakdown is None
    assert expected==a


def test_progress_enabled_breaks_repeated_unproductive_actions():
    ledger=LearningProgressLedger(stagnation_horizon=3)
    for i in range(7):ledger.observe(observed(i,name="again"))
    coordinator=ContractExperimentCoordinator(progress=ledger,use_progress_scoring=True)
    a,b=choice("again"),choice("alternative")
    assert coordinator.select_bootstrap((a,b))==b
    assert coordinator.last_selection_breakdown is not None
    assert coordinator.last_selection_breakdown.experiment_id=="alternative"


def test_world_model_proposals_not_counted_as_confirmed():
    ledger=LearningProgressLedger()
    coordinator=ContractExperimentCoordinator(progress=ledger,use_progress_scoring=True)
    scientist=AutonomousScientist(
        world_model=ProposingWorld(),experiments=coordinator,
        perception=EmptyPublicPerception(),
    )
    result=scientist.observe_transition(raw_transition())
    assert result.added_contract_ids==("proposed-only",)
    assert ledger.total_observed==1
    last=ledger.recent[-1]
    assert not last.confirmed_hypothesis_ids
    assert last.predictive_gain is None


def test_failed_actions_count_cost_without_scientific_credit():
    ledger=LearningProgressLedger()
    coordinator=ContractExperimentCoordinator(progress=ledger,use_progress_scoring=True)
    scientist=AutonomousScientist(
        world_model=WorldModelAcquisitionKernel(), experiments=coordinator,
        perception=EmptyPublicPerception(),
    )
    scientist.observe_transition(raw_transition(success=False))
    assert ledger.total_observed==1
    assessed=ledger.assess(GroundAction("again",()),context_signature=())
    assert assessed.expected_gain is not None
    assert ledger.recent[-1].uncertainty_reduction > 0
    # Failure refines applicability uncertainty but is NOT independent
    # confirmation that the action has the hypothesized causal effect.
    assert ledger.recent[-1].confirmed_hypothesis_ids == ()
    assert ledger.recent[-1].action_cost>0


def test_existing_eig_decision_survives_progress_weight():
    ledger=LearningProgressLedger(stagnation_horizon=2)
    for i in range(8):ledger.observe(observed(i,name="informative"))
    a=ContractExperiment("a",GroundAction("informative",()),(
        ContractOutcomePrediction("th1","a",.95),
        ContractOutcomePrediction("th2","a",.05),
    ))
    b=ContractExperiment("b",GroundAction("uninformative",()),(
        ContractOutcomePrediction("th1","b",.5),
        ContractOutcomePrediction("th2","b",.5),
    ))
    coordinator=ContractExperimentCoordinator(progress=ledger,use_progress_scoring=True)
    posterior=TheoryPosterior((("th1",.5),("th2",.5)))
    result=coordinator.select_active(posterior=posterior,experiments=(a,b))
    assert result==a
    assert coordinator.last_selection_breakdown is not None
    assert coordinator.last_selection_breakdown.information_gain_bits>0
