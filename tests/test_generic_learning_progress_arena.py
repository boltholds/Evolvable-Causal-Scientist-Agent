"""No benchmark semantics: same control loop in two unrelated public worlds."""
from ecsa.autonomy.scientist import AutonomousScientist
from ecsa.world_model.contracts import (
    RawObservation, RawActionSchema, RawActionOutcome, GroundAction,
    InteractionTransition, freeze_raw_value,
)
from ecsa.world_model.perception.base import PerceptualObservation, ObservedFeature
from ecsa.world_model.kernel import WorldModelAcquisitionKernel
from ecsa.world_model.learning_progress import ProgressEvidence, LearningProgressLedger
from ecsa.world_model.experiments import ContractExperimentCoordinator, ExperimentBudget


class OpaqueWorld:
    def __init__(self, labels, *, productive=False):
        self.labels=tuple(labels)
        self.productive=productive
        self.steps=0
        self.value=0

    @property
    def done(self):
        return self.steps>=25

    def observe_raw(self):
        return RawObservation(
            f"observation-{self.steps}",self.steps,
            freeze_raw_value({"status":{"reading":self.value}}),
        )

    def list_raw_actions(self):
        return tuple(RawActionSchema(name,(),freeze_raw_value({}))
                     for name in self.labels)

    def execute_raw_action(self, action):
        assert action.schema_id in self.labels
        if self.productive and len(self.labels)>1 and action.schema_id==self.labels[1]:
            self.value+=1
        self.steps+=1
        return RawActionOutcome(
            action.schema_id==self.labels[-1] and self.productive,
            freeze_raw_value({"observed":True}),
        )


class ScalarView:
    def perceive(self, observed):
        raw=observed.payload.thaw()["status"]["reading"]
        return PerceptualObservation(
            observed.observation_id,(),
            (ObservedFeature("reading",freeze_raw_value(raw),observed.observation_id),),
        )


def seed_idle(ledger, action, context):
    for i in range(8):
        ledger.observe(ProgressEvidence(
            transition_id=f"seeded-{action}-{i}",
            action=GroundAction(action,()),
            context_signature=context,before_signature=("stable",),
            after_signature=("stable",),predictive_gain=None,
            uncertainty_reduction=0.0,confirmed_hypothesis_ids=(),
            contradicted_hypothesis_ids=(),action_cost=1.0,
            independent_trial_group=None,
        ))


def make_agent(world, enabled, ledger=None):
    kernel=WorldModelAcquisitionKernel(max_contracts=4)
    progress=ledger or LearningProgressLedger(max_recent=8,stagnation_horizon=3)
    coordinator=ContractExperimentCoordinator(progress=progress,use_progress_scoring=enabled)
    scientist=AutonomousScientist(world_model=kernel,experiments=coordinator,perception=ScalarView())
    return scientist,kernel,progress


def run(world, steps, *, enabled=True, seeded=False):
    agent,kernel,ledger=make_agent(world,enabled)
    if seeded:
        seed_idle(ledger,world.labels[0],("global:reading",))
    sequence=[]
    for i in range(steps):
        before=world.observe_raw()
        experiment=agent.choose_experiment(before,world.list_raw_actions(),
                                            freeze_raw_value({}),ExperimentBudget(8))
        sequence.append(experiment.action.schema_id)
        result=world.execute_raw_action(experiment.action)
        agent.observe_transition(InteractionTransition(
            f"online-{i}",before,experiment.action,result,world.observe_raw(),
        ))
    return sequence,kernel,ledger


def test_repetition_reduces_without_domain_cues():
    old=run(OpaqueWorld(("A","B"),productive=True),1,enabled=False,seeded=True)[0]
    new=run(OpaqueWorld(("A","B"),productive=True),1,enabled=True,seeded=True)[0]
    assert old==["A"]
    assert new==["B"]


def test_all_actions_uninformative_stays_bounded():
    _,kernel,ledger=run(OpaqueWorld(("ONLY",)),24,enabled=True)
    assert ledger.total_observed==24
    assert len(ledger.recent)<=8
    assert len(kernel.contract_hypotheses())<=4


def test_no_unbounded_contract_generation_under_cycles():
    _,kernel,ledger=run(OpaqueWorld(("Z",)),24,enabled=True)
    assert len(kernel.contract_hypotheses())==0
    assert ledger.assess(GroundAction("Z",()),context_signature=("global:reading",)).stagnation_penalty>0


def test_distinct_domain_symbols_preserve_decisions():
    first=run(OpaqueWorld(("A","B"),productive=True),1,enabled=True,seeded=True)[0]
    second=run(OpaqueWorld(("M","N"),productive=True),1,enabled=True,seeded=True)[0]
    assert first==["B"] and second==["N"]


def test_action_budget_identical_across_ablations():
    a,_,_=run(OpaqueWorld(("F","G")),10,enabled=False)
    b,_,_=run(OpaqueWorld(("F","G")),10,enabled=True)
    assert len(a)==len(b)==10


def test_scientist_records_prequential_uncertainty_progress_then_saturates():
    _,_,ledger=run(OpaqueWorld(("ONLY",)),24,enabled=True)
    gains=[event.uncertainty_reduction for event in ledger.recent]
    assert gains[0]>gains[-1]>=0
    assert ledger.assess(
        GroundAction("ONLY",()),context_signature=("global:reading",),
    ).repeated_uninformative_count>=3
