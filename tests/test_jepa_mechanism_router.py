"""Router RED/GREEN gates: online evidence, safe novelty, and real JEPA port."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from ecsa.experimental.jepa_world import Action, TransitionDataset
from ecsa.experimental.jepa_mechanism_router import (
    Checkpoint, ConfirmationEvidence, JepaPredictivePort,
    MechanismRouter, RouterConfig, RouteKind, BOCPDChangeSignal,
)
from ecsa.experimental.action_jepa import ActionJEPA, JepaConfig


class SimpleExpert:
    """Unprivileged deterministic predictive port, no simulator mechanism enum."""
    def __init__(self, increments: tuple[tuple[float,float], ...], space='shared-v1'):
        self.increments=np.asarray(increments,dtype=np.float32)
        self.space_id=space

    def encode(self, observations: np.ndarray) -> np.ndarray:
        return np.asarray(observations,dtype=np.float32)

    def predict(self, before: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.asarray(before,dtype=np.float32) + np.asarray(actions,dtype=np.float32) @ self.increments


INCREMENTS={
    'one':((.43,.04),(-.08,.35),(.0,.0)),
    'two':((-.45,.04),(-.08,.35),(.0,.0)),
    'novel':((.05,-.58),(-.08,.35),(.0,.0)),
}


def dataset(regime: str, seed: int, count: int=240) -> TransitionDataset:
    rng=np.random.default_rng(seed)
    before=rng.normal(0,.6,(count,2)).astype(np.float32)
    index=rng.integers(0,3,count)
    actions=np.eye(3,dtype=np.float32)[index]
    after=before+actions@np.asarray(INCREMENTS[regime],dtype=np.float32)
    after+=rng.normal(0,.025,after.shape).astype(np.float32)
    return TransitionDataset(before,actions,after)


def checkpoint(regime: str, *, name: str | None=None, seed: int=0) -> Checkpoint:
    cal=dataset(regime,1200+seed,count=300)
    return Checkpoint(
        checkpoint_id=name or regime,
        predictor=SimpleExpert(INCREMENTS[regime]),
        calibration=cal,
        calibration_ids=tuple(f'cal-{regime}-{seed}-{i}' for i in range(len(cal.before))),
        training_ids=tuple(f'train-{regime}-{seed}-{i}' for i in range(200)),
    )


def router(**kwargs):
    config=RouterConfig(**kwargs)
    return MechanismRouter((checkpoint('one'),checkpoint('two')),config=config)


def feed(model: MechanismRouter, data: TransitionDataset, offset: int=0, prefix='stream'):
    events=[]
    for i in range(len(data.before)):
        events.append(model.observe(
            transition_id=f'{prefix}-{offset+i}',
            before=data.before[i], action=data.actions[i], after=data.after[i]))
    return events


def test_recognizes_known_modes_and_returns_without_retraining():
    model=router(switch_hazard=.12,accept_probability=.70,novelty_p=.008,novelty_streak=4)
    first=feed(model,dataset('one',31,60))
    middle=feed(model,dataset('two',32,70),prefix='middle')
    last=feed(model,dataset('one',33,70),prefix='return')
    assert first[-1].selected_checkpoint=='one'
    assert middle[-1].selected_checkpoint=='two'
    assert last[-1].selected_checkpoint=='one'
    assert model.checkpoint_ids==('one','two')
    assert not any(row.kind is RouteKind.PROPOSE_NEW for row in first+middle+last)
    assert first[-1].posterior.probability('one')>.70
    assert last[-1].posterior.probability('one')>.70


def test_ambiguous_posterior_can_request_information_gain_experiment():
    model=router(accept_probability=.99999)
    before=np.array([0.12,-.20],dtype=np.float32)
    action,score=model.select_experiment(before, candidate_actions=tuple(Action))
    assert isinstance(action, Action)
    assert score.information_gain_bits>=0
    assert action is Action.PULSE  # only this action separates the two experts


def test_stationary_noise_does_not_invent_new_model():
    model=router(novelty_streak=3)
    events=feed(model,dataset('one',100,100))
    assert not any(event.kind is RouteKind.PROPOSE_NEW for event in events)
    assert model.pending_proposal is None


def test_novel_mode_creates_proposal_without_promoting_it():
    model=router(novelty_streak=3,novelty_p=.03)
    events=feed(model,dataset('novel',101,240))
    assert any(e.kind is RouteKind.PROPOSE_NEW for e in events)
    assert model.pending_proposal is not None
    assert 'novel' not in model.checkpoint_ids
    assert len(model.pending_proposal.evidence_ids)>=3


def test_confirmation_requires_independent_evidence_and_predictive_advantage():
    model=router(novelty_streak=3,novelty_p=.03)
    feed(model,dataset('novel',103,240))
    assert model.pending_proposal is not None
    new=checkpoint('novel',name='candidate-x')
    audit=dataset('novel',300,60)
    ids=tuple(f'independent-novel-{i}' for i in range(len(audit.before)))
    evidence=ConfirmationEvidence(
        hypothesis_id=model.pending_proposal.proposal_id,
        validation=audit,
        validation_ids=ids,
    )
    outcome=model.admit_checkpoint(new,evidence,prior_mass=.15)
    assert outcome.admitted
    assert outcome.candidate_mse<outcome.best_existing_mse
    assert 'candidate-x' in model.checkpoint_ids
    replay=feed(model,dataset('novel',301,60),prefix='known-novel')
    assert sum(row.selected_checkpoint=='candidate-x' for row in replay)>=40
    assert max(row.posterior.probability('candidate-x') for row in replay)>.95


def test_candidate_cannot_use_proposal_or_training_data_as_confirmation():
    model=router(novelty_streak=3,novelty_p=.03)
    feed(model,dataset('novel',107,240))
    assert model.pending_proposal is not None
    new=checkpoint('novel',name='candidate-x')
    audit=dataset('novel',305,40)
    with pytest.raises(ValueError,match='disjoint'):
        model.admit_checkpoint(new,ConfirmationEvidence(
            hypothesis_id=model.pending_proposal.proposal_id,
            validation=audit,
            validation_ids=(new.training_ids[0],)+tuple(f'good-{i}' for i in range(39))))
    with pytest.raises(ValueError,match='disjoint'):
        model.admit_checkpoint(new,ConfirmationEvidence(
            hypothesis_id=model.pending_proposal.proposal_id,
            validation=audit,
            validation_ids=(model.pending_proposal.evidence_ids[0],)+tuple(f'good2-{i}' for i in range(39))))
    assert model.checkpoint_ids==('one','two')


def test_known_model_rejected_as_new_hypothesis():
    model=router(novelty_streak=3,novelty_p=.03)
    feed(model,dataset('novel',111,240))
    proposal=model.pending_proposal
    assert proposal is not None
    duplicate=checkpoint('one',name='duplicate')
    data=dataset('one',900,50)
    validation=ConfirmationEvidence(proposal.proposal_id,data,tuple(f'valid-{i}' for i in range(50)))
    answer=model.admit_checkpoint(duplicate,validation)
    assert answer.admitted is False
    assert model.checkpoint_ids==('one','two')
    assert model.pending_proposal is None
    subsequent=feed(model,dataset('one',901,90),prefix='recovery')
    assert sum(x.selected_checkpoint=='one' for x in subsequent)>40


def test_different_latent_spaces_or_invalid_observations_fail_closed():
    first=checkpoint('one')
    second=Checkpoint('other',SimpleExpert(INCREMENTS['two'],'incompatible'),checkpoint('two').calibration,
                      checkpoint('two').calibration_ids,checkpoint('two').training_ids)
    with pytest.raises(ValueError,match='latent'):
        MechanismRouter((first,second))
    model=router()
    with pytest.raises(ValueError,match='nonfinite'):
        model.observe(transition_id='bad',before=np.array([float('nan'),0]),
                      action=np.array([1,0,0],dtype=np.float32),after=np.array([0.,0.]))
    with pytest.raises(ValueError,match='duplicate'):
        model.observe(transition_id='ok',before=np.array([0.,0.]),
                      action=np.array([1,0,0],dtype=np.float32),after=np.array([.4,0.]))
        model.observe(transition_id='ok',before=np.array([0.,0.]),
                      action=np.array([1,0,0],dtype=np.float32),after=np.array([.4,0.]))


def test_real_jepa_port_shared_encoder_fingerprint_and_no_gradients():
    cfg=JepaConfig(latent_dim=8,hidden_dim=16,steps=3,batch_size=8,seed=4)
    model=ActionJEPA(24,cfg).eval()
    port=JepaPredictivePort(model)
    before=np.ones((6,24),dtype=np.float32)
    actions=np.eye(3,dtype=np.float32)[[0,1,2,0,1,2]]
    z=port.predict(before,actions)
    assert z.shape==(6,8)
    assert np.array_equal(z,port.predict(before,actions))
    assert port.encode(before).shape==(6,8)
    assert all(p.grad is None for p in model.parameters())
    assert port.space_id==JepaPredictivePort(model).space_id


def test_bocpd_change_adapter_is_opt_in_and_fails_helpfully_without_dependency():
    adapter=BOCPDChangeSignal(min_points=8,min_reset_drop=3)
    with pytest.raises(RuntimeError,match='bocpd'):
        adapter.add(0.1)


def test_port_snapshots_encoder_and_predictions_when_caller_trains_again():
    cfg=JepaConfig(latent_dim=8,hidden_dim=16,steps=3,batch_size=8,seed=5)
    model=ActionJEPA(24,cfg).eval()
    frozen=JepaPredictivePort(model)
    states=np.ones((4,24),dtype=np.float32)
    acts=np.eye(3,dtype=np.float32)[[0,1,2,0]]
    initial=frozen.predict(states,acts)
    with torch.no_grad():
        for p in model.encoder.parameters():p.add_(1.)
    assert np.array_equal(initial,frozen.predict(states,acts))
    assert frozen.space_id != JepaPredictivePort(model).space_id


def test_better_fit_to_old_calibration_is_a_repair_not_a_new_mechanism():
    one=checkpoint('one',name='old-one')
    stale=Checkpoint(one.checkpoint_id,SimpleExpert(((.30,.04),(-.08,.35),(.0,.0))),
                     one.calibration,one.calibration_ids,one.training_ids)
    model=MechanismRouter((stale,checkpoint('two')),config=RouterConfig(novelty_streak=3,novelty_p=.03))
    feed(model,dataset('novel',912,240))
    assert model.pending_proposal is not None
    candidate=checkpoint('one',name='fresh-one')
    calibration=dataset('one',913,80)
    ids=tuple(f'new-independent-{i}' for i in range(len(calibration.before)))
    outcome=model.admit_checkpoint(candidate,ConfirmationEvidence(
        model.pending_proposal.proposal_id,calibration,ids))
    assert outcome.candidate_mse < outcome.best_existing_mse
    assert outcome.admitted is False, 'New model that only repairs known regime must not create novel mechanism'
