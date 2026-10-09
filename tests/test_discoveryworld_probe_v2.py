"""Diagnostics must compare every logged action alternative, not first alphabetic."""
import numpy as np
from ecsa.experimental.action_jepa import ActionJEPA, JepaConfig
from ecsa.benchmarks.discoveryworld.jepa_replay import ReplayTransitions
from ecsa.benchmarks.discoveryworld.jepa_probe import (
    score_actions, ProbeConfig, run_probe,
)


def samples(names, d=9, k=6):
    n=len(names)
    before=np.zeros((n,d),dtype=np.float32)
    after=np.zeros((n,d),dtype=np.float32)
    actions=np.zeros((n,k),dtype=np.float32)
    categories=sorted(set(names))
    for i,name in enumerate(names):
        before[i,0]=float(i%3)
        after[i,0]=float(i%3)+.1
        actions[i,categories.index(name)]=1.
    return ReplayTransitions(before,actions,after,tuple(names),tuple(range(n)))


def test_action_contrast_scores_all_logged_types_and_reports_macro_accuracy():
    train=samples(['Z','B','A']*4)
    heldout=samples(['B','A']*3)
    model=ActionJEPA(9,JepaConfig(steps=1,hidden_dim=12,latent_dim=5),action_dim=6)
    report=score_actions(model,train,heldout)
    assert report['action_choice_pairs']==12  # two OTHER action types per row
    assert report['action_choice_eligible_rows']==6
    assert set(report['action_choice_by_type'])=={'B','A'}
    assert report['action_choice_macro_accuracy']>=0
    assert report['action_choice_macro_accuracy']<=1


def test_probe_reports_public_space_delta_prediction_with_same_target_scale():
    train=samples(['A','B','C']*12)
    validation=samples(['B','A','C']*4)
    test=samples(['A','C','B']*4)
    from ecsa.benchmarks.discoveryworld.jepa_replay import ReplaySplits
    splits=ReplaySplits(train,validation,test,(0,),(1,),(2,),'independent_seed_holdout')
    output=run_probe(splits,ProbeConfig(steps=3,batch_size=12,hidden_dim=12,latent_dim=4,seed=2),device='cpu')
    assert output['shared_public_target']==True
    for k in ('jepa','action_shuffled','no_action'):
        m=output['arms'][k]
        assert np.isfinite(m['public_delta_mse'])
        assert np.isfinite(m['public_delta_improvement'])
        assert m['public_delta_heldout_count']==len(test.before)
        assert 'collapse_warning' in m


def test_zero_public_dynamics_has_undefined_not_perfect_improvement():
    from ecsa.benchmarks.discoveryworld.jepa_replay import ReplaySplits
    x=samples(['A','B','C']*8)
    static=ReplayTransitions(x.before,x.actions,x.before.copy(),x.action_names,x.steps)
    output=run_probe(ReplaySplits(static,static,static,(0,),(1,),(2,),
        'synthetic_stationary'),ProbeConfig(steps=2,batch_size=8,hidden_dim=12,latent_dim=4,seed=5),device='cpu')
    for arm in output['arms'].values():
        assert arm['public_delta_improvement'] is None
        assert arm['public_delta_target_nontrivial'] is False
