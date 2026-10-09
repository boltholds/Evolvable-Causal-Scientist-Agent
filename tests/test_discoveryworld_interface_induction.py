"""Source-blind contracts: assertions use unfamiliar actions/properties on purpose."""
from pathlib import Path
import json
import pytest

from ecsa.benchmarks.discoveryworld.interface_induction import (
    PublicTransition, InterfaceLearner, InterfaceStatus,
    LearnedInterfaceEncoder, load_contract_episode,
)


def obs(*entities, step=0):
    return {"ui": {"accessibleEnvironmentObjects": list(entities), "worldStep": step}}


def obj(ref, **props):
    return {"uuid": ref, **props}


def transition(i, *, name="Z9", source=101, before=3., after=4.,
               target=101, outcome=True, property_name="charge"):
    return PublicTransition(
        evidence_id=f"episode-train:{i}",
        before=obs(obj(source, **{property_name: before}), obj(202, **{property_name: 50.})),
        action={"action": name, "arg1": target},
        after=obs(obj(source, **{property_name: after}), obj(202, **{property_name: 50.})),
        success=outcome,
    )


def test_learns_action_argument_role_and_effect_without_benchmark_names():
    learner=InterfaceLearner(min_support=3)
    for i in range(5):
        learner.observe(transition(i, before=float(i), after=float(i+1)))
    contracts=learner.proposals()
    assert len(contracts)==1
    c=contracts[0]
    assert c.schema_id=="Z9"
    assert c.parameter_names==("arg1",)
    assert c.roles[0].object_reference_fraction==1.
    assert any(e.role=="arg1" and e.feature=="charge" and e.support==5 for e in c.effects)
    assert c.status is InterfaceStatus.PROPOSED


def test_independent_cross_episode_validation_admits_only_predictive_contracts():
    learner=InterfaceLearner(min_support=3)
    for i in range(5):learner.observe(transition(i,before=i,after=i+1))
    good=tuple(PublicTransition(f"valid-good:{i}",
            obs(obj(19,charge=i)),{"action":"Z9","arg1":19},
            obs(obj(19,charge=i+1)),True) for i in range(4))
    checked=learner.validate(good,min_validation=4)
    assert checked[0].status is InterfaceStatus.SUPPORTED
    assert checked[0].correct_effect_rate==1.
    assert checked[0].effect_f1==1.
    assert checked[0].causally_validated is False
    assert checked[0].train_evidence_count==5
    assert checked[0].validation_evidence_count==4
    bad=tuple(PublicTransition(f"valid-bad:{i}",
            obs(obj(21,charge=i)),{"action":"Z9","arg1":21},
            obs(obj(21,charge=i)),True) for i in range(4))
    rejected=learner.validate(bad,min_validation=4)
    assert rejected[0].status is InterfaceStatus.REJECTED
    assert rejected[0].correct_effect_rate==0.
    assert rejected[0].effect_f1==0.
    assert learner.proposals()[0].status is InterfaceStatus.PROPOSED


def test_failed_and_unseen_action_are_not_silently_treated_as_successful_effects():
    learner=InterfaceLearner(min_support=3)
    for i in range(5):learner.observe(transition(i,outcome=False,after=i,before=i))
    c=learner.proposals()[0]
    assert c.success_rate==0.
    assert not c.effects
    assert learner.evaluate((transition(55,name="NEW"),)).evaluated_count==0


def test_preconditions_are_inferred_from_successful_and_failed_actions_not_handcoded():
    def trial(i, enabled, success, evidence_prefix):
        return PublicTransition(
            f'{evidence_prefix}-{i}',
            obs(obj(800+i,armed=enabled,charge=float(i))),
            {'action':'UNKNOWN-ACTION','arg1':800+i},
            obs(obj(800+i,armed=enabled,charge=float(i+(1 if success else 0)))),
            success,
        )
    learner=InterfaceLearner(min_support=3)
    for i in range(8):
        learner.observe(trial(i,bool(i%2),bool(i%2),'train'))
    contract=learner.proposals()[0]
    assert any(c.role=='arg1' and c.feature=='armed' and c.value is True
               for c in contract.preconditions)
    assert all(c.value != 800 for c in contract.preconditions)
    valid=tuple(trial(100+i,bool(i%2),bool(i%2),'heldout') for i in range(8))
    result=learner.validate(valid,min_validation=4)[0]
    assert result.applicability_accuracy==1.


def test_encoder_invariant_to_object_order_and_new_ids_but_sensitive_to_bound_object():
    learner=InterfaceLearner(min_support=2)
    for i in range(3):learner.observe(transition(i))
    encoder=LearnedInterfaceEncoder.fit(learner.proposals(),width=96)
    before=obs(obj(101,charge=3),obj(202,charge=7),step=10)
    permuted=obs(obj(992,charge=7),obj(993,charge=3),step=999)
    a=encoder.encode_before(before,{"action":"Z9","arg1":101})
    b=encoder.encode_before(permuted,{"action":"Z9","arg1":993})
    c=encoder.encode_before(before,{"action":"Z9","arg1":202})
    assert a.shape==(96,)
    assert (a==b).all()
    assert not (a==c).all()


def test_jepa_state_target_representation_does_not_depend_on_previous_action():
    learner=InterfaceLearner(min_support=2)
    for i in range(4):learner.observe(transition(i))
    encoder=LearnedInterfaceEncoder.fit(learner.proposals(),width=96)
    state=obs(obj(101,charge=1.),obj(202,charge=9.))
    other=obs(obj(301,charge=9.),obj(302,charge=1.))
    assert (encoder.encode_observation(state)==encoder.encode_observation(other)).all()
    assert (encoder.encode_observation(state)==encoder.encode_observation(state)).all()


def test_encoder_rejects_oracle_keys_even_if_nested():
    learner=InterfaceLearner(min_support=1)
    learner.observe(transition(0))
    encoder=LearnedInterfaceEncoder.fit(learner.proposals(),width=64)
    with pytest.raises(ValueError,match="oracle"):
        encoder.encode_before({"ui":{"scorecard":{"completedSuccessfully":True}}},{"action":"Z9","arg1":101})


def test_two_compatible_public_views_of_one_object_are_merged_without_leaking_uuid():
    learner=InterfaceLearner(min_support=1)
    before={'ui':{
        'accessibleEnvironmentObjects':[obj(101,charge=3)],
        'nearbyObjects':{'objects':{'objects':[obj(101,charge=3,name='thing')]}}
    }}
    after={'ui':{'accessibleEnvironmentObjects':[obj(101,charge=4,name='thing')]}}
    learner.observe(PublicTransition('merged-view',before,{'action':'Z9','arg1':101},after,True))
    effects=learner.proposals()[0].effects
    assert any(e.role=='arg1' and e.feature=='charge' for e in effects)
    # No public source identifier becomes a discovered numeric feature.
    assert 'uuid' not in learner.proposals()[0].observed_properties


def test_contract_and_feature_interface_adapt_to_new_observed_effect_without_relabelling():
    learner=InterfaceLearner(min_support=3,max_recent_examples=5)
    for i in range(5):learner.observe(transition(i))
    original=learner.proposals()[0]
    assert {e.feature for e in original.effects}=={'charge'}
    prior=LearnedInterfaceEncoder.fit((original,),width=64)
    drift=tuple(transition(i+100,property_name='potential',before=i,after=i+1)
                for i in range(5))
    result=learner.adapt(drift)
    latest=learner.proposals()[0]
    assert {e.feature for e in latest.effects}=={'potential'}
    assert original in result.previous
    assert latest in result.current
    assert latest.status is InterfaceStatus.PROPOSED
    assert result.requires_independent_validation is True
    evolved=LearnedInterfaceEncoder.fit(result.current,width=64)
    assert 'potential' in evolved.vocabulary
    assert 'potential' not in prior.vocabulary


def test_validation_cannot_reuse_training_evidence():
    learner=InterfaceLearner(min_support=2)
    data=[transition(i) for i in range(4)]
    for row in data:learner.observe(row)
    with pytest.raises(ValueError,match="overlap"):
        learner.validate(tuple(data),min_validation=3)


def test_loader_uses_only_public_logs_and_keeps_action_outcome_unknown_if_missing(tmp_path:Path):
    root=tmp_path/'seed-2'/'cold';root.mkdir(parents=True)
    (root/'run.json').write_text(json.dumps({'scenario':'Reactor Lab','difficulty':'Normal','seed':2}))
    (root/'final_scorecard.json').write_text('SENSITIVE NOT JSON')
    (root/'actions.jsonl').write_text(json.dumps({'step':0,'action':{'action':'Z9','arg1':3}})+'\n')
    (root/'observations.jsonl').write_text('\n'.join([
        json.dumps({'step':0,'phase':'pre','observation':obs(obj(3,charge=4))}),
        json.dumps({'step':1,'phase':'post','observation':obs(obj(3,charge=7))}),
    ])+'\n')
    rows=load_contract_episode(root)
    assert len(rows)==1 and rows[0].success is None
    assert rows[0].evidence_id.endswith(':0')


def test_loader_rejects_duplicate_or_misaligned_public_steps(tmp_path:Path):
    root=tmp_path/'seed-0'/'cold';root.mkdir(parents=True)
    (root/'run.json').write_text(json.dumps({'scenario':'Reactor Lab','difficulty':'Normal','seed':0}))
    (root/'actions.jsonl').write_text('\n'.join([json.dumps({'step':0,'action':{'action':'Q'}})]*2)+'\n')
    (root/'observations.jsonl').write_text('\n'.join([
        json.dumps({'step':0,'phase':'pre','observation':obs(obj(1,charge=0))}),
        json.dumps({'step':1,'phase':'post','observation':obs(obj(1,charge=1))}),
    ])+'\n')
    with pytest.raises(ValueError,match='duplicate'):
        load_contract_episode(root)


def test_public_nonobject_readings_and_messages_can_be_induced_without_semantic_map():
    learner=InterfaceLearner(min_support=2)
    for index in range(3):
        learner.observe(PublicTransition(
            evidence_id=f'public-result:{index}',
            before={'ui':{'publicReading':float(index), 'message':'before'}},
            action={'action':'CUSTOM-TEST'},
            after={'ui':{'publicReading':float(index+1), 'message':'new measurement'}},
            success=True,
        ))
    candidate=learner.proposals()[0]
    assert any(e.role=='global' and e.feature=='ui/publicReading' for e in candidate.effects)
    encoder=LearnedInterfaceEncoder.fit((candidate,),width=96)
    assert 'global:ui/message' in encoder.vocabulary
    a=encoder.encode_observation({'ui':{'publicReading':4.,'message':'before'}})
    b=encoder.encode_observation({'ui':{'publicReading':5.,'message':'new measurement'}})
    assert not (a==b).all()


def test_unidentified_public_lists_have_order_invariant_observation_features():
    learner=InterfaceLearner(min_support=1)
    learner.observe(PublicTransition('unidentified-0',
        {'ui':{'reports':[{'message':'alpha'}, {'message':'beta'}]}},
        {'action':'CUSTOM-TEST'},
        {'ui':{'reports':[{'message':'gamma'}, {'message':'beta'}]}},True))
    encoder=LearnedInterfaceEncoder.fit(learner.proposals(),width=96)
    first={'ui':{'reports':[{'message':'alpha'}, {'message':'beta'}]}}
    reordered={'ui':{'reports':[{'message':'beta'}, {'message':'alpha'}]}}
    assert (encoder.encode_observation(first)==encoder.encode_observation(reordered)).all()


def test_induced_interface_study_uses_disjoint_seed_cohorts(tmp_path:Path):
    from ecsa.benchmarks.discoveryworld.interface_induction import load_interface_splits
    for seed in range(5):
        folder=tmp_path/f'seed-{seed}'/'cold';folder.mkdir(parents=True)
        (folder/'run.json').write_text(json.dumps({'scenario':'Reactor Lab','difficulty':'Normal','seed':seed}))
        with (folder/'observations.jsonl').open('w') as obs_log, (folder/'actions.jsonl').open('w') as actions_log:
            for i in range(12):
                feature='unseen_only_in_test' if seed==4 else 'charge'
                before=obs(obj(seed*1000+i,**{feature:float(i)}))
                after=obs(obj(seed*1000+i,**{feature:float(i+1)}))
                obs_log.write(json.dumps({'step':i,'phase':'pre','observation':before})+'\n')
                obs_log.write(json.dumps({'step':i+1,'phase':'post','observation':after})+'\n')
                actions_log.write(json.dumps({'step':i,'action':{'action':'Z9','arg1':seed*1000+i}})+'\n')
    output=load_interface_splits(tmp_path,arm='cold',mode='study',observation_dim=64,action_dim=32)
    splits,report=output.splits,output
    assert splits.train_seeds==(0,1,2)
    assert splits.validation_seeds==(3,)
    assert splits.test_seeds==(4,)
    assert 'charge' in report.feature_vocabulary
    assert 'unseen_only_in_test' not in report.feature_vocabulary
    assert splits.train.before.shape==(36,64)
    assert splits.test.actions.shape==(12,32)
    assert report.contract_validation.evaluated_count==12
    assert report.contract_heldout.evaluated_count==12
    assert report.contract_validation.per_schema[0].status is InterfaceStatus.SUPPORTED
    from ecsa.benchmarks.discoveryworld.jepa_probe import run_probe_from_logs
    result=run_probe_from_logs(tmp_path,output=tmp_path/'report.json',mode='study',
        arm='cold',device='cpu',observation_dim=64,action_dim=32,
        steps=6,batch_size=12,latent_dim=4,hidden_dim=12,seed=1,
        representation='induced_interface')
    assert result['representation']=='induced_interface'
    assert result['contract_induction']['train_seed_ids']==[0,1,2]
    assert result['contract_induction']['validation']['evaluated_count']==12
    assert result['contract_induction']['heldout']['evaluated_count']==12
    assert result['contract_induction']['causally_validated']==False
    assert result['contract_induction']['unseen_test_properties']==['unseen_only_in_test']


def test_induced_interface_featurizes_action_arguments_from_observed_roles():
    learner=InterfaceLearner(min_support=2)
    for i in range(4):learner.observe(transition(i))
    encoder=LearnedInterfaceEncoder.fit(learner.proposals(),width=96)
    action_a={'action':'Z9','arg1':101}
    action_b={'action':'Z9','arg1':202}
    assert encoder.encode_action(action_a,obs(obj(101,charge=2),obj(202,charge=5))).shape==(48,)
    assert not (encoder.encode_action(action_a,obs(obj(101,charge=2)))==encoder.encode_action(action_b,obs(obj(101,charge=2)))).all()
