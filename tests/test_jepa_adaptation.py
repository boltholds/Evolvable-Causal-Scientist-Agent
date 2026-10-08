"""RED/GREEN: one action JEPA learns changed dynamics without privileged law labels."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecsa.experimental.jepa_world import Action, Mechanism, collect_transitions
from ecsa.experimental.action_jepa import JepaConfig, train_jepa
from ecsa.experimental.jepa_recurrence_world import RecurrenceWorld, observe_recurrence_stream, segment
from ecsa.experimental.jepa_adaptation import (
    AdaptationConfig, adapt_predictor, predict_mse, route_by_prediction,
)
from ecsa.experimental.jepa_adaptation_arena import run_adaptation_arena


def test_fixture_switches_back_without_disclosing_physics():
    world=RecurrenceWorld(switch_at=(1,2),seed=19,noise_std=0)
    start=world.reset(level=.1,gate=False)
    first=world.step(Action.PULSE)
    assert not np.array_equal(first,start)
    start=world.reset(level=.1,gate=False)
    middle=world.step(Action.PULSE)
    assert np.array_equal(middle,start)
    start=world.reset(level=.1,gate=False)
    final=world.step(Action.PULSE)
    assert not np.array_equal(final,start)
    assert not hasattr(world,'public_mechanism')
    with pytest.raises(ValueError,match='boundaries'):
        RecurrenceWorld(switch_at=(5,5))
    with pytest.raises(ValueError,match='boundaries'):
        RecurrenceWorld(switch_at=(3,2))


def test_recurrence_stream_is_blinded_reproducible_and_sliced_only_by_evaluator():
    a=observe_recurrence_stream(RecurrenceWorld(switch_at=(12,24),seed=4,noise_std=.16),count=36,seed=7)
    b=observe_recurrence_stream(RecurrenceWorld(switch_at=(12,24),seed=4,noise_std=.16),count=36,seed=7)
    assert set(a.__dataclass_fields__)=={'before','actions','after'}
    assert np.array_equal(a.before,b.before) and np.array_equal(a.after,b.after)
    assert np.array_equal(a.actions,b.actions)
    assert segment(a,12,24).before.shape==(12,24)
    with pytest.raises(ValueError,match='slice'):
        segment(a,24,24)
    with pytest.raises(ValueError,match='count'):
        observe_recurrence_stream(RecurrenceWorld(switch_at=(2,4)),count=0,seed=1)


def _fixtures(seed=2):
    add=collect_transitions(mechanism=Mechanism.ADDITIVE,count=180,seed=101+seed,noise_std=.16)
    new=collect_transitions(mechanism=Mechanism.GATED,count=40,seed=201+seed,noise_std=.16)
    old_test=collect_transitions(mechanism=Mechanism.ADDITIVE,count=60,seed=300+seed,noise_std=.16)
    new_test=collect_transitions(mechanism=Mechanism.GATED,count=60,seed=400+seed,noise_std=.16)
    model,_=train_jepa(add,JepaConfig(steps=24,seed=seed,batch_size=48))
    return model,add,new,old_test,new_test


def test_predictor_repair_keeps_single_shared_embedding_frame_and_does_not_mutate_original():
    model,old,new,old_test,new_test=_fixtures()
    before_encoder={key:value.detach().clone() for key,value in model.encoder.state_dict().items()}
    before_predictor={key:value.detach().clone() for key,value in model.predictor.state_dict().items()}
    cfg=AdaptationConfig(steps=24,batch_size=32,max_new_examples=32,max_replay_examples=36,seed=8)
    naive,trace_naive=adapt_predictor(model,new,config=cfg,validation=new_test)
    replay,trace_replay=adapt_predictor(model,new,old_replay=old,config=cfg,validation=new_test)
    for updated in (naive,replay):
        assert all(torch.equal(updated.encoder.state_dict()[key],value)
                   for key,value in before_encoder.items())
        assert all(torch.equal(model.encoder.state_dict()[key],value)
                   for key,value in before_encoder.items())
        assert all(not torch.equal(updated.predictor.state_dict()[key],value)
                   for key,value in before_predictor.items() if key.endswith('weight'))
        assert np.isfinite(predict_mse(updated,new_test))
        assert np.isfinite(predict_mse(updated,old_test))
    assert all(torch.equal(model.predictor.state_dict()[key],value)
               for key,value in before_predictor.items())
    assert trace_naive.updates==trace_replay.updates==24
    assert trace_naive.new_examples==trace_replay.new_examples==32
    assert trace_naive.old_examples==0
    assert trace_replay.old_examples==36
    assert len(trace_replay.validation_mse_by_update)>=2
    assert trace_replay.validation_mse_by_update[-1][0]==24


def test_repair_validation_cannot_change_training_updates_or_model_weights():
    model,old,new,_,new_test=_fixtures(seed=4)
    cfg=AdaptationConfig(steps=12,batch_size=24,max_new_examples=24,max_replay_examples=24,seed=12)
    with_val,_=adapt_predictor(model,new,old_replay=old,config=cfg,validation=new_test)
    without_val,_=adapt_predictor(model,new,old_replay=old,config=cfg)
    assert all(torch.equal(with_val.predictor.state_dict()[key],without_val.predictor.state_dict()[key])
               for key in with_val.predictor.state_dict())
    with pytest.raises(ValueError,match='replay'):
        adapt_predictor(model,new,config=cfg,old_replay=None,require_replay=True)
    with pytest.raises(ValueError,match='positive'):
        AdaptationConfig(steps=0)
    with pytest.raises(ValueError,match='fraction'):
        AdaptationConfig(replay_fraction=1.1)


def test_checkpoint_routing_never_reads_regime_labels_or_switch_time():
    model,old,new,old_test,new_test=_fixtures(seed=7)
    cfg=AdaptationConfig(steps=24,max_new_examples=32,seed=3)
    adapted,_=adapt_predictor(model,new,config=cfg)
    vote=route_by_prediction(model,adapted,new_test,window=12)
    assert vote.prechange_mse>=0 and vote.adapted_mse>=0
    assert vote.selected_checkpoint in ('prechange','adapted')
    import ecsa.experimental.jepa_adaptation as module
    src=inspect.getsource(module)
    assert 'Mechanism' not in src and 'switch_after' not in src
    assert 'collect_transitions' not in src


def test_arena_reports_skipped_alarm_as_failure_not_oracle_trigger():
    report=run_adaptation_arena(mode='smoke',seed=4,steps_override=8,device='cpu')
    assert report['protocol']=='action_jepa_recurrent_adaptation_v1'
    assert report['phases']==['additive','gated','additive']  # evaluator only
    assert report['adaptation']['triggered'] is (
        report['detection']['change_confirmed'] is True
    )
    if not report['detection']['change_confirmed']:
        assert report['adaptation']['variants']=={}
        assert report['adaptation']['new_transition_count']==0
    assert report['measurement']['pretrain']['additive']['prediction_mse']>=0
    assert report['measurement']['pretrain']['gated']['prediction_mse']>=0
    assert report['detection']['false_alarm_stationary'] in (True,False)
    assert report['phase_lengths'][0]==report['phase_lengths'][1]==report['phase_lengths'][2]
    assert 'limitations' in report


def test_arena_exposes_rehearsal_forgetting_budget_and_return_without_leakage():
    report=run_adaptation_arena(mode='smoke',seed=13,steps_override=30,device='cpu',
                                adaptation_steps=32,acquisition_budget=24)
    changed=report['adaptation']
    assert changed['new_transition_count']<=24
    assert changed['intervention_steps']>=changed['new_transition_count']
    if changed['triggered']:
        assert {'frozen','naive','replay','scratch','raw_ridge'} <= set(changed['variants'])
        for name in ('naive','replay'):
            assert changed['variants'][name]['additive']['prediction_mse']>=0
            assert changed['variants'][name]['gated']['prediction_mse']>=0
            assert changed['variants'][name]['updates']==32
            assert 'forgetting_delta' in changed['variants'][name]
            assert 'gated_gain' in changed['variants'][name]
        assert changed['checkpoint_router']['gated']['selected_checkpoint'] in ('prechange','adapted')
        assert changed['checkpoint_router']['return']['selected_checkpoint'] in ('prechange','adapted')
        assert changed['checkpoint_router']['replay_return']['selected_checkpoint'] in ('prechange','adapted')
        assert 'replay_return_monitor' in changed
        assert 'return_monitor' in changed
        return_monitor=changed['return_monitor']
        assert 'false_alarm_gated_placebo' in return_monitor
        assert return_monitor['monitor_started_after_confirmation'] is True
        assert return_monitor['alarm_index_global'] is None or return_monitor['alarm_index_global']>report['detection']['alarm_index']
        if return_monitor['alarm_index_global'] is None:
            assert changed['return_evidence'] is None, 'no return intervention without an alarm'
        else:
            assert changed['return_evidence']['independent_anchor_and_return']
            assert not set(changed['return_evidence']['reference_evidence_ids']) & set(changed['return_evidence']['confirmation_evidence_ids'])
            assert changed['return_evidence']['selected_pair']==report['detection']['causal_probe']['selected_pair']
            assert changed['return_evidence']['interpretation'] in ('decline_supported','decline_not_established')
            assert changed['return_evidence']['return_to_old_mechanism_proven'] is False
            assert changed['return_evidence']['intervention_at_alarm_epoch'] is True
        assert return_monitor['return_recognized'] is (
            return_monitor['alarm_index_global'] is not None
            and return_monitor['alarm_index_global']>=sum(report['phase_lengths'][:2])
            and changed['return_evidence'] is not None
            and changed['return_evidence']['effect_decrease_confirmed']
            and not changed['return_evidence']['supports_order_effect'])
        assert changed['replay_memory_examples']<=changed['prechange_training_count']


def test_cli_smoke_writes_json_and_reports_core_metrics(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/benchmarks/run_jepa_adaptation.py'
    output=tmp_path/'adaptation.json'
    res=subprocess.run([sys.executable,str(path),'--mode','smoke','--seed','2',
                        '--steps','8','--output',str(output)],
                       text=True,capture_output=True,timeout=90)
    assert res.returncode==0,res.stderr
    result=json.loads(output.read_text())
    assert result['protocol']=='action_jepa_recurrent_adaptation_v1'
    assert result['seed']==2
    assert 'adaptation' in result and 'measurement' in result


def test_return_scientist_only_probes_after_a_real_second_alarm():
    report=run_adaptation_arena(mode='full',seed=13,steps_override=700,
                                adaptation_steps=25,acquisition_budget=20)
    assert report['detection']['change_confirmed'] is True
    stage=report['adaptation']
    assert stage['triggered'] is True
    assert stage['reference_order_effect']['selected_pair']==report['detection']['causal_probe']['selected_pair']
    assert stage['reference_order_effect']['fresh_cohort'] is True
    monitor=stage['return_monitor']
    assert monitor['monitor_started_after_confirmation'] is True
    assert monitor['calibration_source']=='observed_transitions_at_confirmed_epoch'
    assert stage['replay_return_monitor']['calibration_source']=='observed_transitions_at_confirmed_epoch'
    if monitor['alarm_index_global'] is None:
        assert stage['return_evidence'] is None
    else:
        assert stage['return_evidence']['intervention_at_alarm_epoch'] is True
        assert monitor['return_recognized'] is (
            monitor['alarm_index_global']>=2*report['phase_lengths'][0]
            and stage['return_evidence']['effect_decrease_confirmed']
            and not stage['return_evidence']['supports_order_effect'])


def test_return_requires_evidence_of_decline_on_the_same_confirmed_action_pair():
    from ecsa.experimental.jepa_adaptation_scientist import compare_order_effect
    from ecsa.experimental.jepa_causal_scientist import collect_commutator_observations
    from ecsa.experimental.jepa_shift_world import HiddenShiftWorld
    from ecsa.experimental.action_jepa import RawObservationEncoder
    pair=(Action.PULSE,Action.FLIP)
    anchor=collect_commutator_observations(
        HiddenShiftWorld(switch_after=0,seed=124,noise_std=.16),
        *pair,trials=64,seed=125)
    same=collect_commutator_observations(
        HiddenShiftWorld(switch_after=0,seed=126,noise_std=.16),
        *pair,trials=64,seed=127)
    returned=collect_commutator_observations(
        HiddenShiftWorld(switch_after=None,seed=128,noise_std=.16),
        *pair,trials=64,seed=129)
    encoder=RawObservationEncoder()
    contrast=compare_order_effect(encoder,anchor,returned,pair=pair,permutations=2047,seed=9)
    placebo=compare_order_effect(encoder,anchor,same,pair=pair,permutations=2047,seed=9)
    assert contrast.effect_decrease_confirmed is True
    assert contrast.decline>0
    assert placebo.effect_decrease_confirmed is False
    assert not set(contrast.anchor_evidence_ids)&set(contrast.return_evidence_ids)
    with pytest.raises(ValueError,match='pair'):
        compare_order_effect(encoder,anchor,returned,pair=(Action.FLIP,Action.FLIP))
