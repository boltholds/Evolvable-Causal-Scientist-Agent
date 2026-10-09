"""Red/green gates for evaluator-only third-law arena and GPU/CPU protocol."""
import inspect
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from ecsa.experimental.jepa_world import Action, TransitionDataset
from ecsa.experimental.jepa_router_world import (
    RouterFixtureLaw, collect_router_transitions, generate_blinded_stream,
)
from ecsa.experimental.jepa_router_arena import run_router_arena
from ecsa.experimental.jepa_mechanism_router import MechanismRouter


def test_third_law_has_distinct_predictive_transition_without_public_labels():
    trio=[collect_router_transitions(law=l,count=30,seed=19,noise_std=0) for l in RouterFixtureLaw]
    assert len(RouterFixtureLaw)==3
    assert all(isinstance(d,TransitionDataset) for d in trio)
    for data in trio:
        assert np.array_equal(data.before,trio[0].before)
        assert np.array_equal(data.actions,trio[0].actions)
    assert not np.array_equal(trio[0].after,trio[1].after)
    assert not np.array_equal(trio[0].after,trio[2].after)
    assert not np.array_equal(trio[1].after,trio[2].after)


def test_blinded_stream_observations_and_identifiers_are_disjoint_by_phase():
    seq=(RouterFixtureLaw.ADDITIVE,RouterFixtureLaw.GATED,
         RouterFixtureLaw.ADDITIVE,RouterFixtureLaw.REVERSE_GATED,
         RouterFixtureLaw.ADDITIVE)
    samples=generate_blinded_stream(laws=seq,phase_length=24,seed=9,noise_std=.12)
    assert len(samples.before)==120
    assert len(samples.actions)==120
    assert np.all(np.isfinite(samples.after))
    import ecsa.experimental.jepa_mechanism_router as router_module
    code=inspect.getsource(router_module)
    assert 'RouterFixtureLaw' not in code
    assert 'switch_at' not in code
    assert 'REVERSE_GATED' not in code


def test_arena_smoke_reports_open_set_checks_budgets_and_no_oracle_routing():
    result=run_router_arena(mode='smoke',seed=14,device='cpu')
    assert result['protocol']=='jepa_mechanism_router_v1'
    assert result['mode']=='smoke'
    assert result['source_labels_used_by_router'] is False
    assert result['new_model_auto_admitted'] is False
    assert result['initial_known_models']==2
    assert len(result['phases'])==6
    assert result['phases'][0]['true_mechanism']=='additive'
    assert result['phases'][2]['true_mechanism']=='additive'
    assert result['phases'][3]['true_mechanism']=='reverse_gated'
    assert result['phases'][5]['true_mechanism']=='reverse_gated'
    assert result['phases'][5]['expected_checkpoint']==('checkpoint-2' if result['novelty']['admitted'] else None)
    assert result['interventions']['training_new_examples']==32*len(result['novelty']['attempts'])
    assert result['interventions']['confirming_examples']==32*len(result['novelty']['attempts'])
    assert result['interventions']['total_active_actions'] == sum(
        result['interventions'][key] for key in (
            'training_new_examples','candidate_calibration_examples','confirming_examples'))
    law_sequence=('additive','gated','additive','reverse_gated','additive','reverse_gated')
    assert all(law_sequence[a['alarm_index']//60] == a['evaluator_law_at_alarm']
               for a in result['novelty']['attempts'])
    assert isinstance(result['novelty']['proposed'],bool)
    assert isinstance(result['novelty']['admitted'],bool)
    assert result['test_sets_disjoint_from_training'] is True


def test_smoke_deterministic_same_seed_and_no_mechanism_leakage():
    first=run_router_arena(mode='smoke',seed=2,device='cpu')
    second=run_router_arena(mode='smoke',seed=2,device='cpu')
    assert first['phases']==second['phases']
    assert first['novelty']==second['novelty']
    assert first['interventions']==second['interventions']


def test_invalid_arena_options_rejected():
    with pytest.raises(ValueError):run_router_arena(mode='bad',seed=0,device='cpu')
    with pytest.raises(ValueError):run_router_arena(mode='smoke',seed=-1,device='cpu')


def test_extending_fixture_horizon_does_not_rewrite_past_observations():
    first=(RouterFixtureLaw.ADDITIVE,RouterFixtureLaw.GATED)
    a=generate_blinded_stream(laws=first,phase_length=24,seed=18)
    b=generate_blinded_stream(laws=first+(RouterFixtureLaw.REVERSE_GATED,),phase_length=24,seed=18)
    assert np.array_equal(a.before,b.before[:len(a.before)])
    assert np.array_equal(a.actions,b.actions[:len(a.actions)])
    assert np.array_equal(a.after,b.after[:len(a.after)])


def test_already_admitted_law_cannot_be_counted_as_a_new_discovery():
    report=run_router_arena(mode='smoke',seed=14,device='cpu')
    attempted=report['novelty']['attempts']
    assert all('evaluator_was_novel_at_alarm' in a for a in attempted)
    assert report['novelty']['false_proposal_attempts']==sum(
        not a['evaluator_was_novel_at_alarm'] for a in attempted)
    assert report['novelty']['false_admissions']==sum(
        not a['evaluator_was_novel_at_alarm'] and a['result']['admitted']
        for a in attempted)
