"""Blinded regime-shift and intervention verification contracts (offline)."""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecsa.experimental.jepa_world import Action, Mechanism, collect_transitions
from ecsa.experimental.action_jepa import JepaConfig, train_jepa, RawObservationEncoder
from ecsa.experimental.jepa_shift_world import HiddenShiftWorld, observe_stream
from ecsa.experimental.jepa_shift_detection import (
    RidgeDynamics, FrozenJEPAResiduals, CalibratedShiftMonitor, ObservedActionDeltaTemplate,
)
from ecsa.experimental.jepa_shift_scientist import (
    confirm_order_effect, discover_order_hypothesis, score_order_effect, discover_from_pilot_batches,
)
from ecsa.experimental.jepa_shift_arena import run_shift_arena, _laboratory_at_alarm, periodic_symbolic_baseline


def test_world_switches_without_resetting_physics_or_exposing_regime_label():
    world = HiddenShiftWorld(switch_after=2, seed=8, noise_std=0)
    world.reset(level=.1, gate=False)
    start = world.step(Action.WAIT)
    assert np.allclose(start, world.step(Action.WAIT))
    after = world.step(Action.PULSE)  # now gated, gate still false
    assert np.array_equal(start, after)
    world.step(Action.FLIP)
    assert not np.array_equal(after, world.step(Action.PULSE))
    assert not hasattr(world, "public_mechanism")


def test_null_world_never_switches_and_stream_has_observational_only_contract():
    null = HiddenShiftWorld(switch_after=None, seed=4, noise_std=.1)
    data = observe_stream(null, count=30, seed=2)
    assert data.before.shape == data.after.shape == (30,24)
    assert data.actions.shape == (30,3)
    assert set(data.__dataclass_fields__) == {"before","actions","after"}
    assert not np.array_equal(data.before, data.after)
    repeated = observe_stream(HiddenShiftWorld(switch_after=None,seed=4,noise_std=.1),count=30,seed=2)
    assert np.array_equal(data.before,repeated.before)


def test_ridge_raw_and_random_control_predict_only_observed_next_state():
    train=collect_transitions(mechanism=Mechanism.ADDITIVE,count=192,seed=37,noise_std=.1)
    cal=collect_transitions(mechanism=Mechanism.ADDITIVE,count=48,seed=38,noise_std=.1)
    ridge=RidgeDynamics.fit(train, RawObservationEncoder(), regularization=1e-2)
    errors=ridge.errors(cal)
    assert errors.shape == (48,)
    assert np.isfinite(errors).all()
    assert float(errors.mean()) < .3
    with pytest.raises(ValueError,match="dimension"):
        ridge.errors(collect_transitions(mechanism=Mechanism.ADDITIVE,count=8,observation_dim=32))



def test_symbolic_observation_delta_template_is_not_an_oracle():
    train=collect_transitions(mechanism=Mechanism.ADDITIVE,count=150,seed=31,noise_std=.12)
    held=collect_transitions(mechanism=Mechanism.GATED,count=60,seed=32,noise_std=.12)
    symbolic=ObservedActionDeltaTemplate.fit(train)
    values=symbolic.errors(held)
    assert values.shape == (60,)
    assert np.isfinite(values).all()
    assert values.mean() >= 0


def test_shared_pilot_and_confirmation_banks_keep_selection_independent():
    from ecsa.experimental.jepa_causal_scientist import collect_commutator_observations
    from itertools import combinations
    world=HiddenShiftWorld(switch_after=0,seed=17,noise_std=.16)
    pairs=tuple(combinations(tuple(Action),2))
    pilots={pair:collect_commutator_observations(world,*pair,trials=12,seed=900+i)
            for i,pair in enumerate(pairs)}
    confirms={pair:collect_commutator_observations(world,*pair,trials=64,seed=1900+i)
              for i,pair in enumerate(pairs)}
    report=discover_from_pilot_batches(RawObservationEncoder(),pilots,confirms,
                                        permutations=1023,seed=4)
    assert report.selected_pair == (Action.PULSE,Action.FLIP)
    assert report.confirmation.supports_order_effect
    assert len(report.pilot_evidence_ids)==36
    assert not set(report.pilot_evidence_ids)&set(report.confirmation.evidence_ids)


def test_frozen_jepa_errors_and_monitor_calibration_do_not_use_shift_truth():
    training=collect_transitions(mechanism=Mechanism.ADDITIVE,count=144,seed=11,noise_std=.1)
    cal=collect_transitions(mechanism=Mechanism.ADDITIVE,count=128,seed=12,noise_std=.1)
    model,_ =train_jepa(training,JepaConfig(steps=12,seed=1),device="cpu")
    scorer=FrozenJEPAResiduals(model)
    errors=scorer.errors(cal)
    assert errors.shape == (128,)
    assert np.isfinite(errors).all()
    actions=np.eye(3,dtype=np.float32)[np.arange(128)%3]
    calibrator=CalibratedShiftMonitor(errors,cal.actions,actions,horizon=128,window=12,bootstrap=256,seed=23)
    null=calibrator.detect(errors,cal.actions)
    assert null.threshold>0
    assert null.peak_score>=0
    assert null.alarm_index is None or null.alarm_index>=11
    assert not model.training


def test_monitor_detects_synthetic_shift_and_accounts_for_look_elsewhere():
    rng=np.random.default_rng(61)
    actions=np.eye(3,dtype=np.float32)[np.arange(160)%3]
    cal_actions=np.tile(np.eye(3,dtype=np.float32),(150,1))
    cal_errors=np.maximum(.005, rng.normal(.1,.025,450)).astype(np.float64)
    monitor=CalibratedShiftMonitor(cal_errors,cal_actions,actions,horizon=160,window=14,bootstrap=512,seed=3)
    null=np.maximum(.005,rng.normal(.1,.025,160))
    shifted=null.copy()
    shifted[80:]+= .11
    assert monitor.detect(shifted,actions).alarm_index is not None
    assert monitor.detect(shifted,actions).alarm_index>=80
    assert monitor.detect(shifted,actions).alarm_index<105
    assert monitor.detect(null,actions).alarm_index is None
    changed_future=shifted.copy()
    changed_future[125:]+= 100
    assert monitor.detect(changed_future,actions).alarm_index==monitor.detect(shifted,actions).alarm_index
    assert monitor.detect(changed_future,actions).peak_score==monitor.detect(shifted,actions).peak_score
    with pytest.raises(ValueError,match="horizon"):
        monitor.detect(null[:-1],actions[:-1])


def test_paired_order_effect_is_significant_for_gated_and_null_for_additive():
    for mechanism in Mechanism:
        world=HiddenShiftWorld(switch_after=0 if mechanism is Mechanism.GATED else None,
                               seed=21,noise_std=.16)
        result=confirm_order_effect(RawObservationEncoder(), world, (Action.PULSE,Action.FLIP),
                                    trials=64,seed=33,permutations=1023,alpha=.01)
        assert result.p_value>0 and result.p_value<=1
        assert len(result.evidence_ids)==64
        assert result.supports_order_effect is (mechanism is Mechanism.GATED)
        assert result.positive_effect >= -10


def test_discovery_selects_action_pair_from_pilot_and_confirms_independently():
    world=HiddenShiftWorld(switch_after=0,seed=18,noise_std=.16)
    result=discover_order_hypothesis(RawObservationEncoder(),world,
                                     pilot_trials=12,confirmation_trials=64,
                                     permutations=1023,seed=2)
    assert result.hypotheses == ("order_independence", "order_dependence")
    assert len(result.pilot_evidence_ids)==36
    assert len(result.confirmation.evidence_ids)==64
    assert not set(result.pilot_evidence_ids)&set(result.confirmation.evidence_ids)
    assert len(result.pilot_scores)==3
    assert result.selected_pair in ((Action.PULSE,Action.FLIP),(Action.FLIP,Action.PULSE),
                                    (Action.PULSE,Action.WAIT),(Action.FLIP,Action.WAIT))
    assert result.confirmation.supports_order_effect


def test_scientist_does_not_import_fixture_mechanism_or_ground_truth():
    import ecsa.experimental.jepa_shift_detection as detect
    import ecsa.experimental.jepa_shift_scientist as scientist
    for source in (inspect.getsource(detect),inspect.getsource(scientist)):
        assert "Mechanism" not in source
        assert "_mechanism" not in source
        assert "switch_after" not in source
        assert "OpaqueSwitchWorld" not in source



def test_confirmatory_laboratory_freezes_regime_at_alert_not_at_end_of_stream():
    additive=_laboratory_at_alarm(alarm_index=20,change_after=48,seed=9,noise_std=.16)
    gated=_laboratory_at_alarm(alarm_index=60,change_after=48,seed=9,noise_std=.16)
    for lab,expected in ((additive,False),(gated,True)):
        evidence=confirm_order_effect(RawObservationEncoder(),lab,
                         (Action.PULSE,Action.FLIP),trials=64,seed=51,
                         permutations=1023,alpha=.01)
        assert evidence.supports_order_effect is expected



def test_periodic_symbolic_baseline_probes_before_and_after_without_oracle_pair():
    report=periodic_symbolic_baseline(
        seed=8, noise_std=.16, horizon=64,change_after=32,
        interval=16,pilot_trials=10,confirmation_trials=48,
        permutations=2047,alpha=.02,
    )
    assert report["method"]=="periodic_raw_intervention_permutation"
    assert report["intervention_steps"]>0
    assert report["null_intervention_steps"]>0
    assert report["alarm_index"] is not None and report["alarm_index"] >= 32
    assert report["confirmed_order_effect"] is True
    assert report["null_false_alarm"] is False
    assert report["null_confirmed_order_effect"] is False
    assert report["selected_pair"] == ["pulse","flip"]


def test_shift_arena_emits_separated_stream_alerts_and_confirmations():
    result=run_shift_arena(mode="smoke",seed=5,steps_override=24)
    assert result["protocol"] == "action_jepa_hidden_shift_v1"
    assert result["train_count"]>=128
    assert result["change_after"]==result["prechange_steps"]
    assert set(result["controls"]) == {"jepa","shuffled_actions","random_ridge","raw_ridge","delta_template","symbolic"}
    assert result["null_stream"]["mechanism_changes"] == 0
    for mode in ("jepa","shuffled_actions","random_ridge","raw_ridge"):
        stage=result["controls"][mode]
        assert "alarm_index" in stage and "null_alarm_index" in stage
        assert stage["calibration_count"]>0
        assert "null_confirmed_order_effect" in stage
        if stage["null_alarm_index"] is not None:
            assert stage["null_confirmed_order_effect"] in (True,False)
        if stage["alarm_index"] is not None:
            assert stage["hypotheses"] == ["order_independence","order_dependence"]
            assert not set(stage["pilot_evidence_ids"])&set(stage["confirmation_evidence_ids"])
    symbolic=result["controls"]["symbolic"]
    assert symbolic["method"]=="periodic_raw_intervention_permutation"
    assert symbolic["intervention_steps"]>0
    assert symbolic["null_intervention_steps"]>0
    assert "limitations" in result
    assert "is_causal_proof" not in result


def test_shift_cli_writes_report(tmp_path):
    script=Path(__file__).resolve().parents[1]/"scripts/benchmarks/run_jepa_shift.py"
    output=tmp_path/"result.json"
    process=subprocess.run([sys.executable,str(script),"--mode","smoke","--steps","8",
                            "--seed","1","--output",str(output)],text=True,capture_output=True,timeout=100)
    assert process.returncode==0,process.stderr
    data=json.loads(output.read_text())
    assert data["protocol"]=="action_jepa_hidden_shift_v1"
