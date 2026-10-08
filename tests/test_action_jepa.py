"""Offline RED/GREEN gates for action-conditioned JEPA and causal commutator arena."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import inspect
import json
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecsa.experimental.jepa_world import (
    Action, Mechanism, OpaqueSwitchWorld, collect_transitions,
)
from ecsa.experimental.action_jepa import (
    ActionJEPA, JepaConfig, train_jepa, evaluate_prediction,
    RawObservationEncoder, RandomEncoder,
)
from ecsa.experimental.jepa_causal_scientist import (
    examine_commutator, select_intervention_pair,
    collect_commutator_observations, score_commutator,
)
from ecsa.experimental.jepa_commutator_arena import run_arena


def _end(world, level, gate, *actions):
    world.reset(level=level, gate=gate)
    for action in actions:
        observation = world.step(action)
    return observation


def test_world_is_action_conditioned_and_same_start_paths_identify_noncommutativity():
    for law in Mechanism:
        world = OpaqueSwitchWorld(mechanism=law, observation_dim=24, seed=7, noise_std=0.0)
        ab = _end(world, .2, False, Action.PULSE, Action.FLIP)
        ba = _end(world, .2, False, Action.FLIP, Action.PULSE)
        gap = np.linalg.norm(ab - ba)
        if law == Mechanism.ADDITIVE:
            assert gap == pytest.approx(0.0, abs=1e-7)
        else:
            assert gap > .1
        assert world.step(Action.WAIT).shape == (24,)


def test_training_observations_contain_no_law_labels_and_reproduce_with_seed():
    a = collect_transitions(mechanism=Mechanism.GATED, count=28, observation_dim=20, seed=23)
    b = collect_transitions(mechanism=Mechanism.GATED, count=28, observation_dim=20, seed=23)
    assert np.array_equal(a.before, b.before)
    assert np.array_equal(a.after, b.after)
    assert np.array_equal(a.actions, b.actions)
    assert a.before.shape == (28,20)
    assert a.actions.shape == (28,3)
    assert a.after.shape == (28,20)
    assert set(a.__dataclass_fields__) == {"before","actions","after"}
    assert np.allclose(a.actions.sum(axis=1), 1.0)
    assert not np.array_equal(a.before, a.after)


def test_invalid_protocols_fail_closed():
    with pytest.raises(ValueError):
        JepaConfig(latent_dim=0)
    with pytest.raises(ValueError):
        JepaConfig(steps=0)
    with pytest.raises(ValueError):
        OpaqueSwitchWorld(mechanism=Mechanism.GATED, observation_dim=3)
    with pytest.raises(TypeError):
        OpaqueSwitchWorld(mechanism="gated")
    with pytest.raises(ValueError):
        collect_transitions(mechanism=Mechanism.GATED, count=0)


def test_train_one_jepa_is_finite_noncollapsed_and_frozen_after_training():
    data = collect_transitions(mechanism=Mechanism.ADDITIVE, count=128, seed=8)
    model, report = train_jepa(data, JepaConfig(steps=60, batch_size=48, seed=11), device="cpu")
    assert isinstance(model, ActionJEPA)
    assert np.isfinite(report.start_loss)
    assert np.isfinite(report.end_loss)
    assert report.train_steps == 60
    assert model.training is False
    z = model.encode(data.before)
    assert z.shape == (128, 8)
    assert np.isfinite(z).all()
    assert z.std(axis=0).mean() > .005
    assert all(param.grad is None for param in model.parameters())
    with torch.no_grad():
        p = model.predict_tensor(torch.as_tensor(z[:3]), torch.as_tensor(data.actions[:3]))
    assert p.shape == (3, 8)


def test_prediction_evaluation_and_shuffled_action_diagnostic():
    train = collect_transitions(mechanism=Mechanism.GATED, count=128, seed=4)
    heldout = collect_transitions(mechanism=Mechanism.GATED, count=64, seed=5)
    cfg = JepaConfig(steps=35, batch_size=32, seed=0)
    correct, _ = train_jepa(train, cfg, device="cpu")
    scrambled, _ = train_jepa(train, cfg, device="cpu", shuffle_actions=True)
    a = evaluate_prediction(correct, heldout)
    b = evaluate_prediction(scrambled, heldout)
    for report in (a, b):
        assert np.isfinite(report.prediction_mse)
        assert 0 <= report.action_choice_accuracy <= 1
        assert report.mean_latent_std > 0
        assert report.effective_rank >= 1
        assert report.heldout_count == 64


class IdentityEncoder:
    def encode(self, observations):
        return np.asarray(observations, dtype=np.float32)


class NoEffectEncoder:
    def encode(self, observations):
        return np.zeros((len(observations), 3), dtype=np.float32)


def test_commutator_observes_intervention_outcomes_not_simulator_law():
    for law in Mechanism:
        world = OpaqueSwitchWorld(mechanism=law, seed=16, noise_std=.01)
        result = examine_commutator(IdentityEncoder(), world, Action.PULSE, Action.FLIP,
                                   trials=24, seed=19)
        assert len(result.evidence_ids) == 24
        assert result.effect_distance >= 0
        assert result.null_distance >= 0
        assert result.prediction is (law == Mechanism.GATED)
    source = inspect.getsource(examine_commutator) + inspect.getsource(score_commutator)
    assert "_mechanism" not in source
    assert "_level" not in source
    assert "_gate" not in source


def test_controls_are_scored_on_exactly_the_same_acquired_intervention_outcomes():
    world = OpaqueSwitchWorld(mechanism=Mechanism.GATED, seed=5, noise_std=.16)
    batch = collect_commutator_observations(world,Action.PULSE,Action.FLIP,trials=16,seed=22)
    config = JepaConfig(steps=8,seed=5)
    raw = score_commutator(RawObservationEncoder(),batch,Action.PULSE,Action.FLIP)
    random = score_commutator(RandomEncoder(24,config),batch,Action.PULSE,Action.FLIP)
    assert raw.evidence_ids == random.evidence_ids == batch.evidence_ids
    assert raw.effect_distance > 0
    assert random.effect_distance > 0


def test_collapsed_representation_does_not_claim_causal_discovery():
    world = OpaqueSwitchWorld(mechanism=Mechanism.GATED, seed=22)
    result = examine_commutator(NoEffectEncoder(), world, Action.PULSE, Action.FLIP,
                               trials=12, seed=2)
    assert result.prediction is None
    assert result.effect_distance == 0


def test_active_pair_selection_uses_only_encoders_and_actions():
    train = collect_transitions(mechanism=Mechanism.GATED, count=100, seed=3)
    model,_ = train_jepa(train, JepaConfig(steps=20, seed=7), device="cpu")
    observation = OpaqueSwitchWorld(mechanism=Mechanism.GATED, seed=3).reset(level=0,gate=False)
    pair, estimates = select_intervention_pair(model, observation)
    assert pair[0] != pair[1]
    assert len(estimates) == 3
    assert set(estimates) == {"flip:pulse", "flip:wait", "pulse:wait"}
    assert np.isfinite(list(estimates.values())).all()


def test_arena_returns_provenance_and_explicit_controls_without_false_success_claims():
    report = run_arena(mode="smoke", seed=5, steps_override=20)
    assert report["protocol"] == "action_jepa_commutator_v1"
    assert report["mode"] == "smoke"
    assert len(report["mechanisms"]) == 2
    for row in report["mechanisms"]:
        assert row["mechanism"] in ("additive", "gated")
        assert row["jepa"]["trained_on_law_labels"] is False
        assert row["jepa"]["trained_on_future_observations"] is True
        assert row["jepa"]["steps"] == 20
        assert row["raw_observation_control"]["evidence_ids"]
        assert row["raw_observation_control"]["evidence_ids"] == row["scientist"]["evidence_ids"]
        assert row["active_scientist"]["evidence_ids"]
        assert row["shuffled_action_scientist_control"]["evidence_ids"]
        assert row["random_encoder_control"]["evidence_ids"]
        assert row["shuffled_action_control"]["prediction_mse"] >= 0
        assert 0 <= row["jepa_prediction"]["action_choice_accuracy"] <= 1
        assert isinstance(row["scientist"]["predicted_noncommutative"], (bool,type(None)))
    assert "success" not in report  # evidence, not a claim of autonomous science
    assert report["limitations"]


def test_cli_emits_json_report(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/benchmarks/run_action_jepa.py"
    destination = tmp_path / "result.json"
    result = subprocess.run(
        [sys.executable,str(script),"--mode","smoke","--steps","8","--output",str(destination)],
        capture_output=True,text=True,timeout=80,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(destination.read_text())["protocol"] == "action_jepa_commutator_v1"


def test_learner_and_scientist_modules_never_import_simulator_mechanism():
    from ecsa.experimental import action_jepa, jepa_causal_scientist
    assert "Mechanism" not in inspect.getsource(action_jepa)
    assert "Mechanism" not in inspect.getsource(jepa_causal_scientist)
    assert "OpaqueSwitchWorld" not in inspect.getsource(jepa_causal_scientist)


def test_noise_and_seed_are_recorded_for_reproduction():
    report = run_arena(mode="smoke", seed=9, steps_override=6, noise_std=.16)
    assert report["noise_std"] == pytest.approx(.16)
    assert report["seed"] == 9
    assert report["config"]["steps"] == 6
    assert report["train_count"] == 192 and report["heldout_count"] == 72
    assert report["trials_per_world"] == 16
