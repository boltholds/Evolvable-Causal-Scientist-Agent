"""Graded benchmark orchestration: law names are evaluator-only truth."""
from __future__ import annotations

from dataclasses import asdict

from .action_jepa import (
    JepaConfig, train_jepa, evaluate_prediction,
    RawObservationEncoder, RandomEncoder,
)
from .jepa_world import Action, Mechanism, OpaqueSwitchWorld, collect_transitions
from .jepa_causal_scientist import (
    collect_commutator_observations, score_commutator, select_intervention_pair,
    _scientist_dict,
)

def run_arena(
    *, mode: str = "smoke", seed: int = 0,
    device: str = "cpu", steps_override: int | None = None,
    noise_std: float = .16,
) -> dict[str, object]:
    """Run JEPA plus shuffled-action/raw/random controls for both mechanisms.

    Never reports fixture grading as proof of general causal identification.
    """
    if mode not in ("smoke", "full"):
        raise ValueError("mode must be smoke or full")
    steps = steps_override if steps_override is not None else (70 if mode == "smoke" else 700)
    cfg = JepaConfig(steps=steps, seed=seed)
    train_size, heldout_size, trials = (192, 72, 16) if mode == "smoke" else (2048, 256, 64)
    results: list[dict[str, object]] = []
    for mechanism in Mechanism:
        # Sensor geometry is held fixed across train/test; observations and
        # initial states are independent by non-overlapping trial seeds.
        train = collect_transitions(
            mechanism=mechanism, count=train_size,
            seed=100 + seed, noise_std=noise_std,
        )
        heldout = collect_transitions(
            mechanism=mechanism, count=heldout_size,
            seed=10_000 + seed, noise_std=noise_std,
        )
        learned, train_report = train_jepa(train, cfg, device=device)
        shuffled, shuffle_report = train_jepa(
            train, cfg, device=device, shuffle_actions=True,
        )
        pred = evaluate_prediction(learned, heldout)
        shuffled_pred = evaluate_prediction(shuffled, heldout)
        lab = OpaqueSwitchWorld(
            mechanism=mechanism, seed=14_003 + seed, noise_std=noise_std,
        )
        initial = lab.reset(trial_seed=21_000 + seed)
        proposed, estimates = select_intervention_pair(learned, initial)
        # Primary preregistered causal test targets the pulse/flip algebra.
        pair = (Action.PULSE, Action.FLIP)
        observations = collect_commutator_observations(
            lab, *pair, trials=trials, seed=seed,
        )
        scientist = score_commutator(learned, observations, *pair)
        raw = score_commutator(RawObservationEncoder(), observations, *pair)
        random = score_commutator(RandomEncoder(24, cfg), observations, *pair)
        scrambled = score_commutator(shuffled, observations, *pair)
        active_batch = (observations if set(proposed) == set(pair) else
                        collect_commutator_observations(
                            lab, *proposed, trials=trials, seed=seed + 41_000,
                        ))
        active = score_commutator(learned, active_batch, *proposed)
        results.append({
            "mechanism": mechanism.value,  # evaluator only; never a training input
            "jepa": {
                **asdict(train_report), "steps": cfg.steps,
                "trained_on_law_labels": False,
                "trained_on_future_observations": True,
            },
            "jepa_prediction": asdict(pred),
            "shuffled_action_control": {
                **asdict(shuffle_report),**asdict(shuffled_pred),
            },
            "scientist": _scientist_dict(scientist),
            "raw_observation_control": _scientist_dict(raw),
            "random_encoder_control": _scientist_dict(random),
            "shuffled_action_scientist_control": _scientist_dict(scrambled),
            "active_scientist": _scientist_dict(active),
            "proposed_active_pair": [v.value for v in proposed],
            "predicted_commutator_scores": estimates,
            "fixture_expected_noncommuting": mechanism is Mechanism.GATED,
            "graded_match": scientist.prediction is (mechanism is Mechanism.GATED),
        })
    return {
        "protocol": "action_jepa_commutator_v1",
        "mode": mode,
        "seed": seed,
        "device": device,
        "config": asdict(cfg),
        "train_count": train_size,
        "heldout_count": heldout_size,
        "trials_per_world": trials,
        "noise_std": noise_std,
        "mechanisms": results,
        "limitations": [
            "Small synthetic two-mechanism simulator with known actions; not generic causal discovery.",
            "A separate JEPA is trained for each environment mechanism; no cross-mechanism adaptation tested.",
            "Actor sees action tokens and observations, but no semantic world variables or law labels.",
            "EMA target plus variance/covariance penalties are JEPA-inspired, not faithful LeWorldModel SIGReg.",
            "A commutator test detects interaction of two interventions, not a unique physical causal graph.",
            "Raw and random encoder controls are essential; commutator may be visible without representation learning.",
            "Training and heldout runs share sensor geometry but not sampled states; no distribution shift claim.",
            "Active pair proposal is separately tested; preregistered pulse/flip determines primary grading.",
            "All representation controls score the EXACT same acquired commutator observations.",
            "The 2.5x noise decision threshold is a predefined heuristic, not a calibrated statistical test.",
        ],
    }
