"""Four-arm KAN hypothesis evolution versus fixed and random controls.

The environment exposes only do(X) -> observed Y. Mutations create *new,
versioned* hypotheses. Bayesian ScienceKernel updates use only observations
collected AFTER each versioned population is assembled; older experimental
outcomes can inform the new population's training, but are NEVER replayed as
independent evidence for that trained-on population. Epoch priors are an
empirical validation-score weighting, NOT a cross-generation Bayesian posterior.

All held-out scoring happens after the intervention loop, outside the policy.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import exp, isfinite, log
from pathlib import Path

import numpy as np
import torch

from ecsa.contracts import Observation, PosteriorUpdate, TheoryPosterior
from ecsa.science import ScienceKernel

from .intervention_relations import (
    InterventionWorld,
    ObservedPair,
    RelationHypothesis,
    StudyConfig,
    SyntheticInterventionWorld,
    _as_dataset,
    _fit_hypotheses,
    _grid,
    _heldout_nll,
    _knn_nll,
    _mass,
    _predictions,
    _reference_mass,
    _scores,
    _temperature,
    _unconditional_nll,
    outcome_index,
)
from .relation_discovery import HeadKind, Law, RelationModel, fit


class EvolutionArm(StrEnum):
    FIXED_RANDOM = "fixed_random"
    FIXED_EIG = "fixed_eig"
    EVOLVING_RANDOM = "evolving_random"
    EVOLVING_ADAPTIVE_EIG = "evolving_adaptive_eig"
    RETRAIN_ONLY_RANDOM = "retrain_only_random"


class MutationKind(StrEnum):
    PARAMETER = "parameter"
    STRUCTURE = "structure"
    CLONE = "clone"


class ChoiceReason(StrEnum):
    RANDOM = "random"
    EIG = "eig"
    LOW_RELIABILITY = "low_reliability"
    LOW_INFORMATION_GAIN = "low_information_gain"
    RANDOM_FLOOR = "random_floor"


@dataclass(frozen=True)
class EvolutionConfig:
    study: StudyConfig = StudyConfig()
    population_size: int = 3
    offspring_per_generation: int = 3
    mutation_interval: int = 4
    mutation_steps: int = 32
    mutation_sigma: float = 0.05
    random_floor: float = 0.20
    min_calibration_improvement: float = 0.08
    min_gain_bits: float = 0.005
    epoch_prior_scale: float = 0.30

    def __post_init__(self) -> None:
        if not isinstance(self.study, StudyConfig):
            raise TypeError("StudyConfig required")
        for field, value, minimum in (
            ("population_size", self.population_size, 2),
            ("offspring_per_generation", self.offspring_per_generation, 2),
            ("mutation_interval", self.mutation_interval, 1),
            ("mutation_steps", self.mutation_steps, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{field} must be an integer >= {minimum}")
        if self.population_size != 3:
            raise ValueError("v1 requires three hypotheses, matching the frozen bank")
        for label, value in (
            ("mutation_sigma", self.mutation_sigma),
            ("min_gain_bits", self.min_gain_bits),
            ("min_calibration_improvement", self.min_calibration_improvement),
            ("epoch_prior_scale", self.epoch_prior_scale),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and nonnegative")
        if not isfinite(self.random_floor) or not 0 <= self.random_floor < 1:
            raise ValueError("random_floor must be in [0, 1)")


@dataclass(frozen=True)
class MutationEvent:
    epoch: int
    parent_id: str
    child_id: str
    kind: MutationKind
    knots: int
    hidden: int
    accepted: bool
    calibration_nll: float


@dataclass(frozen=True)
class EvolutionTrial:
    step: int
    epoch: int
    choice: ChoiceReason
    x: tuple[float, float]
    observed_y: float
    outcome_bin: int
    nll_before_update: float
    scored_gain_bits: float
    calibration_nll: float
    calibration_reference_nll: float
    posterior_prior: tuple[tuple[str, float], ...]
    posterior_updated: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class EvolutionResult:
    law: str
    seed: int
    arm: EvolutionArm
    warmup_observations: int
    interventions: int
    heldout_observations: int
    initial_nll: float
    final_nll: float
    heldout_brier: float
    heldout_ece_top1: float
    nominal_80_mass: float
    observed_80_coverage: float
    support_clipped: int
    checkpoints_nll: tuple[tuple[int, float], ...]
    online_prequential_nll: float
    heldout_knn_nll: float
    heldout_unconditional_nll: float
    eig_choices: int
    random_choices: int
    reliability_gate_passes: int
    mutation_proposals: int
    mutation_accepted: int
    mutations_by_kind: tuple[tuple[str, int], ...]
    mutation_training_steps: int
    epoch_count: int
    final_posterior: tuple[tuple[str, float], ...]
    trials: tuple[EvolutionTrial, ...]
    mutation_events: tuple[MutationEvent, ...]


@dataclass(frozen=True)
class _Snapshot:
    count: int
    bank: tuple[RelationHypothesis, ...]
    posterior: TheoryPosterior


@dataclass(frozen=True)
class _EvolutionTrace:
    initial: _Snapshot
    snapshots: tuple[_Snapshot, ...]
    trials: tuple[EvolutionTrial, ...]
    events: tuple[MutationEvent, ...]
    gate_passes: int
    mutation_steps: int
    final: _Snapshot


def _calibration_nll(
    hypothesis: RelationHypothesis,
    validation: tuple[ObservedPair, ...],
    grid: np.ndarray,
) -> float:
    query = np.asarray([observation.x for observation in validation], dtype=np.float32)
    scores = _scores(hypothesis.model, query, grid)
    probability = _mass(scores, hypothesis.temperature, hypothesis.y_reference)
    indices = [outcome_index(grid, observation.y) for observation in validation]
    return float(-np.log(probability[np.arange(len(validation)), indices]).mean())


def _calibration_reference_nll(
    calibration: tuple[ObservedPair, ...],
    grid: np.ndarray,
    reference: np.ndarray,
) -> float:
    return float(-np.mean([
        log(max(float(reference[outcome_index(grid, row.y)]), 1e-12))
        for row in calibration
    ]))


def _mixture_calibration_nll(
    bank: tuple[RelationHypothesis, ...],
    posterior: TheoryPosterior,
    calibration: tuple[ObservedPair, ...],
    grid: np.ndarray,
) -> float:
    probabilities = np.zeros((len(calibration), len(grid)), dtype=np.float64)
    query = np.asarray([entry.x for entry in calibration], dtype=np.float32)
    for candidate in bank:
        probs = _mass(
            _scores(candidate.model, query, grid),
            candidate.temperature,
            candidate.y_reference,
        )
        probabilities += posterior.probability(candidate.hypothesis_id) * probs
    bins = [outcome_index(grid, item.y) for item in calibration]
    return float(-np.log(probabilities[np.arange(len(calibration)), bins]).mean())


def _epoch_posterior(
    bank: tuple[RelationHypothesis, ...],
    validation: tuple[ObservedPair, ...],
    grid: np.ndarray,
    scale: float,
) -> TheoryPosterior:
    """Empirical prior at a version boundary; never replay post-training data."""
    losses = np.asarray([_calibration_nll(h, validation, grid) for h in bank])
    # Normalized validation pseudo-prior, softened to retain alternative laws.
    scores = -losses * len(validation) * scale
    scores -= scores.max()
    weights = np.exp(scores)
    weights /= weights.sum()
    weights = 0.9 * weights + 0.1 / len(weights)
    return TheoryPosterior(tuple(
        (h.hypothesis_id, float(p)) for h, p in zip(bank, weights)
    ))


def _structure(model: RelationModel) -> tuple[int, int]:
    first = model.relation.first
    return (int(len(first.centers)), int(first.coefficients.shape[1]))


def _copy_spline_layer(source, destination) -> None:
    with torch.no_grad():
        left = min(source.coefficients.shape[0], destination.coefficients.shape[0])
        right = min(source.coefficients.shape[1], destination.coefficients.shape[1])
        from_knots = source.centers.detach().cpu().numpy()
        to_knots = destination.centers.detach().cpu().numpy()
        for i in range(left):
            for j in range(right):
                interpolated = np.interp(
                    to_knots, from_knots,
                    source.coefficients[i, j].detach().cpu().numpy(),
                )
                destination.coefficients[i, j].copy_(torch.as_tensor(interpolated))
        destination.residual[:left, :right].copy_(source.residual[:left, :right])
        destination.bias[:right].copy_(source.bias[:right])


def mutate_kan(
    parent: RelationHypothesis,
    *,
    kind: MutationKind,
    rng: np.random.Generator,
    sigma: float,
    child_id: str,
) -> RelationHypothesis:
    """Stochastic parameter/structural mutation, with no world-oracle access."""
    if not isinstance(kind, MutationKind):
        raise TypeError("MutationKind required")
    if not isinstance(parent.model, RelationModel) or not hasattr(parent.model.relation, "first"):
        raise TypeError("spline KAN parent required")
    if not child_id:
        raise ValueError("new hypothesis ID required")
    if kind in (MutationKind.PARAMETER, MutationKind.CLONE):
        child = copy.deepcopy(parent.model)
    else:
        knots, hidden = _structure(parent.model)
        knots = int(np.clip(knots + rng.choice([-2, 2]), 3, 11))
        hidden = int(np.clip(hidden + rng.choice([-2, 2]), 3, 16))
        with torch.random.fork_rng():
            torch.manual_seed(int(rng.integers(1, 2**31 - 1)))
            child = RelationModel(HeadKind.SPLINE_KAN, kan_knots=knots, kan_hidden=hidden)
        child.encoder.load_state_dict(parent.model.encoder.state_dict())
        _copy_spline_layer(parent.model.relation.first, child.relation.first)
        _copy_spline_layer(parent.model.relation.last, child.relation.last)
    if kind is not MutationKind.CLONE:
        with torch.no_grad():
            for parameter in child.relation.parameters():
                perturbation = rng.normal(0, sigma, parameter.shape).astype(np.float32)
                parameter.add_(torch.from_numpy(perturbation))
    child.eval()
    return RelationHypothesis(child_id, child, parent.temperature, parent.y_reference.copy())


def _evolve(
    bank: tuple[RelationHypothesis, ...],
    *,
    epoch: int,
    config: EvolutionConfig,
    warmup: tuple[ObservedPair, ...],
    observations: tuple[ObservedPair, ...],
    grid: np.ndarray,
    rng: np.random.Generator,
    mutate_enabled: bool = True,
) -> tuple[tuple[RelationHypothesis, ...], TheoryPosterior, tuple[MutationEvent, ...]]:
    """Create new version IDs and a fresh calibration prior (no old replay)."""
    calibration = warmup[-config.study.calibration:]
    # Online interventions may contribute to training; the fixed warmup
    # calibration partition never enters fit(). Neither uses heldout data.
    training = _as_dataset(warmup[:-config.study.calibration] + observations)
    reference = _reference_mass(training, grid)
    scored_parents = sorted(bank, key=lambda h: (_calibration_nll(h, calibration, grid), h.hypothesis_id))
    proposals: list[tuple[RelationHypothesis, MutationKind, str, float]] = []
    for index in range(config.offspring_per_generation):
        parent = scored_parents[index % min(2, len(scored_parents))]
        kind = (
            (MutationKind.PARAMETER if index % 2 == 0 else MutationKind.STRUCTURE)
            if mutate_enabled else MutationKind.CLONE
        )
        hypothesis = mutate_kan(
            parent, kind=kind, rng=rng, sigma=config.mutation_sigma,
            child_id=f"epoch-{epoch}-mutation-{index}",
        )
        fit(
            hypothesis.model,
            training,
            seed=int(rng.integers(0, 2**31 - 1)),
            steps=config.mutation_steps,
            lr=config.study.lr * 0.5,
        )
        hypothesis = RelationHypothesis(
            hypothesis.hypothesis_id,
            hypothesis.model,
            _temperature(hypothesis.model, _as_dataset(calibration), grid, reference),
            reference,
        )
        proposals.append((
            hypothesis,
            kind,
            parent.hypothesis_id,
            _calibration_nll(hypothesis, calibration, grid),
        ))
    # Elitism retains the strongest unmodified parent, while guaranteed
    # child admission tests the new hypothesis in each generation.
    ranked_children = sorted(proposals, key=lambda item: (item[3], item[0].hypothesis_id))
    survivors = scored_parents[:config.population_size - 1]
    survivors += [item[0] for item in ranked_children[:1]]
    new_bank = tuple(
        RelationHypothesis(
            f"epoch-{epoch}-slot-{slot}",
            h.model, h.temperature, h.y_reference,
        )
        for slot, h in enumerate(survivors)
    )
    prior = _epoch_posterior(new_bank, calibration, grid, config.epoch_prior_scale)
    accepted = {ranked_children[0][0].hypothesis_id}
    events = tuple(
        MutationEvent(
            epoch, parent_id, hypothesis.hypothesis_id, kind,
            *_structure(hypothesis.model),
            hypothesis.hypothesis_id in accepted, loss,
        )
        for hypothesis, kind, parent_id, loss in proposals
    )
    return new_bank, prior, events


def _choose(
    bank: tuple[RelationHypothesis, ...],
    posterior: TheoryPosterior,
    predictions: tuple[tuple, ...],
    *,
    arm: EvolutionArm,
    rng: np.random.Generator,
    calibration_nll: float,
    calibration_reference: float,
    config: EvolutionConfig,
) -> tuple[int, float, ChoiceReason, bool]:
    science = ScienceKernel()
    gains = np.asarray([
        science.score_experiment(posterior, candidate).information_gain_bits
        for candidate in predictions
    ])
    best = float(gains.max())
    if arm in (
        EvolutionArm.FIXED_RANDOM, EvolutionArm.EVOLVING_RANDOM,
        EvolutionArm.RETRAIN_ONLY_RANDOM,
    ):
        return int(rng.integers(len(predictions))), best, ChoiceReason.RANDOM, False
    if arm is EvolutionArm.EVOLVING_ADAPTIVE_EIG:
        reliable = (
            calibration_nll + config.min_calibration_improvement
            < calibration_reference
        )
        if not reliable:
            return int(rng.integers(len(predictions))), best, ChoiceReason.LOW_RELIABILITY, False
        if best <= config.min_gain_bits:
            return int(rng.integers(len(predictions))), best, ChoiceReason.LOW_INFORMATION_GAIN, True
        if float(rng.random()) < config.random_floor:
            return int(rng.integers(len(predictions))), best, ChoiceReason.RANDOM_FLOOR, True
    elif best <= 1e-10:
        return int(rng.integers(len(predictions))), best, ChoiceReason.LOW_INFORMATION_GAIN, False
    ties = np.flatnonzero(np.isclose(gains, best, rtol=1e-6, atol=1e-12))
    return int(rng.choice(ties)), best, ChoiceReason.EIG, True


def run_interventions(
    *,
    world: InterventionWorld,
    arm: EvolutionArm,
    initial_bank: tuple[RelationHypothesis, ...],
    grid: np.ndarray,
    warmup: tuple[ObservedPair, ...],
    candidate_pools: tuple[np.ndarray, ...],
    config: EvolutionConfig,
    rng_seed: int,
) -> _EvolutionTrace:
    """No heldout parameter: policy cannot accidentally inspect eval labels."""
    if not isinstance(arm, EvolutionArm):
        raise TypeError("EvolutionArm required")
    if not isinstance(world, InterventionWorld):
        raise TypeError("world must expose do(X)->observed Y")
    if not initial_bank or len(set(h.hypothesis_id for h in initial_bank)) != len(initial_bank):
        raise ValueError("unique initial hypotheses required")
    if not candidate_pools:
        raise ValueError("nonempty intervention budget required")
    if not all(len(pool) > 0 and pool.shape[1:] == (2,) for pool in candidate_pools):
        raise ValueError("nonempty 2D intervention candidates required")

    bank = initial_bank
    posterior = TheoryPosterior(tuple(
        (h.hypothesis_id, 1.0 / len(bank)) for h in bank
    ))
    rng = np.random.default_rng(rng_seed)
    mutation_rng = np.random.default_rng(rng_seed + 100_003)
    calibration = warmup[-config.study.calibration:]
    executed: list[ObservedPair] = []
    trials: list[EvolutionTrial] = []
    events: list[MutationEvent] = []
    snapshots: list[_Snapshot] = []
    first = _Snapshot(0, bank, posterior)
    gate_passes = 0
    epoch = 0
    mutation_steps = 0
    evolutionary = arm in (
        EvolutionArm.EVOLVING_RANDOM, EvolutionArm.EVOLVING_ADAPTIVE_EIG,
        EvolutionArm.RETRAIN_ONLY_RANDOM,
    )
    mutation_enabled = arm is not EvolutionArm.RETRAIN_ONLY_RANDOM
    if evolutionary:
        bank, posterior, new_events = _evolve(
            bank, epoch=epoch, config=config, warmup=warmup,
            observations=(), grid=grid, rng=mutation_rng,
            mutate_enabled=mutation_enabled,
        )
        events.extend(new_events)
        mutation_steps += len(new_events) * config.mutation_steps

    for step, pool in enumerate(candidate_pools):
        if evolutionary and step > 0 and step % config.mutation_interval == 0:
            epoch += 1
            bank, posterior, new_events = _evolve(
                bank, epoch=epoch, config=config, warmup=warmup,
                observations=tuple(executed), grid=grid, rng=mutation_rng,
                mutate_enabled=mutation_enabled,
            )
            events.extend(new_events)
            mutation_steps += len(new_events) * config.mutation_steps

        cal_nll = _mixture_calibration_nll(bank, posterior, calibration, grid)
        cal_ref = _calibration_reference_nll(calibration, grid, bank[0].y_reference)
        pools = np.asarray(pool, dtype=np.float32)
        ids = tuple(f"epoch-{epoch}-step-{step}-candidate-{i}" for i in range(len(pools)))
        predictions = _predictions(bank, pools, grid, ids)
        index, gain, reason, passed = _choose(
            bank, posterior, predictions, arm=arm,
            rng=rng, calibration_nll=cal_nll,
            calibration_reference=cal_ref, config=config,
        )
        gate_passes += int(passed)
        previous = posterior.probabilities
        x = tuple(float(number) for number in pools[index])
        actual_y = float(world.intervene(x))
        executed.append(ObservedPair(x, actual_y))
        observation = Observation(ids[index], (outcome_index(grid, actual_y),))
        updated = ScienceKernel().update(posterior, predictions[index], observation)
        if not isinstance(updated, PosteriorUpdate):
            raise RuntimeError("nonzero likelihood floor should prevent evidence collapse")
        posterior = updated.posterior
        trials.append(EvolutionTrial(
            step, epoch, reason, x, actual_y,
            observation.outcome[0], -log(updated.evidence_probability), gain,
            cal_nll, cal_ref, previous, posterior.probabilities,
        ))
        snapshots.append(_Snapshot(step + 1, bank, posterior))

    return _EvolutionTrace(
        first, tuple(snapshots), tuple(trials), tuple(events),
        gate_passes, mutation_steps, snapshots[-1],
    )


def _predictive_quality(
    bank: tuple[RelationHypothesis, ...],
    posterior: TheoryPosterior,
    heldout: tuple[ObservedPair, ...],
    grid: np.ndarray,
) -> tuple[float, float, float, float, int]:
    """Heldout proper Brier score, top-class ECE and 80% set coverage.

    These are diagnostics, not available to the selection policy.
    """
    query = np.asarray([point.x for point in heldout], dtype=np.float32)
    masses = np.zeros((len(heldout), len(grid)), dtype=np.float64)
    for hypothesis in bank:
        mass = _mass(
            _scores(hypothesis.model, query, grid),
            hypothesis.temperature, hypothesis.y_reference,
        )
        masses += posterior.probability(hypothesis.hypothesis_id) * mass
    indices = np.asarray([outcome_index(grid, item.y) for item in heldout])
    target = np.eye(len(grid), dtype=np.float64)[indices]
    brier = float(((masses - target) ** 2).sum(axis=1).mean())

    top = masses.argmax(axis=1)
    confidence = masses.max(axis=1)
    correct = top == indices
    ece = 0.0
    for left in np.linspace(0.0, 1.0, 11)[:-1]:
        right = left + 0.1
        mask = (confidence >= left) & ((confidence < right) if right < 1 else (confidence <= right))
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(confidence[mask].mean()) - float(correct[mask].mean())
            )

    ordered = np.argsort(-masses, axis=1)
    sorted_probs = np.take_along_axis(masses, ordered, axis=1)
    cumulative = sorted_probs.cumsum(axis=1)
    included = cumulative - sorted_probs < 0.80
    nominal = (sorted_probs * included).sum(axis=1)
    hit = (ordered == indices[:, None]) & included
    clipped = sum(int(item.y < grid[0] or item.y > grid[-1]) for item in heldout)
    return (
        brier, float(ece), float(nominal.mean()), float(hit.any(axis=1).mean()),
        clipped,
    )


def evaluate_trace(
    trace: _EvolutionTrace,
    *,
    law: Law,
    seed: int,
    arm: EvolutionArm,
    heldout: tuple[ObservedPair, ...],
    warmup: tuple[ObservedPair, ...],
    grid: np.ndarray,
) -> EvolutionResult:
    """All heldout access isolated AFTER the experimental policy returns."""
    initial = _heldout_nll(trace.initial.bank, trace.initial.posterior, grid, heldout)
    final = _heldout_nll(trace.final.bank, trace.final.posterior, grid, heldout)
    quality = _predictive_quality(
        trace.final.bank, trace.final.posterior, heldout, grid,
    )
    checkpoints = tuple(
        (snapshot.count, _heldout_nll(snapshot.bank, snapshot.posterior, grid, heldout))
        for snapshot in trace.snapshots
        if snapshot.count % 4 == 0 or snapshot.count == len(trace.snapshots)
    )
    return EvolutionResult(
        law.value, seed, arm, len(warmup), len(trace.trials), len(heldout),
        initial, final, *quality, checkpoints,
        float(np.mean([t.nll_before_update for t in trace.trials])),
        _knn_nll(warmup, grid, heldout),
        _unconditional_nll(warmup, grid, heldout),
        sum(t.choice is ChoiceReason.EIG for t in trace.trials),
        sum(t.choice is not ChoiceReason.EIG for t in trace.trials),
        trace.gate_passes,
        sum(event.kind is not MutationKind.CLONE for event in trace.events),
        sum(event.accepted and event.kind is not MutationKind.CLONE for event in trace.events),
        tuple((kind.value, sum(e.kind is kind for e in trace.events)) for kind in MutationKind),
        trace.mutation_steps,
        (max(t.epoch for t in trace.trials) + 1),
        trace.final.posterior.probabilities,
        trace.trials, trace.events,
    )


def benchmark(
    law: Law, seed: int, config: EvolutionConfig,
    *, include_retrain_control: bool = False,
    only_arms: tuple[EvolutionArm, ...] | None = None,
) -> tuple[EvolutionResult, ...]:
    """Fair world queries, warmup, candidate pools and heldout; compute tracked."""
    if not isinstance(law, Law) or type(seed) is not int:
        raise TypeError("typed law and integer seed required")
    study = config.study
    warmup_world = SyntheticInterventionWorld(law, seed=seed * 991 + 11)
    rng = np.random.default_rng(seed * 991 + 12)
    initial_x = rng.uniform(-0.9, 0.9, (study.bootstrap, 2))
    warmup = tuple(
        ObservedPair(tuple(float(v) for v in row), warmup_world.intervene(tuple(row)))
        for row in initial_x
    )
    bank, _mlp, grid = _fit_hypotheses(warmup, study, seed)

    evaluation_world = SyntheticInterventionWorld(law, seed=seed * 991 + 541)
    heldout_rng = np.random.default_rng(seed * 991 + 542)
    heldout_x = heldout_rng.uniform(-study.radius, study.radius, (study.heldout, 2))
    heldout = tuple(
        ObservedPair(tuple(float(v) for v in x), evaluation_world.intervene(tuple(x)))
        for x in heldout_x
    )
    candidate_rng = np.random.default_rng(seed * 991 + 401)
    pools = tuple(
        candidate_rng.uniform(-study.radius, study.radius, (study.candidates, 2)).astype(np.float32)
        for _ in range(study.interventions)
    )

    results = []
    arms = (
        EvolutionArm.FIXED_RANDOM, EvolutionArm.FIXED_EIG,
        EvolutionArm.EVOLVING_RANDOM, EvolutionArm.EVOLVING_ADAPTIVE_EIG,
    )
    if include_retrain_control:
        arms += (EvolutionArm.RETRAIN_ONLY_RANDOM,)
    if only_arms is not None:
        if not only_arms or any(not isinstance(arm, EvolutionArm) for arm in only_arms):
            raise ValueError("nonempty typed arm tuple required")
        arms = only_arms
    for arm in arms:
        world = SyntheticInterventionWorld(law, seed=seed * 991 + 201)
        trace = run_interventions(
            world=world, arm=arm, initial_bank=bank,
            grid=grid, warmup=warmup, candidate_pools=pools,
            config=config, rng_seed=seed * 991 + 301,
        )
        if len(world.calls) != study.interventions:
            raise AssertionError("intervention budget exceeded")
        results.append(evaluate_trace(
            trace, law=law, seed=seed, arm=arm,
            heldout=heldout, warmup=warmup, grid=grid,
        ))
    return tuple(results)


def benchmark_suite(
    *,
    seeds: tuple[int, ...],
    config: EvolutionConfig,
    include_retrain_control: bool = False,
) -> tuple[EvolutionResult, ...]:
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("nonempty integer seed tuple required")
    return tuple(
        result for law in Law for seed in seeds
        for result in benchmark(
            law, seed, config, include_retrain_control=include_retrain_control,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--laws", nargs="+", choices=[law.value for law in Law], default=[law.value for law in Law])
    parser.add_argument("--bootstrap", type=int, default=64)
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--interventions", type=int, default=12)
    parser.add_argument("--heldout", type=int, default=96)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--mutation-steps", type=int, default=32)
    parser.add_argument("--output", default="")
    parser.add_argument("--include-retrain-control", action="store_true")
    args = parser.parse_args()
    config = EvolutionConfig(
        study=StudyConfig(
            bootstrap=args.bootstrap, train_steps=args.steps,
            interventions=args.interventions, heldout=args.heldout,
            candidates=args.candidates,
        ),
        mutation_steps=args.mutation_steps,
    )
    results = tuple(
        row for law_name in args.laws for seed in args.seeds
        for row in benchmark(
            Law(law_name), seed, config,
            include_retrain_control=args.include_retrain_control,
        )
    )
    report = {
        "config": asdict(config),
        "results": [asdict(row) for row in results],
        "important_limitations": (
            "Synthetic laws; fixed Y-bin support; off-policy heldout scoring. "
            "Mutation epochs reset to an empirical calibration prior rather "
            "than a coherent across-generation Bayesian posterior. "
            "Heldout labels are never fed to intervention policy. "
            "Evolutionary arms use more training compute than fixed arms. "
            "Calibration may overfit due to repeated use across epochs."
        ),
    }
    json_output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_output + "\n", encoding="utf-8")
    aggregates = {}
    for arm in EvolutionArm:
        rows = [r for r in results if r.arm is arm]
        if not rows:
            continue
        aggregates[arm.value] = {
            "mean_final_nll": sum(r.final_nll for r in rows) / len(rows),
            "mean_initial_nll": sum(r.initial_nll for r in rows) / len(rows),
            "mean_brier": sum(r.heldout_brier for r in rows) / len(rows),
            "mean_top_class_ece": sum(r.heldout_ece_top1 for r in rows) / len(rows),
            "nominal_80_mass": sum(r.nominal_80_mass for r in rows) / len(rows),
            "observed_80_coverage": sum(r.observed_80_coverage for r in rows) / len(rows),
            "mean_eig_choices": sum(r.eig_choices for r in rows) / len(rows),
            "mean_mutations_accepted": sum(r.mutation_accepted for r in rows) / len(rows),
            "mean_extra_training_steps": sum(r.mutation_training_steps for r in rows) / len(rows),
        }
    print(json.dumps({"total": len(results), "arms": aggregates}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
