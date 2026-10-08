"""Compute-matched structure-vs-weight mutations after an unseen mechanism shift.

The scientist accesses a world only through intervene(X) -> observed Y.
Warmup follows one law; all post-warmup outcomes follow an unknown new law.
No post-shift Y is available when a candidate X is selected. Each generation
gets a new hypothesis version ID and a new calibration-based pseudo-prior;
previously used observations are NEVER replayed as independent posterior
likelihoods. Exact number of training updates is equal across primary arms.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite, log
from pathlib import Path

import numpy as np
import torch

from ecsa.contracts import Observation, PosteriorUpdate, TheoryPosterior
from ecsa.science import ScienceKernel

from .evolutionary_relations import (
    MutationKind,
    _calibration_nll,
    _copy_spline_layer,
    _epoch_posterior,
    _structure,
    mutate_kan, _predictive_quality,
)
from .intervention_relations import (
    InterventionWorld, ObservedPair, RelationHypothesis, StudyConfig,
    SyntheticInterventionWorld, _as_dataset, _fit_hypotheses, _grid,
    _heldout_nll, _mass, _predictions, _reference_mass, _scores,
    _temperature, outcome_index,
)
from .relation_discovery import Law, fit, parameters


class NewMechanism(StrEnum):
    """Only the benchmark world knows which of these governs new outcomes."""
    BILINEAR = "bilinear"
    THRESHOLD = "threshold"
    TWO_BRANCH = "two_branch"
    UNCHANGED_CONTROL = "unchanged_control"


class ShiftArm(StrEnum):
    RETRAIN = "retrain"
    GRID_GEOMETRY = "grid_geometry"
    PARAMETERS = "parameters"
    STRUCTURE = "structure"
    MIXED = "mixed"
    STRUCTURE_ADAPTIVE_EIG = "structure_adaptive_eig"


@dataclass(frozen=True)
class ShiftConfig:
    study: StudyConfig = StudyConfig(
        bootstrap=48, calibration=12, train_steps=85, interventions=24,
        candidates=16, bins=31, heldout=48,
    )
    generation_interval: int = 6
    offspring: int = 3
    steps_per_offspring: int = 18
    mutation_sigma: float = 0.05
    grid_sigma: float = 0.20
    warmup_replay: int = 12
    calibration_stride: int = 4
    reliable_gain_bits: float = 0.01
    reliable_nll_margin: float = 0.08
    random_floor: float = 0.20
    recovery_nll_delta: float = 1.00

    def __post_init__(self) -> None:
        if not isinstance(self.study, StudyConfig):
            raise TypeError("typed study required")
        for label, value, lower in (
            ("generation_interval", self.generation_interval, 2),
            ("offspring", self.offspring, 2),
            ("steps_per_offspring", self.steps_per_offspring, 1),
            ("warmup_replay", self.warmup_replay, 0),
            ("calibration_stride", self.calibration_stride, 2),
        ):
            if type(value) is not int or value < lower:
                raise ValueError(f"{label} must be >= {lower}")
        if self.warmup_replay > self.study.bootstrap - self.study.calibration:
            raise ValueError("cannot replay calibration observations")
        for label, value in (
            ("mutation_sigma", self.mutation_sigma),
            ("grid_sigma", self.grid_sigma),
            ("reliable_gain_bits", self.reliable_gain_bits),
            ("reliable_nll_margin", self.reliable_nll_margin),
            ("recovery_nll_delta", self.recovery_nll_delta),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{label} must be nonnegative finite")
        if not isfinite(self.random_floor) or not 0 <= self.random_floor < 1:
            raise ValueError("random_floor must be in [0,1)")


@dataclass(frozen=True)
class GenerationRecord:
    epoch: int
    evidence_available: int
    training_pairs: int
    calibration_pairs: int
    mutation_kinds: tuple[str, ...]
    proposed_shapes: tuple[tuple[int, int], ...]
    accepted_kind: str
    accepted_shape: tuple[int, int]
    accepted_changed_topology: bool
    accepted_changed_basis: bool
    gradient_steps: int
    parameter_steps: int
    parent_ids: tuple[str, ...]
    child_ids: tuple[str, ...]
    bank_ids: tuple[str, ...]
    bank_structures: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ShiftTrial:
    step: int
    epoch: int
    x: tuple[float, float]
    y: float
    selection: str
    gain_bits: float
    predictive_nll: float
    prior_ids: tuple[str, ...]
    posterior_ids: tuple[str, ...]
    prior: tuple[tuple[str, float], ...]
    posterior: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ShiftSnapshot:
    observations: int
    models: tuple[RelationHypothesis, ...]
    posterior: TheoryPosterior


@dataclass(frozen=True)
class ShiftTrace:
    initial: ShiftSnapshot
    checkpoints: tuple[ShiftSnapshot, ...]
    trials: tuple[ShiftTrial, ...]
    generations: tuple[GenerationRecord, ...]
    gradient_steps: int
    parameter_steps: int
    eig_choices: int
    random_choices: int
    final: ShiftSnapshot


@dataclass(frozen=True)
class ShiftResult:
    mechanism: str
    seed: int
    arm: str
    warmup_law: str
    initial_law_seen: bool
    world_interventions: int
    initial_nll: float
    final_nll: float
    improvement_nll: float
    heldout_brier: float
    heldout_top_class_ece: float
    nominal_80_mass: float
    observed_80_coverage: float
    normalized_nll_auc: float
    checkpoints_nll: tuple[tuple[int, float], ...]
    recovery_step: int | None
    prequential_nll: float
    heldout_outside_grid: int
    gradient_steps: int
    parameter_steps: int
    proposals: int
    topology_changes_proposed: int
    basis_changes_proposed: int
    basis_changes_accepted: int
    topology_changes_accepted: int
    unique_accepted_shapes: int
    eig_choices: int
    random_choices: int
    trials: tuple[ShiftTrial, ...]
    generations: tuple[GenerationRecord, ...]


class ShiftWorld:
    """Opaque post-change interventional system; policy never sees its law."""
    def __init__(self, mechanism: NewMechanism, seed: int, noise: float = 0.015) -> None:
        if not isinstance(mechanism, NewMechanism):
            raise TypeError("typed hidden mechanism required")
        self._mechanism = mechanism
        self._rng = np.random.default_rng(seed)
        self._noise = noise
        self.calls: list[ObservedPair] = []

    def intervene(self, x: tuple[float, float]) -> float:
        if len(x) != 2 or any(not isfinite(float(v)) for v in x):
            raise ValueError("finite X intervention required")
        u, v = float(x[0]), float(x[1])
        if self._mechanism is NewMechanism.UNCHANGED_CONTROL:
            y = 0.7*u - 0.45*v + 0.20*np.sin(2*u)
        elif self._mechanism is NewMechanism.BILINEAR:
            y = 0.75*u*v + 0.22*np.sin(3.0*v)
        elif self._mechanism is NewMechanism.THRESHOLD:
            y = 0.48*np.tanh(9.0*(u + 0.23*v - 0.05)) + 0.1*v
        elif self._mechanism is NewMechanism.TWO_BRANCH:
            sign = self._rng.choice([-1.0, 1.0])
            y = sign*np.sqrt(0.29+0.26*u*u+0.30*v*v)
        else:
            raise AssertionError("unsupported hidden mechanism")
        observed = float(y + self._rng.normal(0, self._noise))
        self.calls.append(ObservedPair((u, v), observed))
        return observed


def _train_and_calibration(
    warmup: tuple[ObservedPair, ...],
    recent: tuple[ObservedPair, ...],
    config: ShiftConfig,
) -> tuple[tuple[ObservedPair, ...], tuple[ObservedPair, ...]]:
    """Chronological split; no post-shift example in both fit and calibration."""
    online_calibration = recent[::config.calibration_stride]
    online_training = tuple(
        sample for i, sample in enumerate(recent)
        if i % config.calibration_stride != 0
    )
    initial_training = warmup[:-config.study.calibration]
    replay = initial_training[-config.warmup_replay:] if config.warmup_replay else ()
    training = tuple(replay) + online_training
    # Only post-shift observations rank new models. No hidden-law API use.
    return training, online_calibration


def _mutate_grid(
    parent: RelationHypothesis,
    *, rng: np.random.Generator,
    sigma: float,
    child_id: str,
) -> RelationHypothesis:
    """Change spline basis geometry with exactly the same parameter count.

    The new centers remain ordered with unchanged endpoints. No weights are
    modified here: improvements relative to CLONE can arise from changing
    spline basis geometry, not from extra gradient steps or parameter noise.
    """
    child = mutate_kan(
        parent, kind=MutationKind.CLONE, rng=rng, sigma=0., child_id=child_id,
    )
    with torch.no_grad():
        for layer in (child.model.relation.first, child.model.relation.last):
            centers = layer.centers.detach().cpu().numpy().copy()
            if len(centers) > 2:
                offsets = rng.normal(0., sigma, size=len(centers)-2)
                interior = np.sort(centers[1:-1] + offsets)
                interior = np.clip(interior, centers[0]+0.01, centers[-1]-0.01)
                layer.centers[1:-1].copy_(torch.as_tensor(interior, dtype=layer.centers.dtype))
    return child


def _spawn_generation(
    population: tuple[RelationHypothesis, ...],
    *,
    arm: ShiftArm,
    epoch: int,
    warmup: tuple[ObservedPair, ...],
    recent: tuple[ObservedPair, ...],
    grid: np.ndarray,
    config: ShiftConfig,
    mutation_seed: int,
) -> tuple[tuple[RelationHypothesis, ...], TheoryPosterior, GenerationRecord]:
    if not recent or len(recent) < config.generation_interval:
        raise ValueError("a new generation requires observable post-shift outcomes")
    training, calibration = _train_and_calibration(warmup, recent, config)
    if len(training) < 4 or len(calibration) < 2:
        raise ValueError("insufficient independent training/calibration data")
    reference = _reference_mass(_as_dataset(training), grid)
    calibration_set = _as_dataset(calibration)
    scored_parents = sorted(
        population, key=lambda h: (_calibration_nll(h, calibration, grid), h.hypothesis_id)
    )

    children: list[tuple[RelationHypothesis, str, str, tuple[int, int], float]] = []
    gradient_steps = 0
    parameter_steps = 0
    for index in range(config.offspring):
        parent = scored_parents[index % min(2, len(scored_parents))]
        kind = {
            ShiftArm.RETRAIN: MutationKind.CLONE,
            ShiftArm.GRID_GEOMETRY: MutationKind.CLONE,
            ShiftArm.PARAMETERS: MutationKind.PARAMETER,
            ShiftArm.STRUCTURE: MutationKind.STRUCTURE,
            ShiftArm.MIXED: (MutationKind.PARAMETER if index % 2 == 0 else MutationKind.STRUCTURE),
            ShiftArm.STRUCTURE_ADAPTIVE_EIG: MutationKind.STRUCTURE,
        }[arm]
        random = np.random.default_rng(mutation_seed + epoch*10_003 + index*313)
        child_id = f"{arm.value}:epoch-{epoch}:candidate-{index}"
        if arm is ShiftArm.GRID_GEOMETRY:
            child = _mutate_grid(parent, rng=random, sigma=config.grid_sigma,
                                 child_id=child_id)
            kind_name = "grid"
        else:
            child = mutate_kan(
                parent, kind=kind, rng=random,
                sigma=(config.mutation_sigma if kind is MutationKind.PARAMETER
                       or arm is ShiftArm.MIXED else 0.0),
                child_id=child_id,
            )
            kind_name = kind.value
        # Equal random optimizer schedule independent of how many random draws
        # the mutation itself requires. All variants get identical step counts.
        fit(
            child.model, _as_dataset(training),
            seed=mutation_seed + epoch*10_003 + index*313 + 202,
            steps=config.steps_per_offspring,
            lr=config.study.lr * 0.5,
        )
        gradient_steps += config.steps_per_offspring
        parameter_steps += config.steps_per_offspring * parameters(child.model)
        recalibrated = RelationHypothesis(
            child.hypothesis_id, child.model,
            _temperature(child.model, calibration_set, grid, reference),
            reference.copy(),
        )
        children.append((
            recalibrated, kind_name, parent.hypothesis_id, _structure(parent.model),
            _calibration_nll(recalibrated, calibration, grid),
        ))

    best_child = min(children, key=lambda c: (c[4], c[0].hypothesis_id))
    survivors = scored_parents[:len(population)-1] + [best_child[0]]
    next_population = tuple(
        RelationHypothesis(
            f"{arm.value}:epoch-{epoch}:slot-{i}", member.model,
            member.temperature, member.y_reference,
        )
        for i, member in enumerate(survivors)
    )
    # The new prior is empirical, not a continuation of previous posteriors.
    posterior = _epoch_posterior(next_population, calibration, grid, scale=0.15)
    record = GenerationRecord(
        epoch=epoch,
        evidence_available=len(recent),
        training_pairs=len(training),
        calibration_pairs=len(calibration),
        mutation_kinds=tuple(kind for _,kind,_,_,_ in children),
        proposed_shapes=tuple(_structure(item.model) for item,_,_,_,_ in children),
        accepted_kind=best_child[1],
        accepted_shape=_structure(best_child[0].model),
        accepted_changed_topology=_structure(best_child[0].model) != best_child[3],
        accepted_changed_basis=best_child[1] == "grid",
        gradient_steps=gradient_steps,
        parameter_steps=parameter_steps,
        parent_ids=tuple(item.hypothesis_id for item in scored_parents),
        child_ids=tuple(item.hypothesis_id for item,_,_,_,_ in children),
        bank_ids=tuple(member.hypothesis_id for member in next_population),
        bank_structures=tuple(_structure(member.model) for member in next_population),
    )
    return next_population, posterior, record


def run_shift(
    *,
    world: InterventionWorld,
    arm: ShiftArm,
    initial_bank: tuple[RelationHypothesis, ...],
    warmup: tuple[ObservedPair, ...],
    grid: np.ndarray,
    candidates: tuple[np.ndarray, ...],
    config: ShiftConfig,
    rng_seed: int,
) -> ShiftTrace:
    """The policy cannot receive held-out labels or the hidden world law."""
    if not isinstance(world, InterventionWorld):
        raise TypeError("interventional world port required")
    if not isinstance(arm, ShiftArm) or not initial_bank:
        raise TypeError("typed intervention arm and starting hypotheses required")
    if len(candidates) != config.study.interventions:
        raise ValueError("all arms must receive an equal intervention budget")
    if len({h.hypothesis_id for h in initial_bank}) != len(initial_bank):
        raise ValueError("initial hypotheses need unique IDs")
    torch.set_num_threads(1)
    random = np.random.default_rng(rng_seed)
    bank = initial_bank
    posterior = TheoryPosterior(tuple((h.hypothesis_id,1.0/len(bank)) for h in bank))
    initial = ShiftSnapshot(0, bank, posterior)
    observations: list[ObservedPair] = []
    trials: list[ShiftTrial] = []
    generations: list[GenerationRecord] = []
    snapshots: list[ShiftSnapshot] = []
    epoch = 0
    total_gradient = 0
    total_parameters = 0
    eig_choices = 0
    random_choices = 0
    for step, raw_pool in enumerate(candidates):
        if step >= config.generation_interval and step % config.generation_interval == 0:
            epoch += 1
            bank, posterior, event = _spawn_generation(
                bank, arm=arm, epoch=epoch,
                warmup=warmup, recent=tuple(observations), grid=grid,
                config=config, mutation_seed=rng_seed + 100_003,
            )
            generations.append(event)
            total_gradient += event.gradient_steps
            total_parameters += event.parameter_steps
        pool = np.asarray(raw_pool, dtype=np.float32)
        if pool.ndim != 2 or pool.shape != (config.study.candidates, 2):
            raise ValueError("bounded intervention candidate pool required")
        ids = tuple(f"epoch-{epoch}:step-{step}:candidate-{i}" for i in range(len(pool)))
        if arm is ShiftArm.STRUCTURE_ADAPTIVE_EIG:
            prospective = _predictions(bank, pool, grid, ids)
            gains = np.asarray([
                ScienceKernel().score_experiment(posterior, p).information_gain_bits
                for p in prospective
            ])
            # Only act on posterior disagreement when the independently held-out
            # online calibration improves on its unconditional reference.
            _train, calibration = _train_and_calibration(warmup, tuple(observations), config)
            reliable = False
            if len(calibration) >= 3:
                loss = np.mean([_calibration_nll(h, calibration, grid) for h in bank])
                ref = bank[0].y_reference
                reference_loss = -np.mean([
                    log(float(ref[outcome_index(grid, o.y)]) + 1e-12)
                    for o in calibration
                ])
                reliable = loss + config.reliable_nll_margin < reference_loss
            if reliable and float(gains.max()) > config.reliable_gain_bits and float(random.random()) >= config.random_floor:
                index = int(np.argmax(gains))
                reason = "eig"
                eig_choices += 1
            else:
                index = int(random.integers(len(pool)))
                reason = "random_gate"
                random_choices += 1
            gain = float(gains[index])
            selected_prediction = prospective[index]
        else:
            index = int(random.integers(len(pool)))
            reason = "random"
            gain = 0.0
            random_choices += 1
            selected_prediction = _predictions(
                bank, pool[index:index+1], grid, (ids[index],)
            )[0]
        x = tuple(float(v) for v in pool[index])
        prior = posterior.probabilities
        # The only environment call: do(X) -> unknown Y.
        observed_y = float(world.intervene(x))
        observations.append(ObservedPair(x, observed_y))
        observation = Observation(ids[index], (outcome_index(grid, observed_y),))
        update = ScienceKernel().update(posterior, selected_prediction, observation)
        if not isinstance(update, PosteriorUpdate):
            raise RuntimeError("positive likelihood floor required")
        posterior = update.posterior
        trials.append(ShiftTrial(
            step=step, epoch=epoch, x=x, y=observed_y,
            selection=reason, gain_bits=gain,
            predictive_nll=-log(update.evidence_probability),
            prior_ids=tuple(i for i,_ in prior),
            posterior_ids=tuple(i for i,_ in posterior.probabilities),
            prior=prior, posterior=posterior.probabilities,
        ))
        if (step+1) % config.generation_interval == 0 or step+1 == len(candidates):
            snapshots.append(ShiftSnapshot(step+1, bank, posterior))
    return ShiftTrace(
        initial=initial, checkpoints=tuple(snapshots), trials=tuple(trials),
        generations=tuple(generations), gradient_steps=total_gradient,
        parameter_steps=total_parameters, eig_choices=eig_choices,
        random_choices=random_choices, final=ShiftSnapshot(len(candidates),bank,posterior),
    )


def evaluate_shift(
    trace: ShiftTrace,
    *,
    mechanism: NewMechanism,
    seed: int,
    arm: ShiftArm,
    heldout: tuple[ObservedPair, ...],
    grid: np.ndarray,
    config: ShiftConfig,
) -> ShiftResult:
    """Read heldout data only AFTER the complete intervention trajectory."""
    initial_nll = _heldout_nll(
        trace.initial.models, trace.initial.posterior, grid, heldout,
    )
    checkpoints = ((0, initial_nll),) + tuple(
        (snap.observations, _heldout_nll(snap.models, snap.posterior, grid, heldout))
        for snap in trace.checkpoints
    )
    recovered = next(
        (step for step,nll in checkpoints[1:]
         if nll <= initial_nll - config.recovery_nll_delta),
        None,
    )
    final_nll = checkpoints[-1][1]
    brier, ece, nominal, coverage, _ = _predictive_quality(
        trace.final.models, trace.final.posterior, heldout, grid,
    )
    # Time-weighted held-out learning-curve metric; lower means faster recovery.
    area = float(np.trapezoid(
        [loss for _, loss in checkpoints],
        [step for step, _ in checkpoints],
    ) / max(1, len(trace.trials)))
    accepted = [record.accepted_shape for record in trace.generations]
    return ShiftResult(
        mechanism=mechanism.value, seed=seed, arm=arm.value,
        warmup_law=Law.ADDITIVE.value,
        initial_law_seen=mechanism is NewMechanism.UNCHANGED_CONTROL,
        world_interventions=len(trace.trials),
        initial_nll=initial_nll, final_nll=final_nll,
        improvement_nll=initial_nll-final_nll,
        heldout_brier=brier, heldout_top_class_ece=ece,
        nominal_80_mass=nominal, observed_80_coverage=coverage,
        normalized_nll_auc=area,
        checkpoints_nll=checkpoints, recovery_step=recovered,
        prequential_nll=float(np.mean([t.predictive_nll for t in trace.trials])),
        heldout_outside_grid=sum(int(p.y < grid[0] or p.y > grid[-1]) for p in heldout),
        gradient_steps=trace.gradient_steps, parameter_steps=trace.parameter_steps,
        proposals=sum(len(event.child_ids) for event in trace.generations),
        topology_changes_proposed=sum(
            sum(kind == MutationKind.STRUCTURE.value for kind in e.mutation_kinds)
            for e in trace.generations
        ),
        basis_changes_proposed=sum(
            sum(kind == "grid" for kind in e.mutation_kinds)
            for e in trace.generations
        ),
        basis_changes_accepted=sum(e.accepted_changed_basis for e in trace.generations),
        topology_changes_accepted=sum(e.accepted_changed_topology for e in trace.generations),
        unique_accepted_shapes=len(set(accepted)), eig_choices=trace.eig_choices,
        random_choices=trace.random_choices,
        trials=trace.trials, generations=trace.generations,
    )


def benchmark(
    *,
    mechanism: NewMechanism,
    seed: int,
    config: ShiftConfig,
    arms: tuple[ShiftArm, ...] = (
        ShiftArm.RETRAIN, ShiftArm.PARAMETERS, ShiftArm.STRUCTURE, ShiftArm.MIXED,
        ShiftArm.GRID_GEOMETRY,
    ),
) -> tuple[ShiftResult, ...]:
    if not isinstance(mechanism, NewMechanism) or type(seed) is not int:
        raise TypeError("typed novel mechanism and seed required")
    if not arms or len(set(arms)) != len(arms) or not all(isinstance(a, ShiftArm) for a in arms):
        raise ValueError("unique typed arms required")
    # Everything up to this line is shared between arms, including all
    # initial observations. There are NO post-shift responses in warmup.
    warmup_world = SyntheticInterventionWorld(Law.ADDITIVE, seed=seed*991+11)
    initial_rng = np.random.default_rng(seed*991+12)
    warmup_x = initial_rng.uniform(-0.9,0.9,(config.study.bootstrap,2))
    warmup = tuple(
        ObservedPair(tuple(float(v) for v in x),warmup_world.intervene(tuple(x)))
        for x in warmup_x
    )
    initial_bank, _, grid = _fit_hypotheses(warmup, config.study, seed)
    eval_world = ShiftWorld(mechanism, seed=seed*991+541)
    heldout_rng = np.random.default_rng(seed*991+542)
    heldout_x = heldout_rng.uniform(-config.study.radius, config.study.radius,
                                    (config.study.heldout,2))
    heldout = tuple(
        ObservedPair(tuple(float(v) for v in x), eval_world.intervene(tuple(x)))
        for x in heldout_x
    )
    candidate_rng = np.random.default_rng(seed*991+401)
    pools = tuple(candidate_rng.uniform(-config.study.radius,config.study.radius,
                                         (config.study.candidates,2)).astype(np.float32)
                  for _ in range(config.study.interventions))
    results=[]
    for arm in arms:
        world = ShiftWorld(mechanism,seed=seed*991+201)
        trace = run_shift(
            world=world, arm=arm, initial_bank=initial_bank,
            warmup=warmup,grid=grid,candidates=pools,config=config,
            rng_seed=seed*991+301,
        )
        if len(world.calls) != config.study.interventions:
            raise AssertionError("mismatched intervention budget")
        results.append(evaluate_shift(
            trace, mechanism=mechanism, seed=seed, arm=arm,
            heldout=heldout,grid=grid,config=config,
        ))
    return tuple(results)


def aggregate(results: tuple[ShiftResult, ...]) -> dict[str, object]:
    output={}
    for arm in ShiftArm:
        rows=[r for r in results if r.arm==arm.value]
        if not rows:
            continue
        output[arm.value]={
            "n":len(rows),
            "mean_initial_nll":float(np.mean([r.initial_nll for r in rows])),
            "mean_final_nll":float(np.mean([r.final_nll for r in rows])),
            "mean_nll_improvement":float(np.mean([r.improvement_nll for r in rows])),
            "mean_heldout_brier":float(np.mean([r.heldout_brier for r in rows])),
            "mean_nll_curve_auc":float(np.mean([r.normalized_nll_auc for r in rows])),
            "mean_80_coverage":float(np.mean([r.observed_80_coverage for r in rows])),
            "mean_top_class_ece":float(np.mean([r.heldout_top_class_ece for r in rows])),
            "recovery_fraction":float(np.mean([r.recovery_step is not None for r in rows])),
            "mean_gradient_steps":float(np.mean([r.gradient_steps for r in rows])),
            "mean_parameter_steps":float(np.mean([r.parameter_steps for r in rows])),
            "mean_topology_changes_accepted":float(np.mean([r.topology_changes_accepted for r in rows])),
        }
    return output


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3])
    parser.add_argument("--mechanisms",nargs="+",choices=[m.value for m in NewMechanism],
                        default=[m.value for m in NewMechanism if m is not NewMechanism.UNCHANGED_CONTROL])
    parser.add_argument("--arms",nargs="+",choices=[a.value for a in ShiftArm],default=[
        a.value for a in ShiftArm if a is not ShiftArm.STRUCTURE_ADAPTIVE_EIG
    ])
    parser.add_argument("--bootstrap",type=int,default=48)
    parser.add_argument("--steps",type=int,default=85)
    parser.add_argument("--interventions",type=int,default=24)
    parser.add_argument("--mutation-steps",type=int,default=18)
    parser.add_argument("--include-adaptive",action="store_true")
    parser.add_argument("--output",default="")
    args=parser.parse_args()
    config=ShiftConfig(
        study=StudyConfig(bootstrap=args.bootstrap,train_steps=args.steps,
                          interventions=args.interventions,candidates=16,
                          bins=31,heldout=48,calibration=12),
        steps_per_offspring=args.mutation_steps,
    )
    arms=tuple(ShiftArm(a) for a in args.arms)
    if args.include_adaptive and ShiftArm.STRUCTURE_ADAPTIVE_EIG not in arms:
        arms=(*arms,ShiftArm.STRUCTURE_ADAPTIVE_EIG)
    results=tuple(
        result for mechanism in [NewMechanism(v) for v in args.mechanisms]
        for seed in args.seeds
        for result in benchmark(mechanism=mechanism,seed=seed,config=config,arms=arms)
    )
    report={
        "config":asdict(config),"results":[asdict(r) for r in results],
        "summary":aggregate(results),
        "limitations": (
            "Mutation architectures differ in parameter counts; all arms use "
            "the same optimizer step count but not equal FLOPs. Warmup only "
            "covers additive law; post-shift mechanisms are unseen in training, "
            "not necessarily outside initial network representational capacity. "
            "Epoch priors derive from a small repeatedly used online calibration "
            "slice; not coherent cross-generation Bayesian posterior. "
            "Heldout results are evaluated only after the policy completes."
        ),
    }
    if args.output:
        file=Path(args.output)
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"summary":report["summary"],"results":len(results)},indent=2,sort_keys=True))


if __name__=="__main__":
    main()
