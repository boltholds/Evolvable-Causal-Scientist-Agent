"""Closed-loop intervention-only X -> observed Y mechanism discovery.

The scientist receives candidate X values and a world port exposing
intervene(X) -> observed Y. Independent encoders E_x and E_y feed
competing spline-KAN or MLP implicit-relation hypotheses. The existing
ScienceKernel selects experiments and sequentially updates model posterior
from observed (quantized) Y -- no pair-membership oracle.

Neural hypotheses are FROZEN during posterior updates. Updating their
parameters would change the theory and require replay/recalibration.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite, log, log2
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from ecsa.contracts import (
    Observation,
    PosteriorUpdate,
    PredictiveDistribution,
    TheoryPosterior,
)
from ecsa.science import ScienceKernel

from .relation_discovery import (
    Dataset,
    HeadKind,
    Law,
    RelationModel,
    _physical_y,
    _tensor,
    _x_view,
    _y_view,
    fit,
)


@runtime_checkable
class InterventionWorld(Protocol):
    """The ONLY world method visible to the experiment loop."""
    def intervene(self, x: tuple[float, float]) -> float: ...


class SamplingPolicy(StrEnum):
    EIG = "eig"
    RANDOM = "random"


@dataclass(frozen=True)
class StudyConfig:
    bootstrap: int = 64
    calibration: int = 16
    train_steps: int = 140
    interventions: int = 12
    candidates: int = 24
    bins: int = 41
    heldout: int = 96
    radius: float = 1.05
    lr: float = 0.012

    def __post_init__(self) -> None:
        for name in (
            "bootstrap", "calibration", "train_steps",
            "interventions", "candidates", "bins", "heldout",
        ):
            value = getattr(self, name)
            minimum = 8 if name in ("bootstrap", "calibration") else (
                3 if name == "bins" else 1
            )
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be a sufficiently positive integer")
        if self.calibration >= self.bootstrap // 2:
            raise ValueError("calibration must be less than half of bootstrap")
        if not isfinite(self.lr) or not 0.0 < self.lr < 0.1:
            raise ValueError("invalid learning rate")
        if not isfinite(self.radius) or not 0.3 < self.radius <= 2.0:
            raise ValueError("invalid intervention radius")


@dataclass(frozen=True)
class ObservedPair:
    x: tuple[float, float]
    y: float

    def __post_init__(self) -> None:
        if len(self.x) != 2 or any(not isfinite(v) for v in (*self.x, self.y)):
            raise ValueError("finite two-dimensional X and Y required")


class SyntheticInterventionWorld:
    """Hidden-law benchmark oracle: only an intervention result is exposed."""

    def __init__(self, law: Law, *, seed: int, noise: float = 0.015) -> None:
        self._law = law
        self._rng = np.random.default_rng(seed)
        self._noise = noise
        self.calls: list[ObservedPair] = []

    def intervene(self, x: tuple[float, float]) -> float:
        physical_x = np.asarray([x], dtype=np.float32)
        if physical_x.shape != (1, 2) or not np.isfinite(physical_x).all():
            raise ValueError("finite two-dimensional intervention required")
        y = float(
            _physical_y(self._law, physical_x, self._rng)[0]
            + self._rng.normal(0.0, self._noise)
        )
        self.calls.append(ObservedPair(x, y))
        return y


@dataclass(frozen=True)
class RelationHypothesis:
    hypothesis_id: str
    model: RelationModel
    temperature: float
    y_reference: np.ndarray


@dataclass(frozen=True)
class Trial:
    step: int
    policy: SamplingPolicy
    x: tuple[float, float]
    y_observed: float
    outcome_bin: int
    prequential_nll: float
    information_gain_bits: float
    prior: tuple[tuple[str, float], ...]
    posterior: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class StudyResult:
    law: str
    seed: int
    model_family: str
    policy: str
    warmup_observations: int
    interventions: int
    heldout: int
    initial_nll: float
    final_nll: float
    mlp_nll: float
    knn_nll: float
    unconditional_nll: float
    prequential_nll: float
    avg_information_gain_bits: float
    posterior_entropy_bits: float
    support_clipped: int
    posterior: tuple[tuple[str, float], ...]
    trials: tuple[Trial, ...]


def _as_dataset(observations: tuple[ObservedPair, ...]) -> Dataset:
    x = np.asarray([pair.x for pair in observations], dtype=np.float32)
    y = np.asarray([pair.y for pair in observations], dtype=np.float32)
    return Dataset(x, y, _x_view(x), _y_view(y))


def _grid(observations: tuple[ObservedPair, ...], bins: int) -> np.ndarray:
    y = np.asarray([pair.y for pair in observations])
    expansion = max(float(y.std()) * 1.5, 0.35)
    return np.linspace(
        float(y.min()) - expansion,
        float(y.max()) + expansion,
        bins,
        dtype=np.float32,
    )


def outcome_index(grid: np.ndarray, y: float) -> int:
    if not isfinite(y):
        raise ValueError("nonfinite experimental outcome")
    boundaries = (grid[1:] + grid[:-1]) / 2
    return int(np.clip(np.searchsorted(boundaries, y), 0, len(grid) - 1))


@torch.no_grad()
def _scores(model: RelationModel, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate hypothetical Y only INSIDE a model, never via the world."""
    x_view = _x_view(np.repeat(x.astype(np.float32), len(grid), axis=0))
    y_view = _y_view(np.tile(grid, len(x)))
    parts: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(x_view), 2048):
        logits = model(
            _tensor(x_view[start:start + 2048]),
            _tensor(y_view[start:start + 2048]),
        )
        parts.append(logits.detach().cpu().numpy())
    return np.concatenate(parts).reshape(len(x), len(grid)).astype(np.float64)


def _mass(
    logits: np.ndarray,
    temperature: float,
    reference: np.ndarray,
    floor: float = 0.01,
) -> np.ndarray:
    """Normalized p(Y_bin | do(X), hypothesis).

    Matched-pair-vs-shuffled-Y classification estimates a density RATIO,
    approximately p(Y|X)/q(Y); q(Y) must be multiplied back. Temperature
    is tuned on warmup-only calibration observations.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)
    exp_scores = np.exp(np.maximum(scaled, -700)) * reference
    probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    n_bins = probs.shape[-1]
    return (1.0 - floor) * probs + floor / n_bins


def _reference_mass(train: Dataset, grid: np.ndarray) -> np.ndarray:
    """Empirical training-Y marginal q(Y), estimated without the world law."""
    observed = train.y.astype(np.float64)
    bandwidth = max(0.06, 0.9 * float(observed.std()) * len(observed) ** (-0.2))
    reference = np.exp(
        -0.5 * ((grid[None, :] - observed[:, None]) / bandwidth) ** 2
    ).sum(axis=0)
    reference += 1e-4
    return reference / reference.sum()


def _temperature(
    model: RelationModel,
    validation: Dataset,
    grid: np.ndarray,
    reference: np.ndarray,
) -> float:
    logits = _scores(model, validation.x, grid)
    bins = np.asarray([outcome_index(grid, float(y)) for y in validation.y])
    options = (0.07, 0.12, 0.2, 0.35, 0.55, 0.8, 1.2, 2.0, 3.0)
    return min(
        options,
        key=lambda temperature: float(
            -np.log(
                _mass(logits, temperature, reference)[np.arange(len(bins)), bins]
            ).mean()
        ),
    )


def _fit_hypotheses(
    observations: tuple[ObservedPair, ...],
    config: StudyConfig,
    seed: int,
) -> tuple[tuple[RelationHypothesis, ...], tuple[RelationHypothesis, ...], np.ndarray]:
    cut = len(observations) - config.calibration
    training = _as_dataset(observations[:cut])
    validation = _as_dataset(observations[cut:])
    grid = _grid(observations, config.bins)
    reference = _reference_mass(training, grid)
    torch.set_num_threads(1)

    kan: list[RelationHypothesis] = []
    variants = ((5, 12), (7, 9), (9, 7))
    for index, (knots, hidden) in enumerate(variants):
        torch.manual_seed(seed * 1000 + 7 + index)
        model = RelationModel(HeadKind.SPLINE_KAN, kan_knots=knots, kan_hidden=hidden)
        fit(
            model, training,
            seed=seed * 1000 + 17 + index,
            steps=config.train_steps,
            lr=config.lr,
        )
        kan.append(RelationHypothesis(
            f"kan-{index}",
            model,
            _temperature(model, validation, grid, reference),
            reference,
        ))

    mlp: list[RelationHypothesis] = []
    for index in range(3):
        torch.manual_seed(seed * 1000 + 7 + index)
        model = RelationModel(HeadKind.MLP)
        fit(
            model, training,
            seed=seed * 1000 + 17 + index,
            steps=config.train_steps,
            lr=config.lr,
        )
        mlp.append(RelationHypothesis(
            f"mlp-{index}",
            model,
            _temperature(model, validation, grid, reference),
            reference,
        ))
    return tuple(kan), tuple(mlp), grid


def _ensemble_mass(
    hypotheses: tuple[RelationHypothesis, ...],
    posterior: TheoryPosterior,
    x: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(x), len(grid)), dtype=np.float64)
    for hypothesis in hypotheses:
        likelihood = _mass(
            _scores(hypothesis.model, x, grid),
            hypothesis.temperature,
            hypothesis.y_reference,
        )
        result += posterior.probability(hypothesis.hypothesis_id) * likelihood
    return result


def _predictions(
    hypotheses: tuple[RelationHypothesis, ...],
    x: np.ndarray,
    grid: np.ndarray,
    experiment_ids: tuple[str, ...],
) -> tuple[tuple[PredictiveDistribution, ...], ...]:
    if len(x) != len(experiment_ids):
        raise ValueError("candidate count mismatch")
    predicted = [
        _mass(_scores(h.model, x, grid), h.temperature, h.y_reference)
        for h in hypotheses
    ]
    return tuple(
        tuple(
            PredictiveDistribution(
                theory_id=hypothesis.hypothesis_id,
                experiment_id=experiment_ids[i],
                probabilities=tuple(
                    ((j,), float(mass[i, j])) for j in range(len(grid))
                ),
            )
            for hypothesis, mass in zip(hypotheses, predicted)
        )
        for i in range(len(x))
    )


def _heldout_nll(
    hypotheses: tuple[RelationHypothesis, ...],
    posterior: TheoryPosterior,
    grid: np.ndarray,
    observations: tuple[ObservedPair, ...],
) -> float:
    x = np.asarray([p.x for p in observations], dtype=np.float32)
    mass = _ensemble_mass(hypotheses, posterior, x, grid)
    indices = [outcome_index(grid, p.y) for p in observations]
    return float(-np.log(mass[np.arange(len(observations)), indices]).mean())


def _unconditional_nll(
    observations: tuple[ObservedPair, ...],
    grid: np.ndarray,
    heldout: tuple[ObservedPair, ...],
) -> float:
    counts = np.ones(len(grid), dtype=np.float64)
    for observation in observations:
        counts[outcome_index(grid, observation.y)] += 1
    mass = counts / counts.sum()
    return float(-np.log([
        mass[outcome_index(grid, observation.y)] for observation in heldout
    ]).mean())


def _knn_nll(
    observations: tuple[ObservedPair, ...],
    grid: np.ndarray,
    heldout: tuple[ObservedPair, ...],
) -> float:
    x = np.asarray([p.x for p in observations], dtype=np.float64)
    y = np.asarray([p.y for p in observations], dtype=np.float64)
    query_x = np.asarray([p.x for p in heldout], dtype=np.float64)
    distance = ((query_x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    weights = np.exp(-distance / (2.0 * 0.30 ** 2)) + 0.00001
    # Observational conditional KDE; no oracle or intervention tuning.
    kernel = np.exp(-((grid[None, :] - y[:, None]) ** 2) / (2.0 * 0.12 ** 2))
    kernel /= kernel.sum(axis=1, keepdims=True)
    mass = weights @ kernel / weights.sum(axis=1, keepdims=True)
    mass = 0.99 * mass + 0.01 / len(grid)
    indices = [outcome_index(grid, p.y) for p in heldout]
    return float(-np.log(mass[np.arange(len(heldout)), indices]).mean())


def posterior_entropy(posterior: TheoryPosterior) -> float:
    return -sum(p * log2(p) for _, p in posterior.probabilities if p > 0.0)


def _select(
    science: ScienceKernel,
    posterior: TheoryPosterior,
    predictions: tuple[tuple[PredictiveDistribution, ...], ...],
    rng: np.random.Generator,
    policy: SamplingPolicy,
) -> tuple[int, float]:
    if policy is SamplingPolicy.RANDOM:
        index = int(rng.integers(0, len(predictions)))
        return (
            index,
            science.score_experiment(
                posterior, predictions[index],
            ).information_gain_bits,
        )
    scores = np.asarray([
        science.score_experiment(posterior, item).information_gain_bits
        for item in predictions
    ])
    best = float(scores.max())
    if best < 1e-10:
        return int(rng.integers(0, len(predictions))), best
    ties = np.flatnonzero(np.isclose(scores, best, rtol=1e-6, atol=1e-12))
    return int(rng.choice(ties)), best


def run_closed_loop(
    *,
    world: InterventionWorld,
    policy: SamplingPolicy,
    hypotheses: tuple[RelationHypothesis, ...],
    grid: np.ndarray,
    candidates: tuple[np.ndarray, ...],
    rng_seed: int,
    heldout: tuple[ObservedPair, ...],
    warmup: tuple[ObservedPair, ...],
    mlp: RelationHypothesis,
    law_label: str = "unknown",
    seed: int = 0,
    model_family: str = "unspecified",
) -> StudyResult:
    """Choose do(X), then observe its Y, then update ScienceKernel posterior."""
    if not isinstance(policy, SamplingPolicy):
        raise TypeError("SamplingPolicy required")
    if not hypotheses or any(
        not isinstance(h, RelationHypothesis) for h in hypotheses
    ):
        raise TypeError("typed model hypotheses required")
    if len({h.hypothesis_id for h in hypotheses}) != len(hypotheses):
        raise ValueError("unique hypothesis identifiers required")

    science = ScienceKernel()
    posterior = TheoryPosterior(tuple(
        (h.hypothesis_id, 1.0 / len(hypotheses)) for h in hypotheses
    ))
    initial = _heldout_nll(hypotheses, posterior, grid, heldout)
    mlp_baseline = _heldout_nll(
        (mlp,), TheoryPosterior(((mlp.hypothesis_id, 1.0),)), grid, heldout
    )
    knn_baseline = _knn_nll(warmup, grid, heldout)
    unconditional_baseline = _unconditional_nll(warmup, grid, heldout)

    records: list[Trial] = []
    rng = np.random.default_rng(rng_seed)
    for step, pool in enumerate(candidates):
        pool = np.asarray(pool, dtype=np.float32)
        if pool.ndim != 2 or pool.shape[1] != 2 or not len(pool):
            raise ValueError("nonempty 2D intervention pool required")
        ids = tuple(f"step-{step}-candidate-{i}" for i in range(len(pool)))
        predictions = _predictions(hypotheses, pool, grid, ids)
        selected_index, gain = _select(
            science, posterior, predictions, rng, policy,
        )
        selected_prediction = predictions[selected_index]
        x = tuple(float(v) for v in pool[selected_index])
        previous = posterior.probabilities

        # The environment ONLY receives selected X; it returns Y afterwards.
        observed_y = float(world.intervene(x))
        y_index = outcome_index(grid, observed_y)
        observation = Observation(
            experiment_id=ids[selected_index],
            outcome=(y_index,),
        )
        update = science.update(posterior, selected_prediction, observation)
        if not isinstance(update, PosteriorUpdate):
            raise RuntimeError("positive conditional masses must yield valid updates")
        posterior = update.posterior
        records.append(Trial(
            step,
            policy,
            x,
            observed_y,
            y_index,
            -log(update.evidence_probability),
            gain,
            previous,
            posterior.probabilities,
        ))

    final = _heldout_nll(hypotheses, posterior, grid, heldout)
    return StudyResult(
        law_label,
        seed,
        model_family,
        policy.value,
        len(warmup),
        len(candidates),
        len(heldout),
        initial,
        final,
        mlp_baseline,
        knn_baseline,
        unconditional_baseline,
        float(np.mean([t.prequential_nll for t in records])),
        float(np.mean([t.information_gain_bits for t in records])),
        posterior_entropy(posterior),
        sum(int(p.y < grid[0] or p.y > grid[-1]) for p in heldout),
        posterior.probabilities,
        tuple(records),
    )


def benchmark(
    law: Law, seed: int, config: StudyConfig,
) -> tuple[StudyResult, ...]:
    """Pair all policies on the same warmup, candidate pools and heldout Y."""
    if not isinstance(law, Law):
        raise TypeError("Law required")
    warmup_world = SyntheticInterventionWorld(law, seed=seed * 991 + 11)
    initial_rng = np.random.default_rng(seed * 991 + 12)
    initial_x = initial_rng.uniform(-0.9, 0.9, (config.bootstrap, 2))
    warmup = tuple(
        ObservedPair(tuple(x), float(warmup_world.intervene(tuple(x))))
        for x in initial_x
    )
    kan, mlp, grid = _fit_hypotheses(warmup, config, seed)

    evaluation_world = SyntheticInterventionWorld(law, seed=seed * 991 + 541)
    heldout_rng = np.random.default_rng(seed * 991 + 542)
    heldout_x = heldout_rng.uniform(
        -config.radius, config.radius, (config.heldout, 2),
    )
    heldout = tuple(
        ObservedPair(tuple(x), float(evaluation_world.intervene(tuple(x))))
        for x in heldout_x
    )
    candidate_rng = np.random.default_rng(seed * 991 + 401)
    pools = tuple(
        candidate_rng.uniform(
            -config.radius, config.radius, (config.candidates, 2),
        ).astype(np.float32)
        for _ in range(config.interventions)
    )

    rows: list[StudyResult] = []
    for family, hypotheses in (("kan", kan), ("mlp", mlp)):
        for policy in (SamplingPolicy.EIG, SamplingPolicy.RANDOM):
            world = SyntheticInterventionWorld(law, seed=seed * 991 + 201)
            rows.append(run_closed_loop(
                world=world,
                policy=policy,
                hypotheses=hypotheses,
                grid=grid,
                candidates=pools,
                rng_seed=seed * 991 + 301,
                heldout=heldout,
                warmup=warmup,
                mlp=mlp[0],
                law_label=law.value,
                seed=seed,
                model_family=family,
            ))
    return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--bootstrap", type=int, default=64)
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--interventions", type=int, default=12)
    parser.add_argument("--heldout", type=int, default=96)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    config = StudyConfig(
        bootstrap=args.bootstrap,
        train_steps=args.steps,
        interventions=args.interventions,
        heldout=args.heldout,
        candidates=args.candidates,
    )
    rows = [
        asdict(result)
        for law in Law for seed in args.seeds
        for result in benchmark(law, seed, config)
    ]
    report = {
        "config": asdict(config),
        "results": rows,
        "limitations": (
            "The normalized conditional is derived from a discriminative "
            "pair-energy model and a training-only Y marginal. Temperature "
            "is calibrated on initial observations. Neural hypotheses "
            "are frozen; only posterior weights update from interventions. "
            "No membership oracle or symbolic mechanism extraction."
        ),
    }
    body = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
