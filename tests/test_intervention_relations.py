"""Closed-loop tests: EIG selects do(X) and updates from observed Y."""
from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
from torch import nn

from ecsa.contracts import Observation, PosteriorUpdate, TheoryPosterior
from ecsa.science import ScienceKernel
from ecsa.experimental.intervention_relations import (
    InterventionWorld,
    SamplingPolicy,
    StudyConfig,
    ObservedPair,
    RelationHypothesis,
    SyntheticInterventionWorld,
    _mass,
    _predictions,
    _select,
    benchmark,
    outcome_index,
    run_closed_loop,
)
from ecsa.experimental.relation_discovery import Law


class FixedRelation(nn.Module):
    """Competing X/Y relation hypotheses used only for the test."""
    def __init__(self, sign: float):
        super().__init__()
        self.sign = sign

    def forward(self, x_view, y_view):
        return -8.0 * (y_view[:, 0] - self.sign * x_view[:, 0]).square()


class ObservableWorld:
    """Intervention-only; no ground-truth law or membership-check method."""
    def __init__(self):
        self.calls = []

    def intervene(self, x):
        self.calls.append(x)
        return x[0]


def fixtures():
    grid = np.linspace(-1, 1, 31, dtype=np.float32)
    hypotheses = tuple(
        RelationHypothesis(
            name, FixedRelation(sign), 1.0,
            np.ones(len(grid)) / len(grid),
        )
        for name, sign in (("plus", 1.0), ("minus", -1.0))
    )
    warmup = tuple(
        ObservedPair((float(x), 0.0), float(x))
        for x in np.linspace(-0.8, 0.8, 12)
    )
    return grid, hypotheses, warmup


def test_intervention_selects_x_before_y_is_available():
    grid, hypotheses, warmup = fixtures()
    assert isinstance(ObservableWorld(), InterventionWorld)
    pool = np.asarray([[0.0, 0.0], [0.9, 0.0]], dtype=np.float32)
    candidates = _predictions(hypotheses, pool, grid, ("zero", "informative"))
    science = ScienceKernel()
    prior = TheoryPosterior((("plus", 0.5), ("minus", 0.5)))
    gains = [
        science.score_experiment(prior, item).information_gain_bits
        for item in candidates
    ]
    assert gains[1] > gains[0] + 0.05
    index, gain = _select(
        science, prior, candidates, np.random.default_rng(7),
        SamplingPolicy.EIG,
    )
    assert index == 1
    assert gain > 0

    world = ObservableWorld()
    outcome = run_closed_loop(
        world=world,
        policy=SamplingPolicy.EIG,
        hypotheses=hypotheses,
        grid=grid,
        candidates=(pool, pool),
        rng_seed=7,
        heldout=warmup,
        warmup=warmup,
        mlp=hypotheses[0],
    )
    assert len(world.calls) == 2
    assert world.calls[0][0] == pytest.approx(0.9, abs=1e-5)
    assert outcome.posterior[0][1] > 0.5
    assert outcome.trials[0].prior == (("plus", 0.5), ("minus", 0.5))
    assert outcome.trials[0].posterior != outcome.trials[0].prior
    assert outcome.trials[1].prior == outcome.trials[0].posterior
    assert all(t.y_observed == pytest.approx(t.x[0]) for t in outcome.trials)
    assert outcome.interventions == 2


def test_science_kernel_receives_valid_categorical_y_probabilities():
    grid, hypotheses, _ = fixtures()
    distributions = _predictions(
        hypotheses,
        np.asarray([[0.8, 0.0]], dtype=np.float32),
        grid,
        ("do-1",),
    )[0]
    assert len(distributions) == 2
    for distribution in distributions:
        assert sum(p for _, p in distribution.probabilities) == pytest.approx(
            1.0, abs=1e-9
        )
        assert min(p for _, p in distribution.probabilities) > 0.0
    result = ScienceKernel().update(
        TheoryPosterior((("plus", 0.5), ("minus", 0.5))),
        distributions,
        Observation("do-1", (outcome_index(grid, 0.8),)),
    )
    assert isinstance(result, PosteriorUpdate)
    assert result.posterior.probability("plus") > 0.9


def test_classification_energy_is_reweighted_by_y_marginal():
    marginal = np.asarray([0.05, 0.20, 0.75])
    mass = _mass(np.zeros((1, 3)), 1.0, marginal, floor=0.0)
    assert np.allclose(mass[0], marginal)
    with pytest.raises(ValueError):
        _mass(np.zeros((1, 3)), 0.0, marginal)


def test_random_and_eig_use_equal_action_budgets():
    grid, hypotheses, warmup = fixtures()
    pools = tuple(
        np.asarray([[0.0, 0.0], [0.9, 0.0], [-0.9, 0.0]], dtype=np.float32)
        for _ in range(3)
    )
    outcomes = []
    for policy in (SamplingPolicy.EIG, SamplingPolicy.RANDOM):
        world = ObservableWorld()
        result = run_closed_loop(
            world=world, policy=policy, hypotheses=hypotheses,
            grid=grid, candidates=pools, rng_seed=5,
            heldout=warmup, warmup=warmup, mlp=hypotheses[0],
        )
        assert len(world.calls) == 3
        assert len(result.trials) == 3
        outcomes.append(result)
    assert outcomes[0].initial_nll == outcomes[1].initial_nll


def test_world_returns_multivalued_y_without_membership_oracle():
    world = SyntheticInterventionWorld(Law.IMPLICIT, seed=7, noise=0.0)
    values = np.asarray([world.intervene((0.2, -0.3)) for _ in range(80)])
    assert (values > 0).any() and (values < 0).any()
    assert len(world.calls) == 80
    assert not hasattr(world, "check_pair")
    assert not hasattr(world, "membership")


def test_short_benchmark_matches_kan_and_mlp_interventions():
    config = StudyConfig(
        bootstrap=28, calibration=8, train_steps=25,
        interventions=3, candidates=8, bins=21, heldout=16,
    )
    active, random, mlp_active, mlp_random = benchmark(
        Law.ADDITIVE, seed=3, config=config
    )
    assert active.policy == mlp_active.policy == "eig"
    assert random.policy == mlp_random.policy == "random"
    assert active.model_family == random.model_family == "kan"
    assert mlp_active.model_family == mlp_random.model_family == "mlp"
    assert active.interventions == random.interventions == 3
    assert mlp_active.interventions == mlp_random.interventions == 3
    assert active.initial_nll == random.initial_nll
    assert mlp_active.initial_nll == mlp_random.initial_nll
    assert active.knn_nll == random.knn_nll
    assert 0.0 < active.initial_nll < 12.0
    assert math.isfinite(active.final_nll)


def test_invalid_budget_and_nonfinite_y():
    with pytest.raises(ValueError):
        StudyConfig(calibration=70)
    with pytest.raises(ValueError):
        outcome_index(np.linspace(0.0, 1.0, 9), float("nan"))
