import pytest

from ecsa.contracts import (
    Observation,
    PopulationAnomaly,
    PosteriorUpdate,
    PredictiveDistribution,
    TheoryPosterior,
    TheoryRef,
)
from ecsa.science import ScienceKernel


def dist(theory: str, experiment: str, p_zero: float) -> PredictiveDistribution:
    return PredictiveDistribution(
        theory,
        experiment,
        (((0,), p_zero), ((1,), 1.0 - p_zero)),
    )


def test_information_gain_selects_discriminating_experiment() -> None:
    theories = (TheoryRef("h1", "a1"), TheoryRef("h2", "a2"))
    posterior = TheoryPosterior.uniform(theories)
    kernel = ScienceKernel()

    uninformative = (
        dist("h1", "same", 1.0),
        dist("h2", "same", 1.0),
    )
    discriminating = (
        dist("h1", "split", 0.0),
        dist("h2", "split", 1.0),
    )

    chosen = kernel.select_experiment(
        posterior,
        (uninformative, discriminating),
    )

    assert chosen.experiment_id == "split"
    assert chosen.information_gain_bits == pytest.approx(1.0)


def test_bayesian_update_moves_mass_to_theory_that_predicted_observation() -> None:
    posterior = TheoryPosterior((("h1", 0.5), ("h2", 0.5)))
    predictions = (
        dist("h1", "x", 0.0),
        dist("h2", "x", 1.0),
    )

    result = ScienceKernel().update(
        posterior,
        predictions,
        Observation("x", (0,)),
    )

    assert isinstance(result, PosteriorUpdate)
    assert result.posterior.probability("h1") == pytest.approx(0.0)
    assert result.posterior.probability("h2") == pytest.approx(1.0)
    assert result.evidence_probability == pytest.approx(0.5)


def test_zero_outcome_is_valid_evidence() -> None:
    posterior = TheoryPosterior((("h1", 0.8), ("h2", 0.2)))
    predictions = (
        dist("h1", "x", 0.9),
        dist("h2", "x", 0.1),
    )

    result = ScienceKernel().update(
        posterior,
        predictions,
        Observation("x", (0,)),
    )

    assert isinstance(result, PosteriorUpdate)
    expected = (0.8 * 0.9) / (0.8 * 0.9 + 0.2 * 0.1)
    assert result.posterior.probability("h1") == pytest.approx(expected)


def test_population_anomaly_is_not_uniform_reset() -> None:
    posterior = TheoryPosterior((("h1", 0.7), ("h2", 0.3)))
    predictions = (
        dist("h1", "x", 1.0),
        dist("h2", "x", 1.0),
    )

    result = ScienceKernel().update(
        posterior,
        predictions,
        Observation("x", (1,)),
    )

    assert isinstance(result, PopulationAnomaly)
    assert result.evidence_probability == 0.0
    assert result.likelihoods == (("h1", 0.0), ("h2", 0.0))


def test_posterior_rejects_non_normalized_distribution() -> None:
    with pytest.raises(ValueError):
        TheoryPosterior((("h1", 0.5), ("h2", 0.4)))


def test_admit_theory_assigns_small_prior_mass_and_preserves_relative_old_mass() -> None:
    posterior = TheoryPosterior((("h1", 0.8), ("h2", 0.2)))
    new_theory = TheoryRef("h3", "state-augmented:sha256:" + "a" * 64)

    admitted = ScienceKernel().admit_theory(
        posterior,
        new_theory,
        prior_mass=0.05,
    )

    assert admitted.probability("h3") == pytest.approx(0.05)
    assert admitted.probability("h1") == pytest.approx(0.76)
    assert admitted.probability("h2") == pytest.approx(0.19)
    assert admitted.probability("h1") / admitted.probability("h2") == pytest.approx(4.0)
