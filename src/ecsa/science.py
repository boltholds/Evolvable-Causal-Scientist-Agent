from __future__ import annotations

from collections import defaultdict
from math import log2

from .contracts import (
    ExperimentScore,
    Observation,
    PopulationAnomaly,
    PosteriorUpdate,
    PredictiveDistribution,
    TheoryPosterior,
)


def _entropy_bits(probabilities: tuple[float, ...]) -> float:
    return -sum(p * log2(p) for p in probabilities if p > 0.0)


class ScienceKernel:
    """Minimal Bayesian theory-selection kernel inspired by PiEvo's scientific loop.

    The kernel operates only on prospective predictive distributions. It does not
    generate theories and it does not perform causal inference itself.
    """

    def __init__(self, *, population_anomaly_threshold: float = 0.0) -> None:
        if population_anomaly_threshold < 0.0:
            raise ValueError("population_anomaly_threshold must be nonnegative")
        self.population_anomaly_threshold = float(population_anomaly_threshold)

    @staticmethod
    def _prediction_map(
        posterior: TheoryPosterior,
        predictions: tuple[PredictiveDistribution, ...],
        experiment_id: str,
    ) -> dict[str, PredictiveDistribution]:
        by_theory: dict[str, PredictiveDistribution] = {}
        for prediction in predictions:
            if prediction.experiment_id != experiment_id:
                raise ValueError("prediction belongs to another experiment")
            if prediction.theory_id in by_theory:
                raise ValueError("duplicate prediction for theory")
            by_theory[prediction.theory_id] = prediction
        required = {theory_id for theory_id, _ in posterior.probabilities}
        if set(by_theory) != required:
            missing = sorted(required - set(by_theory))
            extra = sorted(set(by_theory) - required)
            raise ValueError(
                f"prediction set does not match posterior: missing={missing}, extra={extra}"
            )
        return by_theory

    def score_experiment(
        self,
        posterior: TheoryPosterior,
        predictions: tuple[PredictiveDistribution, ...],
    ) -> ExperimentScore:
        if not predictions:
            raise ValueError("predictions are required")
        experiment_id = predictions[0].experiment_id
        by_theory = self._prediction_map(posterior, predictions, experiment_id)
        prior = dict(posterior.probabilities)

        outcome_mass: dict[tuple, float] = defaultdict(float)
        outcomes: set[tuple] = set()
        for prediction in predictions:
            outcomes.update(outcome for outcome, _ in prediction.probabilities)
        for outcome in outcomes:
            outcome_mass[outcome] = sum(
                prior[theory_id] * by_theory[theory_id].probability(outcome)
                for theory_id in prior
            )

        prior_entropy = _entropy_bits(tuple(prior.values()))
        expected_posterior_entropy = 0.0
        for outcome, marginal in outcome_mass.items():
            if marginal <= 0.0:
                continue
            conditional = tuple(
                prior[theory_id]
                * by_theory[theory_id].probability(outcome)
                / marginal
                for theory_id in prior
            )
            expected_posterior_entropy += marginal * _entropy_bits(conditional)

        predictive_entropy = _entropy_bits(tuple(outcome_mass.values()))
        information_gain = max(0.0, prior_entropy - expected_posterior_entropy)
        return ExperimentScore(
            experiment_id, information_gain, predictive_entropy
        )

    def select_experiment(
        self,
        posterior: TheoryPosterior,
        candidates: tuple[tuple[PredictiveDistribution, ...], ...],
    ) -> ExperimentScore:
        if not candidates:
            raise ValueError("at least one experiment candidate is required")
        scores = tuple(
            self.score_experiment(posterior, predictions)
            for predictions in candidates
        )
        return min(
            scores,
            key=lambda score: (-score.information_gain_bits, score.experiment_id),
        )

    def admit_theory(
        self,
        posterior: TheoryPosterior,
        theory,
        *,
        prior_mass: float,
    ) -> TheoryPosterior:
        if not isinstance(prior_mass, (int, float)) or isinstance(prior_mass, bool):
            raise ValueError("prior_mass must be numeric")
        prior_mass = float(prior_mass)
        if not 0.0 < prior_mass < 1.0:
            raise ValueError("prior_mass must be strictly between zero and one")
        if posterior.probability(theory.theory_id) > 0.0:
            raise ValueError(f"theory already admitted: {theory.theory_id}")
        retained = 1.0 - prior_mass
        return TheoryPosterior(
            tuple(
                (theory_id, probability * retained)
                for theory_id, probability in posterior.probabilities
            )
            + ((theory.theory_id, prior_mass),)
        )

    def update(
        self,
        posterior: TheoryPosterior,
        predictions: tuple[PredictiveDistribution, ...],
        observation: Observation,
    ) -> PosteriorUpdate | PopulationAnomaly:
        by_theory = self._prediction_map(
            posterior, predictions, observation.experiment_id
        )
        prior = dict(posterior.probabilities)
        likelihoods = tuple(
            (
                theory_id,
                by_theory[theory_id].probability(observation.outcome),
            )
            for theory_id in prior
        )
        unnormalized = {
            theory_id: prior[theory_id] * likelihood
            for theory_id, likelihood in likelihoods
        }
        evidence_probability = sum(unnormalized.values())

        if evidence_probability <= self.population_anomaly_threshold:
            return PopulationAnomaly(
                observation, evidence_probability, likelihoods
            )

        updated = TheoryPosterior(
            tuple(
                (
                    theory_id,
                    unnormalized[theory_id] / evidence_probability,
                )
                for theory_id in prior
            )
        )
        return PosteriorUpdate(
            updated, evidence_probability, likelihoods
        )
