from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isclose, isfinite
from typing import TypeAlias

Scalar: TypeAlias = bool | int | float | str
Outcome: TypeAlias = tuple[Scalar, ...]


class ExperimentKind(StrEnum):
    INTERVENTIONAL = "interventional"


class PredictionStatus(StrEnum):
    COMPLETE = "complete"
    UNDEFINED = "undefined"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class TheoryRef:
    theory_id: str
    artifact_id: str

    def __post_init__(self) -> None:
        if not self.theory_id or not self.artifact_id:
            raise ValueError("theory_id and artifact_id are required")


@dataclass(frozen=True)
class Intervention:
    variable: str
    value: Scalar
    object_id: int = 0

    def __post_init__(self) -> None:
        if not self.variable:
            raise ValueError("intervention variable is required")
        if not isinstance(self.object_id, int) or self.object_id < 0:
            raise ValueError("object_id must be a nonnegative integer")


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    kind: ExperimentKind
    interventions: tuple[Intervention, ...]
    outcomes: tuple[str, ...]
    object_id: int = 0

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not isinstance(self.kind, ExperimentKind):
            raise ValueError("typed experiment kind required")
        if not isinstance(self.interventions, tuple) or not all(
            isinstance(x, Intervention) for x in self.interventions
        ):
            raise ValueError("immutable interventions required")
        if not isinstance(self.outcomes, tuple) or not self.outcomes:
            raise ValueError("at least one outcome variable is required")
        if len(set(self.outcomes)) != len(self.outcomes):
            raise ValueError("outcome variables must be distinct")
        if not all(isinstance(x, str) and x for x in self.outcomes):
            raise ValueError("outcome variables must be nonempty strings")
        if not isinstance(self.object_id, int) or self.object_id < 0:
            raise ValueError("object_id must be a nonnegative integer")


@dataclass(frozen=True)
class Observation:
    experiment_id: str
    outcome: Outcome

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not isinstance(self.outcome, tuple) or not self.outcome:
            raise ValueError("nonempty immutable outcome required")


@dataclass(frozen=True)
class PredictiveDistribution:
    theory_id: str
    experiment_id: str
    probabilities: tuple[tuple[Outcome, float], ...]

    def __post_init__(self) -> None:
        if not self.theory_id or not self.experiment_id:
            raise ValueError("theory_id and experiment_id are required")
        if not isinstance(self.probabilities, tuple) or not self.probabilities:
            raise ValueError("probability table is required")
        seen: set[Outcome] = set()
        total = 0.0
        for outcome, probability in self.probabilities:
            if not isinstance(outcome, tuple) or not outcome:
                raise ValueError("outcomes must be nonempty tuples")
            if outcome in seen:
                raise ValueError("duplicate outcome")
            seen.add(outcome)
            if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                raise ValueError("probabilities must be numeric")
            probability = float(probability)
            if not isfinite(probability) or probability < 0.0 or probability > 1.0:
                raise ValueError("probabilities must be finite in [0,1]")
            total += probability
        if not isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("distribution must sum to one")

    def probability(self, outcome: Outcome) -> float:
        return dict(self.probabilities).get(outcome, 0.0)


@dataclass(frozen=True)
class PredictionUnavailable:
    theory_id: str
    experiment_id: str
    status: PredictionStatus
    reason: str

    def __post_init__(self) -> None:
        if self.status is PredictionStatus.COMPLETE:
            raise ValueError("unavailable prediction cannot be complete")
        if not self.reason:
            raise ValueError("reason is required")


PredictionResult: TypeAlias = PredictiveDistribution | PredictionUnavailable


@dataclass(frozen=True)
class TheoryPosterior:
    probabilities: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.probabilities, tuple) or not self.probabilities:
            raise ValueError("posterior must contain theories")
        ids: set[str] = set()
        total = 0.0
        for theory_id, probability in self.probabilities:
            if not theory_id or theory_id in ids:
                raise ValueError("theory ids must be unique and nonempty")
            ids.add(theory_id)
            if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                raise ValueError("posterior probability must be numeric")
            probability = float(probability)
            if not isfinite(probability) or probability < 0.0 or probability > 1.0:
                raise ValueError("posterior probability must be finite in [0,1]")
            total += probability
        if not isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("posterior must sum to one")

    @classmethod
    def uniform(cls, theories: tuple[TheoryRef, ...]) -> "TheoryPosterior":
        if not theories:
            raise ValueError("at least one theory is required")
        probability = 1.0 / len(theories)
        return cls(tuple((theory.theory_id, probability) for theory in theories))

    def probability(self, theory_id: str) -> float:
        return dict(self.probabilities).get(theory_id, 0.0)


@dataclass(frozen=True)
class ExperimentScore:
    experiment_id: str
    information_gain_bits: float
    predictive_entropy_bits: float


@dataclass(frozen=True)
class PosteriorUpdate:
    posterior: TheoryPosterior
    evidence_probability: float
    likelihoods: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PopulationAnomaly:
    observation: Observation
    evidence_probability: float
    likelihoods: tuple[tuple[str, float], ...]
    reason: str = "observation unsupported by current theory population"


@dataclass(frozen=True)
class EvidenceRecord:
    prior: TheoryPosterior
    predictions: tuple[PredictiveDistribution, ...]
    observation: Observation
    result: PosteriorUpdate | PopulationAnomaly


@dataclass(frozen=True)
class EvidenceLedger:
    records: tuple[EvidenceRecord, ...] = ()

    def append(self, record: EvidenceRecord) -> "EvidenceLedger":
        if any(r.observation.experiment_id == record.observation.experiment_id for r in self.records):
            raise ValueError("experiment_id already recorded; repeated experiments require a new id")
        return EvidenceLedger(self.records + (record,))
