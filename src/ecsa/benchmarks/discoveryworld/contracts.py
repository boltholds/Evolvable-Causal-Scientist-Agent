from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import TypeAlias

from ecsa.mechanisms import MechanismVersionRef


JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = (
    JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
)
ActionPacket: TypeAlias = dict[str, JSONScalar]


@dataclass(frozen=True)
class DiscoveryWorldEpisodeConfig:
    scenario: str
    difficulty: str
    seed: int
    max_steps: int

    def __post_init__(self) -> None:
        if not self.scenario:
            raise ValueError("scenario is required")
        if not self.difficulty:
            raise ValueError("difficulty is required")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")


@dataclass(frozen=True)
class DiscoveryWorldActionResult:
    success: bool
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.success) is not bool:
            raise ValueError("success must be bool")
        if not isinstance(self.errors, tuple) or not all(
            isinstance(error, str) for error in self.errors
        ):
            raise ValueError("errors must be an immutable string tuple")


@dataclass(frozen=True)
class DiscoveryWorldEvaluation:
    completed_successfully: bool
    score_normalized: float
    steps: int
    scorecard: list[dict[str, JSONValue]]

    def __post_init__(self) -> None:
        if type(self.completed_successfully) is not bool:
            raise ValueError("completed_successfully must be bool")
        if not isinstance(self.score_normalized, (int, float)) or isinstance(
            self.score_normalized, bool
        ):
            raise ValueError("score_normalized must be numeric")
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if not isinstance(self.scorecard, list):
            raise ValueError("scorecard must be a list")



class MeasurementKind(StrEnum):
    DENSITY = "density"
    TEMPERATURE = "temperature"
    QUANTUM_SIZE = "quantum_size"
    RADIATION = "radiation"
    SPECTRUM = "spectrum"


@dataclass(frozen=True)
class ParsedMeasurement:
    kind: MeasurementKind
    values: tuple[float, ...]
    channel_values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MeasurementKind):
            raise ValueError("typed measurement kind required")
        if not self.values or not all(isfinite(value) for value in self.values):
            raise ValueError("measurement values must be finite and nonempty")
        if self.channel_values and not all(
            isfinite(value) for value in self.channel_values
        ):
            raise ValueError("channel values must be finite")


@dataclass(frozen=True)
class ReactorMeasurement:
    evidence_id: str
    step: int
    context_id: str
    crystal_uuid: int
    crystal_name: str
    instrument_uuid: int
    instrument_name: str
    kind: MeasurementKind
    values: tuple[float, ...]
    raw_message: str

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.context_id:
            raise ValueError("measurement evidence/context id is required")
        if type(self.step) is not int or self.step < 0:
            raise ValueError("measurement step must be nonnegative")
        if type(self.crystal_uuid) is not int or type(self.instrument_uuid) is not int:
            raise ValueError("DiscoveryWorld UUIDs must be integers")
        if not self.crystal_name or not self.instrument_name or not self.raw_message:
            raise ValueError("public measurement names/message are required")
        if not self.values or not all(isfinite(value) for value in self.values):
            raise ValueError("measurement values must be finite and nonempty")


@dataclass(frozen=True)
class ReactorFrequencyPrediction:
    target_crystal_uuid: int
    target_reactor_uuid: int
    predicted_frequency: float
    frozen_step: int

    def __post_init__(self) -> None:
        if type(self.target_crystal_uuid) is not int or type(self.target_reactor_uuid) is not int:
            raise ValueError("prediction UUIDs must be integers")
        if not isfinite(self.predicted_frequency):
            raise ValueError("predicted frequency must be finite")
        if type(self.frozen_step) is not int or self.frozen_step < 0:
            raise ValueError("frozen_step must be nonnegative")


@dataclass(frozen=True)
class ReactorMechanismHypothesis:
    hypothesis_id: str
    measurement_kind: MeasurementKind
    slope: float
    offset: float
    source_evidence_ids: tuple[str, ...]
    predictions: tuple[ReactorFrequencyPrediction, ...]
    source_mechanism: MechanismVersionRef | None = None

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id is required")
        if not isinstance(self.measurement_kind, MeasurementKind):
            raise ValueError("typed measurement kind required")
        if not isfinite(self.slope) or not isfinite(self.offset):
            raise ValueError("linear parameters must be finite")
        if not isinstance(self.source_evidence_ids, tuple) or not all(
            isinstance(value, str) and value for value in self.source_evidence_ids
        ):
            raise ValueError("source evidence ids must be immutable strings")
        if not isinstance(self.predictions, tuple) or not self.predictions:
            raise ValueError("at least one frozen prediction is required")
        if not all(
            isinstance(value, ReactorFrequencyPrediction)
            for value in self.predictions
        ):
            raise ValueError("typed reactor predictions required")
        if self.source_mechanism is not None and not isinstance(
            self.source_mechanism, MechanismVersionRef
        ):
            raise ValueError("source_mechanism must be a MechanismVersionRef")


@dataclass(frozen=True)
class ReactorValidationEvent:
    event_id: str
    hypothesis_id: str
    target_crystal_uuid: int
    target_reactor_uuid: int
    predicted_frequency: float
    validation_step: int
    success: bool

    def __post_init__(self) -> None:
        if not self.event_id or not self.hypothesis_id:
            raise ValueError("validation ids are required")
        if type(self.validation_step) is not int or self.validation_step < 0:
            raise ValueError("validation step must be nonnegative")
        if type(self.success) is not bool:
            raise ValueError("validation success must be bool")
