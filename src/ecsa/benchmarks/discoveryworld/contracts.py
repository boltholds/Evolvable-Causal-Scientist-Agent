from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Protocol, TypeAlias, runtime_checkable

from ecsa.autonomy import (
    GenericNumericHypothesis,
    GenericNumericPrediction,
    GenericScalarEvidence,
)
from ecsa.mechanisms import MechanismRecord, MechanismVersionRef


JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = (
    JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
)
ActionPacket: TypeAlias = dict[str, JSONScalar]


def validate_action_packet(action: object) -> None:
    if not isinstance(action, dict):
        raise ValueError("action packet must be a dictionary")
    if not all(
        isinstance(key, str)
        and (
            value is None
            or type(value) in (bool, int, float, str)
        )
        for key, value in action.items()
    ):
        raise ValueError("action packet keys/values must be JSON scalars")

    ordinary = isinstance(action.get("action"), str) and bool(
        action.get("action")
    )
    dialog_value = action.get("chosen_dialog_option_int")
    dialog = type(dialog_value) is int and dialog_value >= 0
    if ordinary == dialog:
        raise ValueError(
            "action packet must contain exactly one of a nonempty 'action' "
            "or a nonnegative 'chosen_dialog_option_int'"
        )


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
    observed_frequency: float | None = None

    def __post_init__(self) -> None:
        if not self.event_id or not self.hypothesis_id:
            raise ValueError("validation ids are required")
        if type(self.validation_step) is not int or self.validation_step < 0:
            raise ValueError("validation step must be nonnegative")
        if type(self.success) is not bool:
            raise ValueError("validation success must be bool")
        if (
            self.observed_frequency is not None
            and not isfinite(self.observed_frequency)
        ):
            raise ValueError("observed_frequency must be finite when present")



@dataclass(frozen=True)
class ScientificContext:
    context_id: str
    measurements: tuple[ReactorMeasurement, ...]
    transfer_candidates: tuple[MechanismRecord, ...]
    admitted_mechanisms: tuple[MechanismRecord, ...]
    generic_evidence: tuple[GenericScalarEvidence, ...] = ()


@dataclass(frozen=True)
class PolicyDecision:
    action: ActionPacket
    reasoning: str | None = None
    memory: str | None = None
    hypotheses: tuple[ReactorMechanismHypothesis, ...] = ()
    validation_hypothesis_ids: tuple[str, ...] = ()
    generic_hypotheses: tuple[GenericNumericHypothesis, ...] = ()
    generic_validation_hypothesis_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_action_packet(self.action)
        if not isinstance(self.hypotheses, tuple):
            raise ValueError("hypotheses must be immutable")
        if (
            not isinstance(self.validation_hypothesis_ids, tuple)
            or not all(
                isinstance(value, str) and value
                for value in self.validation_hypothesis_ids
            )
            or len(set(self.validation_hypothesis_ids))
            != len(self.validation_hypothesis_ids)
        ):
            raise ValueError(
                "validation_hypothesis_ids must be unique immutable strings"
            )
        if (
            not isinstance(self.generic_hypotheses, tuple)
            or not all(
                isinstance(value, GenericNumericHypothesis)
                for value in self.generic_hypotheses
            )
        ):
            raise ValueError("generic_hypotheses must be immutable typed values")
        if (
            not isinstance(self.generic_validation_hypothesis_ids, tuple)
            or not all(
                isinstance(value, str) and value
                for value in self.generic_validation_hypothesis_ids
            )
            or len(set(self.generic_validation_hypothesis_ids))
            != len(self.generic_validation_hypothesis_ids)
        ):
            raise ValueError(
                "generic_validation_hypothesis_ids must be unique immutable strings"
            )


@runtime_checkable
class DiscoveryWorldActionPolicy(Protocol):
    def decide(
        self,
        observation: dict[str, JSONValue],
        available_actions: dict[str, JSONValue],
        teleport_locations: dict[str, JSONValue],
        scientific_context: ScientificContext,
    ) -> PolicyDecision: ...


@dataclass(frozen=True)
class ArenaEpisodeResult:
    config: DiscoveryWorldEpisodeConfig
    evaluation: DiscoveryWorldEvaluation
    admitted_mechanisms: tuple[MechanismRecord, ...]
    measurement_count: int



class ArenaArm(StrEnum):
    COLD = "cold"
    REUSE = "reuse"


@dataclass(frozen=True)
class ArmEpisodeResult:
    arm: ArenaArm
    seed: int
    episode: ArenaEpisodeResult
    starting_mechanism_count: int
    ending_mechanism_count: int
    policy_config_hash: str


@dataclass(frozen=True)
class PairedSeedResult:
    seed: int
    cold: ArmEpisodeResult
    reuse: ArmEpisodeResult


@dataclass(frozen=True)
class TransferArenaResult:
    pairs: tuple[PairedSeedResult, ...]
    policy_config_hash: str
