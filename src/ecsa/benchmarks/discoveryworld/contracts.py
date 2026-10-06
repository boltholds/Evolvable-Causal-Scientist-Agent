from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


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
