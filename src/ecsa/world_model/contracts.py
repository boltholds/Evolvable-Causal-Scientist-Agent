from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias, runtime_checkable


RawScalar: TypeAlias = None | bool | int | float | str


class RawValueKind(StrEnum):
    SCALAR = "scalar"
    SEQUENCE = "sequence"
    MAPPING = "mapping"


FrozenPayload: TypeAlias = (
    RawScalar
    | tuple["FrozenRawValue", ...]
    | tuple[tuple[str, "FrozenRawValue"], ...]
)


@dataclass(frozen=True)
class FrozenRawValue:
    kind: RawValueKind
    value: FrozenPayload

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RawValueKind):
            raise ValueError("typed raw-value kind required")
        if self.kind is RawValueKind.SCALAR:
            if self.value is not None and type(self.value) not in (
                bool,
                int,
                float,
                str,
            ):
                raise ValueError("scalar raw value must be JSON-scalar-like")
            return
        if self.kind is RawValueKind.SEQUENCE:
            if not isinstance(self.value, tuple) or not all(
                isinstance(item, FrozenRawValue)
                for item in self.value
            ):
                raise ValueError(
                    "sequence raw value must contain FrozenRawValue items"
                )
            return
        if not isinstance(self.value, tuple) or not all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], FrozenRawValue)
            for item in self.value
        ):
            raise ValueError(
                "mapping raw value must contain string/FrozenRawValue pairs"
            )
        keys = tuple(key for key, _ in self.value)
        if len(set(keys)) != len(keys):
            raise ValueError("mapping raw-value keys must be unique")

    def thaw(self) -> object:
        if self.kind is RawValueKind.SCALAR:
            return self.value
        if self.kind is RawValueKind.SEQUENCE:
            assert isinstance(self.value, tuple)
            return [
                item.thaw()
                for item in self.value
            ]
        assert isinstance(self.value, tuple)
        return {
            key: item.thaw()
            for key, item in self.value
        }


def freeze_raw_value(value: object) -> FrozenRawValue:
    if isinstance(value, FrozenRawValue):
        return value
    if value is None or type(value) in (bool, int, float, str):
        return FrozenRawValue(RawValueKind.SCALAR, value)
    if isinstance(value, (list, tuple)):
        return FrozenRawValue(
            RawValueKind.SEQUENCE,
            tuple(freeze_raw_value(item) for item in value),
        )
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("raw mapping keys must be strings")
        return FrozenRawValue(
            RawValueKind.MAPPING,
            tuple(
                (
                    key,
                    freeze_raw_value(value[key]),
                )
                for key in sorted(value)
            ),
        )
    raise TypeError(
        "raw value must contain only scalars, sequences, and string-key mappings"
    )


@dataclass(frozen=True)
class RawObservation:
    observation_id: str
    step: int
    payload: FrozenRawValue

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id is required")
        if type(self.step) is not int or self.step < 0:
            raise ValueError("observation step must be nonnegative")
        if not isinstance(self.payload, FrozenRawValue):
            raise ValueError("observation payload must be frozen")


@dataclass(frozen=True)
class RawActionParameter:
    name: str
    public_candidates: tuple[FrozenRawValue, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("action parameter name is required")
        if not isinstance(self.public_candidates, tuple) or not all(
            isinstance(value, FrozenRawValue)
            for value in self.public_candidates
        ):
            raise ValueError(
                "public action candidates must be immutable frozen values"
            )


@dataclass(frozen=True)
class RawActionSchema:
    schema_id: str
    parameters: tuple[RawActionParameter, ...]
    public_metadata: FrozenRawValue

    def __post_init__(self) -> None:
        if not self.schema_id:
            raise ValueError("action schema_id is required")
        if not isinstance(self.parameters, tuple) or not all(
            isinstance(parameter, RawActionParameter)
            for parameter in self.parameters
        ):
            raise ValueError("action parameters must be immutable typed values")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(set(names)) != len(names):
            raise ValueError("action parameter names must be unique")
        if not isinstance(self.public_metadata, FrozenRawValue):
            raise ValueError("public action metadata must be frozen")


@dataclass(frozen=True)
class GroundAction:
    schema_id: str
    arguments: tuple[FrozenRawValue, ...]

    def __post_init__(self) -> None:
        if not self.schema_id:
            raise ValueError("ground action schema_id is required")
        if not isinstance(self.arguments, tuple) or not all(
            isinstance(argument, FrozenRawValue)
            for argument in self.arguments
        ):
            raise ValueError("ground action arguments must be frozen")

    def validate_against(self, schema: RawActionSchema) -> None:
        if not isinstance(schema, RawActionSchema):
            raise TypeError("schema must be RawActionSchema")
        if self.schema_id != schema.schema_id:
            raise ValueError("ground action schema does not match action schema")
        if len(self.arguments) != len(schema.parameters):
            raise ValueError(
                "ground action arity does not match action schema"
            )


@dataclass(frozen=True)
class RawActionOutcome:
    success: bool
    payload: FrozenRawValue

    def __post_init__(self) -> None:
        if type(self.success) is not bool:
            raise ValueError("raw action success must be bool")
        if not isinstance(self.payload, FrozenRawValue):
            raise ValueError("raw action outcome payload must be frozen")


@dataclass(frozen=True)
class InteractionTransition:
    transition_id: str
    before: RawObservation
    action: GroundAction
    outcome: RawActionOutcome
    after: RawObservation

    def __post_init__(self) -> None:
        if not self.transition_id:
            raise ValueError("transition_id is required")
        if not isinstance(self.before, RawObservation):
            raise ValueError("before must be RawObservation")
        if not isinstance(self.action, GroundAction):
            raise ValueError("action must be GroundAction")
        if not isinstance(self.outcome, RawActionOutcome):
            raise ValueError("outcome must be RawActionOutcome")
        if not isinstance(self.after, RawObservation):
            raise ValueError("after must be RawObservation")
        if self.after.step < self.before.step:
            raise ValueError("transition after-step cannot precede before-step")


@runtime_checkable
class RawEnvironmentPort(Protocol):
    def observe_raw(self) -> RawObservation: ...

    def list_raw_actions(self) -> tuple[RawActionSchema, ...]: ...

    def execute_raw_action(
        self,
        action: GroundAction,
    ) -> RawActionOutcome: ...

    @property
    def done(self) -> bool: ...

    @property
    def steps(self) -> int: ...
