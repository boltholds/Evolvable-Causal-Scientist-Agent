from __future__ import annotations

from dataclasses import dataclass

from ecsa.world_model.contracts import (
    FrozenRawValue,
    GroundAction,
    RawActionOutcome,
    RawActionParameter,
    RawActionSchema,
    RawObservation,
    freeze_raw_value,
)

from .environment import DiscoveryWorldEnvironmentAdapter


_WIRE_OPTION_PREFIX = "__dw_wire_option__:"


def _collect_public_scalars(value: object) -> tuple[object, ...]:
    found: list[object] = []

    def visit(current: object) -> None:
        if current is None or type(current) in (int, float, str):
            found.append(current)
            return
        if type(current) is bool:
            return
        if isinstance(current, dict):
            for key in sorted(current):
                visit(key)
                visit(current[key])
            return
        if isinstance(current, (list, tuple)):
            for item in current:
                visit(item)

    visit(value)
    unique: dict[tuple[str, str], object] = {}
    for item in found:
        key = (type(item).__name__, repr(item))
        unique.setdefault(key, item)
    return tuple(unique.values())


def _dialog_options(observation: dict) -> tuple[tuple[int, str], ...]:
    ui = observation.get("ui")
    if not isinstance(ui, dict):
        return ()
    box = ui.get("dialog_box")
    if not isinstance(box, dict):
        return ()
    options = box.get("dialogOptions")
    if not isinstance(options, dict):
        return ()

    parsed: list[tuple[int, str]] = []
    for raw_id, label in options.items():
        try:
            option_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if isinstance(label, str) and label:
            parsed.append((option_id, label))
    return tuple(sorted(parsed))


@dataclass
class DiscoveryWorldRawEnvironment:
    _environment: DiscoveryWorldEnvironmentAdapter

    @classmethod
    def reactor_lab_normal(
        cls,
        seed: int,
        *,
        max_steps: int = 1000,
    ) -> "DiscoveryWorldRawEnvironment":
        return cls(
            DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(
                seed,
                max_steps=max_steps,
            )
        )

    def observe_raw(self) -> RawObservation:
        observation = self._environment.observe()
        payload = {
            "observation": observation,
            "public_auxiliary": {
                "teleport_locations": (
                    self._environment.teleport_locations()
                ),
            },
        }
        return RawObservation(
            observation_id=f"dw-observation:{self.steps}",
            step=self.steps,
            payload=freeze_raw_value(payload),
        )

    def list_raw_actions(self) -> tuple[RawActionSchema, ...]:
        observation = self._environment.observe()
        action_metadata = self._environment.available_actions()
        auxiliary = self._environment.teleport_locations()

        candidates = tuple(
            freeze_raw_value(value)
            for value in _collect_public_scalars(
                {
                    "observation": observation,
                    "action_metadata": action_metadata,
                    "auxiliary": auxiliary,
                }
            )
        )

        schemas: list[RawActionSchema] = []
        for schema_id in sorted(action_metadata):
            metadata = action_metadata[schema_id]
            if not isinstance(metadata, dict):
                continue
            raw_args = metadata.get("args", [])
            if not isinstance(raw_args, list) or not all(
                isinstance(value, str) and value
                for value in raw_args
            ):
                continue
            schemas.append(
                RawActionSchema(
                    schema_id=schema_id,
                    parameters=tuple(
                        RawActionParameter(
                            name=name,
                            public_candidates=candidates,
                        )
                        for name in raw_args
                    ),
                    public_metadata=freeze_raw_value(metadata),
                )
            )

        for option_id, label in _dialog_options(observation):
            schemas.append(
                RawActionSchema(
                    schema_id=(
                        f"{_WIRE_OPTION_PREFIX}{option_id}"
                    ),
                    parameters=(),
                    public_metadata=freeze_raw_value(
                        {
                            "label": label,
                            "wire_kind": "option",
                        }
                    ),
                )
            )

        return tuple(
            sorted(schemas, key=lambda value: value.schema_id)
        )

    def execute_raw_action(
        self,
        action: GroundAction,
    ) -> RawActionOutcome:
        if not isinstance(action, GroundAction):
            raise TypeError("action must be GroundAction")

        if action.schema_id.startswith(_WIRE_OPTION_PREFIX):
            if action.arguments:
                raise ValueError(
                    "wire option actions take no arguments"
                )
            raw_option = action.schema_id[len(_WIRE_OPTION_PREFIX):]
            try:
                option_id = int(raw_option)
            except ValueError as exc:
                raise ValueError(
                    "invalid wire option action id"
                ) from exc
            packet = {
                "chosen_dialog_option_int": option_id,
            }
        else:
            schemas = {
                schema.schema_id: schema
                for schema in self.list_raw_actions()
                if not schema.schema_id.startswith(
                    _WIRE_OPTION_PREFIX
                )
            }
            schema = schemas.get(action.schema_id)
            if schema is None:
                raise ValueError(
                    f"unknown raw action schema: {action.schema_id}"
                )
            action.validate_against(schema)
            packet: dict[str, object] = {
                "action": action.schema_id,
            }
            for parameter, argument in zip(
                schema.parameters,
                action.arguments,
            ):
                value = argument.thaw()
                if value is not None and type(value) not in (
                    bool,
                    int,
                    float,
                    str,
                ):
                    raise ValueError(
                        "DiscoveryWorld raw action arguments "
                        "must thaw to JSON scalars"
                    )
                packet[parameter.name] = value

        result = self._environment.act(packet)  # type: ignore[arg-type]
        return RawActionOutcome(
            success=result.success,
            payload=freeze_raw_value(
                {
                    "errors": list(result.errors),
                }
            ),
        )

    @property
    def done(self) -> bool:
        return self._environment.done

    @property
    def steps(self) -> int:
        return self._environment.steps
