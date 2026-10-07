from __future__ import annotations

from ecsa.autonomy import (
    ActionAffordance,
    ActionRole,
    GENERIC_DISCOVERY_ASSUMPTIONS,
    AutonomousScientist,
    AutonomousWorldView,
    DialogOption,
    PublicEntityView,
)

from ..contracts import (
    JSONValue,
    PolicyDecision,
    ScientificContext,
)


_ROLE_MAP = {
    "TELEPORT_TO_LOCATION": ActionRole.VISIT_LOCATION,
    "TELEPORT_TO_OBJECT": ActionRole.VISIT_ENTITY,
    "TALK": ActionRole.OBSERVE_UNARY,
    "READ": ActionRole.OBSERVE_UNARY,
    "PICKUP": ActionRole.ACQUIRE,
    "OPEN": ActionRole.STATE_CHANGE,
    "ACTIVATE": ActionRole.STATE_CHANGE,
    "USE": ActionRole.PROBE_BINARY,
    "PUT": ActionRole.PLACE,
    "DISCOVERY_FEED_GET_UPDATES": ActionRole.PASSIVE,
}


def _entity_values(value: object):
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
    elif isinstance(value, dict):
        for item in value.values():
            yield from _entity_values(item)


def _entities(
    observation: dict[str, JSONValue],
) -> tuple[PublicEntityView, ...]:
    ui = observation.get("ui")
    if not isinstance(ui, dict):
        return ()
    found: dict[int, PublicEntityView] = {}

    def add(value: object, *, accessible: bool, inventory: bool) -> None:
        for item in _entity_values(value):
            entity_id = item.get("uuid")
            name = item.get("name")
            if type(entity_id) is not int or not isinstance(name, str):
                continue
            description = item.get("description")
            if not isinstance(description, str):
                description = name
            previous = found.get(entity_id)
            found[entity_id] = PublicEntityView(
                entity_id=entity_id,
                name=name,
                description=description,
                accessible=accessible or (
                    previous.accessible if previous is not None else False
                ),
                inventory=inventory or (
                    previous.inventory if previous is not None else False
                ),
            )

    add(ui.get("inventoryObjects"), accessible=True, inventory=True)
    add(
        ui.get("accessibleEnvironmentObjects"),
        accessible=True,
        inventory=False,
    )
    nearby = ui.get("nearbyObjects")
    if isinstance(nearby, dict):
        add(nearby.get("objects"), accessible=False, inventory=False)
    return tuple(
        sorted(found.values(), key=lambda item: item.entity_id)
    )


def _goal_text(observation: dict[str, JSONValue]) -> str:
    ui = observation.get("ui")
    if not isinstance(ui, dict):
        return ""
    progress = ui.get("taskProgress")
    if not isinstance(progress, list):
        return ""
    return " ".join(
        item["description"]
        for item in progress
        if isinstance(item, dict)
        and isinstance(item.get("description"), str)
    )


def _actions(
    available_actions: dict[str, JSONValue],
) -> tuple[ActionAffordance, ...]:
    found: list[ActionAffordance] = []
    for action_id, value in available_actions.items():
        role = _ROLE_MAP.get(action_id)
        if role is None or not isinstance(value, dict):
            continue
        args = value.get("args")
        arity = len(args) if isinstance(args, list) else 0
        found.append(ActionAffordance(action_id, role, arity))
    return tuple(found)


def _dialog(
    observation: dict[str, JSONValue],
) -> tuple[bool, tuple[DialogOption, ...]]:
    ui = observation.get("ui")
    if not isinstance(ui, dict):
        return False, ()
    box = ui.get("dialog_box")
    if not isinstance(box, dict):
        return False, ()
    options = box.get("dialogOptions")
    parsed: list[DialogOption] = []
    if isinstance(options, dict):
        for raw_id, raw_label in options.items():
            try:
                option_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if isinstance(raw_label, str) and raw_label:
                parsed.append(DialogOption(option_id, raw_label))
    flag = box.get("is_in_dialog")
    in_dialog = bool(
        flag
        if type(flag) is bool
        else parsed
    )
    return in_dialog, tuple(sorted(parsed, key=lambda item: item.option_id))


class AutonomousScientistPolicy:
    scientific_assumptions = GENERIC_DISCOVERY_ASSUMPTIONS

    def __init__(self, config: dict[str, JSONValue]) -> None:
        if not isinstance(config, dict):
            raise ValueError("policy config must be a JSON object")
        degree = config.get("max_polynomial_degree", 2)
        pair_trials = config.get("max_pair_trials", 160)
        tolerance = config.get("control_tolerance", 1.5)
        if type(degree) is not int:
            raise ValueError("max_polynomial_degree must be an integer")
        if type(pair_trials) is not int:
            raise ValueError("max_pair_trials must be an integer")
        if not isinstance(tolerance, (int, float)) or isinstance(
            tolerance, bool
        ):
            raise ValueError("control_tolerance must be numeric")
        self._scientist = AutonomousScientist(
            max_polynomial_degree=degree,
            max_pair_trials=pair_trials,
            control_tolerance=float(tolerance),
        )

    def decide(
        self,
        observation: dict[str, JSONValue],
        available_actions: dict[str, JSONValue],
        teleport_locations: dict[str, JSONValue],
        scientific_context: ScientificContext,
    ) -> PolicyDecision:
        in_dialog, dialog_options = _dialog(observation)
        view = AutonomousWorldView(
            goal_text=_goal_text(observation),
            entities=_entities(observation),
            actions=_actions(available_actions),
            locations=tuple(sorted(teleport_locations)),
            in_dialog=in_dialog,
            dialog_options=dialog_options,
        )
        decision = self._scientist.decide(
            view,
            scientific_context.generic_evidence,
            scientific_context.transfer_candidates,
        )

        if decision.dialog_option is not None:
            action = {
                "chosen_dialog_option_int": decision.dialog_option,
            }
        else:
            if decision.action_id is None:
                raise RuntimeError("generic scientist returned no action id")
            action = {"action": decision.action_id}
            if decision.location_arg is not None:
                action["arg1"] = decision.location_arg
            else:
                if len(decision.entity_args) >= 1:
                    action["arg1"] = decision.entity_args[0]
                if len(decision.entity_args) >= 2:
                    action["arg2"] = decision.entity_args[1]

        return PolicyDecision(
            action=action,
            reasoning=decision.reasoning,
            memory="generic ECSA autonomous scientist",
            generic_hypotheses=decision.hypotheses,
            generic_validation_hypothesis_ids=(
                decision.validation_hypothesis_ids
            ),
        )


def create_policy(
    config: dict[str, JSONValue],
) -> AutonomousScientistPolicy:
    return AutonomousScientistPolicy(config)
