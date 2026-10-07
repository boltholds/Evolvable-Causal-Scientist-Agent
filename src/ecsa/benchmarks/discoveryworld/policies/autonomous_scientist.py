from __future__ import annotations

import math
import re
from dataclasses import dataclass
from hashlib import sha256
from itertools import permutations
from typing import Iterable

from ecsa.mechanisms import MechanismRecord, MechanismVersionRef

from ..contracts import (
    GenericNumericHypothesis,
    GenericNumericPrediction,
    GenericScalarEvidence,
    JSONValue,
    PolicyDecision,
    ScientificContext,
)


_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_NUMBER = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d+)?"
)
_STATE = re.compile(r"\(([^()]*)\)")
_TRANSFER_KEY = "scalar-interaction-to-control"


@dataclass(frozen=True)
class _Entity:
    entity_id: int
    name: str
    description: str
    accessible: bool
    inventory: bool


@dataclass(frozen=True)
class _Fit:
    feature_key: str
    control_feature_key: str
    degree: int
    coefficients: tuple[float, ...]
    score: float
    source_evidence_ids: tuple[str, ...]


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN.findall(value))


def _base_name(value: str) -> str:
    value = _STATE.sub(" ", value)
    value = re.sub(r"\d+", " ", value)
    return " ".join(_tokens(value))


def _state_text(value: str) -> str:
    matches = _STATE.findall(value)
    return " ".join(matches).lower()


def _task_text(observation: dict[str, JSONValue]) -> str:
    ui = observation.get("ui")
    if not isinstance(ui, dict):
        return ""
    progress = ui.get("taskProgress")
    if not isinstance(progress, list):
        return ""
    parts: list[str] = []
    for item in progress:
        if isinstance(item, dict):
            description = item.get("description")
            if isinstance(description, str):
                parts.append(description)
    return " ".join(parts)


def _entity_values(value: object) -> Iterable[dict]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
    elif isinstance(value, dict):
        for item in value.values():
            yield from _entity_values(item)


def _entities(observation: dict[str, JSONValue]) -> dict[int, _Entity]:
    ui = observation.get("ui")
    if not isinstance(ui, dict):
        return {}

    found: dict[int, _Entity] = {}

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
            found[entity_id] = _Entity(
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
    return found


def _dialog(observation: dict[str, JSONValue]) -> dict[str, JSONValue]:
    ui = observation.get("ui")
    if not isinstance(ui, dict):
        return {}
    dialog = ui.get("dialog_box")
    return dialog if isinstance(dialog, dict) else {}


def _dialog_options(
    observation: dict[str, JSONValue],
) -> tuple[tuple[int, str], ...]:
    options = _dialog(observation).get("dialogOptions")
    if not isinstance(options, dict):
        return ()
    parsed: list[tuple[int, str]] = []
    for raw_index, raw_label in options.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if isinstance(raw_label, str):
            parsed.append((index, raw_label))
    return tuple(sorted(parsed))


def _is_in_dialog(observation: dict[str, JSONValue]) -> bool:
    dialog = _dialog(observation)
    flag = dialog.get("is_in_dialog")
    if type(flag) is bool:
        return flag
    text = dialog.get("dialogIn")
    options = dialog.get("dialogOptions")
    return bool(
        isinstance(text, str)
        and text
        and isinstance(options, dict)
        and options
    )


def _action_args(
    available_actions: dict[str, JSONValue],
    action_name: str,
) -> tuple[str, ...]:
    value = available_actions.get(action_name)
    if not isinstance(value, dict):
        return ()
    args = value.get("args")
    if not isinstance(args, list):
        return ()
    return tuple(value for value in args if isinstance(value, str))


def _similarity(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    shared_numbers = {
        token
        for token in left_tokens & right_tokens
        if token.isdigit()
    }
    words_left = {token for token in left_tokens if not token.isdigit()}
    words_right = {token for token in right_tokens if not token.isdigit()}
    union = words_left | words_right
    word_score = (
        len(words_left & words_right) / len(union)
        if union
        else 0.0
    )
    return word_score + 2.0 * len(shared_numbers)


def _poly_value(coefficients: tuple[float, ...], x: float) -> float:
    return sum(
        coefficient * (x ** power)
        for power, coefficient in enumerate(coefficients)
    )


def _solve(matrix: list[list[float]], vector: list[float]) -> tuple[float, ...] | None:
    n = len(vector)
    augmented = [
        [*map(float, matrix[row]), float(vector[row])]
        for row in range(n)
    ]
    for column in range(n):
        pivot = max(
            range(column, n),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        scale = augmented[column][column]
        augmented[column] = [
            value / scale
            for value in augmented[column]
        ]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][index]
                - factor * augmented[column][index]
                for index in range(n + 1)
            ]
    return tuple(augmented[row][-1] for row in range(n))


def _fit_polynomial(
    points: tuple[tuple[float, float], ...],
    degree: int,
) -> tuple[tuple[float, ...], float] | None:
    if not points or degree < 0 or len(points) <= degree:
        return None
    width = degree + 1
    matrix = [[0.0] * width for _ in range(width)]
    vector = [0.0] * width
    for x, y in points:
        powers = [x ** power for power in range(width)]
        for row in range(width):
            vector[row] += powers[row] * y
            for column in range(width):
                matrix[row][column] += (
                    powers[row] * powers[column]
                )
    coefficients = _solve(matrix, vector)
    if coefficients is None:
        return None
    residual = sum(
        (_poly_value(coefficients, x) - y) ** 2
        for x, y in points
    )
    scale = max(
        1.0,
        sum(abs(y) for _, y in points) / len(points),
    )
    normalized = residual / (len(points) * scale * scale)
    complexity = (degree + 1) * math.log(len(points) + 1.0)
    score = math.log(normalized + 1e-12) + 0.05 * complexity
    return coefficients, score


def _parameters(mechanism: MechanismRecord) -> dict[str, object]:
    return {name: value for name, value in mechanism.parameters}


class AutonomousScientistPolicy:
    """Generic DiscoveryWorld controller driven by public evidence and novelty."""

    def __init__(self, config: dict[str, JSONValue]) -> None:
        if not isinstance(config, dict):
            raise ValueError("policy config must be a JSON object")
        degree = config.get("max_polynomial_degree", 2)
        if type(degree) is not int or not 0 <= degree <= 4:
            raise ValueError("max_polynomial_degree must be in 0..4")
        pair_trials = config.get("max_pair_trials", 160)
        if type(pair_trials) is not int or pair_trials < 1:
            raise ValueError("max_pair_trials must be positive")
        tolerance = config.get("control_tolerance", 1.5)
        if (
            not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or float(tolerance) <= 0
        ):
            raise ValueError("control_tolerance must be positive")

        self.max_degree = degree
        self.max_pair_trials = pair_trials
        self.control_tolerance = float(tolerance)

        self._step = 0
        self._labels: dict[int, str] = {}
        self._descriptions: dict[int, str] = {}
        self._attempted: set[tuple] = set()
        self._visited_locations: set[str] = set()
        self._visited_objects: set[int] = set()
        self._pair_trials = 0
        self._emitted: set[str] = set()
        self._failed: set[str] = set()
        self._completed_predictions: set[tuple[str, int]] = set()
        self._active: GenericNumericHypothesis | None = None
        self._dialog_entity: int | None = None
        self._pending_validation: tuple[str, int] | None = None
        self._put_attempted: set[tuple[str, int]] = set()

    def decide(
        self,
        observation: dict[str, JSONValue],
        available_actions: dict[str, JSONValue],
        teleport_locations: dict[str, JSONValue],
        scientific_context: ScientificContext,
    ) -> PolicyDecision:
        current = _entities(observation)
        for entity in current.values():
            self._labels[entity.entity_id] = entity.name
            self._descriptions[entity.entity_id] = entity.description

        self._resolve_pending(observation, available_actions)

        if _is_in_dialog(observation):
            decision = self._dialog_decision(
                observation,
                scientific_context,
            )
            self._step += 1
            return decision

        if self._active is None:
            candidates = self._discover_hypotheses(
                scientific_context,
                available_actions,
            )
            if candidates:
                self._active = candidates[0]

        if self._active is not None:
            decision = self._pursue_active(
                observation,
                available_actions,
                scientific_context,
                current,
            )
            if decision is not None:
                self._step += 1
                return decision

        decision = self._explore(
            observation,
            available_actions,
            teleport_locations,
            scientific_context,
            current,
        )
        self._step += 1
        return decision

    def _decision(
        self,
        action: dict,
        *,
        reasoning: str,
        emit: GenericNumericHypothesis | None = None,
        validate: str | None = None,
    ) -> PolicyDecision:
        hypotheses = (
            (emit,)
            if emit is not None
            and emit.hypothesis_id not in self._emitted
            else ()
        )
        if hypotheses:
            self._emitted.add(emit.hypothesis_id)
        validations = (validate,) if validate is not None else ()
        return PolicyDecision(
            action=action,
            reasoning=reasoning,
            memory=(
                f"step={self._step}; evidence-driven generic exploration; "
                f"failed={len(self._failed)}"
            ),
            generic_hypotheses=hypotheses,
            generic_validation_hypothesis_ids=validations,
        )

    def _explore(
        self,
        observation: dict[str, JSONValue],
        available_actions: dict[str, JSONValue],
        teleport_locations: dict[str, JSONValue],
        scientific_context: ScientificContext,
        current: dict[int, _Entity],
    ) -> PolicyDecision:
        structural_prior = self._transfer_source(
            scientific_context
        ) is not None

        location = self._next_location(
            observation,
            teleport_locations,
        )
        if location is not None and not any(
            entity.accessible
            and self._relevance(observation, entity) > 0
            for entity in current.values()
        ):
            self._visited_locations.add(location)
            return self._decision(
                {
                    "action": "TELEPORT_TO_LOCATION",
                    "arg1": location,
                },
                reasoning="visit the most goal-relevant unexplored location",
            )

        ranked = sorted(
            (
                entity
                for entity in current.values()
                if entity.accessible
            ),
            key=lambda entity: (
                -self._relevance(observation, entity),
                entity.entity_id,
            ),
        )

        unary_order = (
            ("TALK", "inspect public dialog/state"),
            ("PICKUP", "test object mobility and expose affordances"),
            ("READ", "collect public textual evidence"),
        )
        if not structural_prior:
            unary_order += (
                ("OPEN", "test container/state affordance"),
                ("ACTIVATE", "test state transition affordance"),
            )

        for action_name, reason in unary_order:
            if action_name not in available_actions:
                continue
            if _action_args(available_actions, action_name) != ("arg1",):
                continue
            for entity in ranked:
                signature = (action_name, entity.entity_id)
                if signature in self._attempted:
                    continue
                self._attempted.add(signature)
                if action_name == "TALK":
                    self._dialog_entity = entity.entity_id
                return self._decision(
                    {
                        "action": action_name,
                        "arg1": entity.entity_id,
                    },
                    reasoning=reason,
                )

        if (
            "USE" in available_actions
            and _action_args(available_actions, "USE")
            == ("arg1", "arg2")
            and self._pair_trials < self.max_pair_trials
        ):
            usable = [
                entity
                for entity in ranked
                if entity.inventory or entity.accessible
            ]
            pair_candidates = sorted(
                permutations(usable, 2),
                key=lambda pair: (
                    -max(
                        self._relevance(observation, pair[0]),
                        self._relevance(observation, pair[1]),
                    ),
                    pair[0].entity_id,
                    pair[1].entity_id,
                ),
            )
            for left, right in pair_candidates:
                signature = ("USE", left.entity_id, right.entity_id)
                if signature in self._attempted:
                    continue
                self._attempted.add(signature)
                self._pair_trials += 1
                return self._decision(
                    {
                        "action": "USE",
                        "arg1": left.entity_id,
                        "arg2": right.entity_id,
                    },
                    reasoning=(
                        "probe an untried binary interaction for observable effects"
                    ),
                )

        if "TELEPORT_TO_OBJECT" in available_actions:
            nearby = sorted(
                (
                    entity
                    for entity in current.values()
                    if entity.entity_id not in self._visited_objects
                    and not entity.accessible
                    and self._relevance(observation, entity) > 0
                ),
                key=lambda entity: (
                    -self._relevance(observation, entity),
                    entity.entity_id,
                ),
            )
            if nearby:
                entity = nearby[0]
                self._visited_objects.add(entity.entity_id)
                return self._decision(
                    {
                        "action": "TELEPORT_TO_OBJECT",
                        "arg1": entity.entity_id,
                    },
                    reasoning="move beside a goal-relevant uninspected entity",
                )

        if location is not None:
            self._visited_locations.add(location)
            return self._decision(
                {
                    "action": "TELEPORT_TO_LOCATION",
                    "arg1": location,
                },
                reasoning="continue environment coverage at an unexplored location",
            )

        fallback = (
            "DISCOVERY_FEED_GET_UPDATES"
            if "DISCOVERY_FEED_GET_UPDATES" in available_actions
            else next(iter(available_actions))
        )
        return self._decision(
            {"action": fallback},
            reasoning="fallback to a zero-argument observable action",
        )

    def _pursue_active(
        self,
        observation: dict[str, JSONValue],
        available_actions: dict[str, JSONValue],
        scientific_context: ScientificContext,
        current: dict[int, _Entity],
    ) -> PolicyDecision | None:
        assert self._active is not None
        prediction = next(
            (
                item
                for item in self._active.predictions
                if (
                    self._active.hypothesis_id,
                    item.control_entity_id,
                )
                not in self._completed_predictions
            ),
            None,
        )
        if prediction is None:
            self._active = None
            return None

        emit = (
            self._active
            if self._active.hypothesis_id not in self._emitted
            else None
        )

        subject = current.get(prediction.subject_entity_id)
        control = current.get(prediction.control_entity_id)

        if subject is None or not subject.accessible:
            if "TELEPORT_TO_OBJECT" in available_actions:
                return self._decision(
                    {
                        "action": "TELEPORT_TO_OBJECT",
                        "arg1": prediction.subject_entity_id,
                    },
                    reasoning="move to the predicted subject entity",
                    emit=emit,
                )

        if subject is not None and not subject.inventory:
            signature = ("PICKUP", prediction.subject_entity_id)
            if (
                "PICKUP" in available_actions
                and signature not in self._attempted
            ):
                self._attempted.add(signature)
                return self._decision(
                    {
                        "action": "PICKUP",
                        "arg1": prediction.subject_entity_id,
                    },
                    reasoning="make the predicted subject manipulable",
                    emit=emit,
                )

        if control is None or not control.accessible:
            if "TELEPORT_TO_OBJECT" in available_actions:
                return self._decision(
                    {
                        "action": "TELEPORT_TO_OBJECT",
                        "arg1": prediction.control_entity_id,
                    },
                    reasoning="move to the predicted control entity",
                    emit=emit,
                )

        put_key = (
            self._active.hypothesis_id,
            prediction.control_entity_id,
        )
        if (
            "PUT" in available_actions
            and subject is not None
            and subject.inventory
            and control is not None
            and put_key not in self._put_attempted
        ):
            self._put_attempted.add(put_key)
            current_value = self._latest_value(
                scientific_context.generic_evidence,
                prediction.control_entity_id,
                self._active.control_feature_key,
            )
            validate = None
            if (
                current_value is not None
                and abs(current_value - prediction.predicted_value)
                <= self.control_tolerance
            ):
                validate = self._active.hypothesis_id
                self._pending_validation = (
                    self._active.hypothesis_id,
                    prediction.control_entity_id,
                )
            return self._decision(
                {
                    "action": "PUT",
                    "arg1": prediction.subject_entity_id,
                    "arg2": prediction.control_entity_id,
                },
                reasoning="apply the predicted subject/control pairing",
                emit=emit,
                validate=validate,
            )

        if "TALK" in available_actions:
            self._dialog_entity = prediction.control_entity_id
            return self._decision(
                {
                    "action": "TALK",
                    "arg1": prediction.control_entity_id,
                },
                reasoning="inspect the public control interface before intervention",
                emit=emit,
            )

        return None

    def _dialog_decision(
        self,
        observation: dict[str, JSONValue],
        scientific_context: ScientificContext,
    ) -> PolicyDecision:
        options = _dialog_options(observation)
        if not options:
            return self._decision(
                {"chosen_dialog_option_int": 0},
                reasoning="advance a public dialog with its only available choice",
            )

        if self._active is not None and self._dialog_entity is not None:
            prediction = next(
                (
                    item
                    for item in self._active.predictions
                    if item.control_entity_id == self._dialog_entity
                    and (
                        self._active.hypothesis_id,
                        item.control_entity_id,
                    )
                    not in self._completed_predictions
                ),
                None,
            )
            if prediction is not None:
                current_value = self._latest_value(
                    scientific_context.generic_evidence,
                    prediction.control_entity_id,
                    self._active.control_feature_key,
                )
                if current_value is not None:
                    delta_options = self._numeric_dialog_options(options)
                    if delta_options:
                        best = min(
                            delta_options,
                            key=lambda item: abs(
                                current_value
                                + item[2]
                                - prediction.predicted_value
                            ),
                        )
                        next_value = current_value + best[2]
                        validate = None
                        if (
                            abs(
                                next_value
                                - prediction.predicted_value
                            )
                            <= self.control_tolerance
                        ):
                            validate = self._active.hypothesis_id
                            self._pending_validation = (
                                self._active.hypothesis_id,
                                prediction.control_entity_id,
                            )
                        return self._decision(
                            {
                                "chosen_dialog_option_int": best[0],
                            },
                            reasoning=(
                                "choose the public numeric control change that "
                                "most reduces prediction error"
                            ),
                            emit=(
                                self._active
                                if self._active.hypothesis_id
                                not in self._emitted
                                else None
                            ),
                            validate=validate,
                        )

        exit_options = [
            item
            for item in options
            if any(
                token in item[1].lower()
                for token in ("exit", "leave", "done", "close", "back")
            )
        ]
        chosen = exit_options[0] if exit_options else options[-1]
        return self._decision(
            {"chosen_dialog_option_int": chosen[0]},
            reasoning="leave a dialog after recording its public state",
        )

    def _resolve_pending(
        self,
        observation: dict[str, JSONValue],
        available_actions: dict[str, JSONValue],
    ) -> None:
        if self._pending_validation is None:
            return
        hypothesis_id, entity_id = self._pending_validation
        entity = _entities(observation).get(entity_id)
        if entity is not None and self._preferred_state(
            entity.name,
            available_actions,
        ):
            self._completed_predictions.add(
                (hypothesis_id, entity_id)
            )
        else:
            self._failed.add(hypothesis_id)
            if (
                self._active is not None
                and self._active.hypothesis_id == hypothesis_id
            ):
                self._active = None
        self._pending_validation = None

    def _discover_hypotheses(
        self,
        scientific_context: ScientificContext,
        available_actions: dict[str, JSONValue],
    ) -> tuple[GenericNumericHypothesis, ...]:
        evidence = scientific_context.generic_evidence
        if len(evidence) < 4:
            return ()

        grouped = self._control_groups(
            evidence,
            available_actions,
        )
        source = self._transfer_source(scientific_context)
        preferred_degree = None
        if source is not None:
            raw = _parameters(source).get("generic_degree")
            if type(raw) is int:
                preferred_degree = raw

        hypotheses: list[tuple[float, GenericNumericHypothesis]] = []
        for reference_ids, target_ids in grouped:
            if len(reference_ids) < 2 or not target_ids:
                continue
            pairings = {
                control_id: self._best_subject(
                    control_id,
                    evidence,
                )
                for control_id in (*reference_ids, *target_ids)
            }
            if any(
                pairings[control_id] is None
                for control_id in reference_ids
            ):
                continue

            input_keys = self._common_input_keys(
                tuple(
                    pairings[control_id]
                    for control_id in reference_ids
                    if pairings[control_id] is not None
                ),
                evidence,
            )
            output_keys = self._common_unary_keys(
                reference_ids,
                evidence,
            )
            for feature_key in input_keys:
                for control_key in output_keys:
                    points: list[tuple[float, float]] = []
                    source_ids: list[str] = []
                    complete = True
                    for control_id in reference_ids:
                        subject_id = pairings[control_id]
                        assert subject_id is not None
                        input_item = self._latest_evidence(
                            evidence,
                            subject_id,
                            feature_key,
                            binary=True,
                        )
                        output_item = self._latest_evidence(
                            evidence,
                            control_id,
                            control_key,
                            binary=False,
                        )
                        if input_item is None or output_item is None:
                            complete = False
                            break
                        points.append(
                            (input_item.value, output_item.value)
                        )
                        source_ids.extend(
                            (
                                input_item.evidence_id,
                                output_item.evidence_id,
                            )
                        )
                    if not complete:
                        continue

                    max_degree = min(
                        self.max_degree,
                        len(points) - 1,
                    )
                    fits: list[_Fit] = []
                    for degree in range(max_degree + 1):
                        fitted = _fit_polynomial(
                            tuple(points),
                            degree,
                        )
                        if fitted is None:
                            continue
                        coefficients, score = fitted
                        if preferred_degree == degree:
                            score -= 0.25
                        fits.append(
                            _Fit(
                                feature_key=feature_key,
                                control_feature_key=control_key,
                                degree=degree,
                                coefficients=coefficients,
                                score=score,
                                source_evidence_ids=tuple(
                                    dict.fromkeys(source_ids)
                                ),
                            )
                        )
                    if not fits:
                        continue
                    best = min(fits, key=lambda item: item.score)

                    predictions: list[GenericNumericPrediction] = []
                    for control_id in target_ids:
                        subject_id = pairings.get(control_id)
                        if subject_id is None:
                            continue
                        input_item = self._latest_evidence(
                            evidence,
                            subject_id,
                            feature_key,
                            binary=True,
                        )
                        if input_item is None:
                            continue
                        predictions.append(
                            GenericNumericPrediction(
                                subject_entity_id=subject_id,
                                control_entity_id=control_id,
                                predicted_value=_poly_value(
                                    best.coefficients,
                                    input_item.value,
                                ),
                                frozen_step=self._step,
                            )
                        )
                    if not predictions:
                        continue

                    identity = (
                        f"{feature_key}:{control_key}:{best.degree}:"
                        f"{best.coefficients}:"
                        f"{tuple((p.subject_entity_id, p.control_entity_id) for p in predictions)}"
                    )
                    hypothesis_id = (
                        "autonomous:"
                        + sha256(identity.encode()).hexdigest()[:20]
                    )
                    if (
                        hypothesis_id in self._failed
                        or hypothesis_id in self._emitted
                    ):
                        continue
                    hypothesis = GenericNumericHypothesis(
                        hypothesis_id=hypothesis_id,
                        transfer_key=_TRANSFER_KEY,
                        feature_key=feature_key,
                        control_feature_key=control_key,
                        polynomial_degree=best.degree,
                        coefficients=best.coefficients,
                        source_evidence_ids=best.source_evidence_ids,
                        predictions=tuple(predictions),
                        source_mechanism=(
                            source.ref if source is not None else None
                        ),
                    )
                    hypotheses.append((best.score, hypothesis))

        return tuple(
            hypothesis
            for _, hypothesis in sorted(
                hypotheses,
                key=lambda item: (
                    item[0],
                    item[1].hypothesis_id,
                ),
            )
        )

    def _control_groups(
        self,
        evidence: tuple[GenericScalarEvidence, ...],
        available_actions: dict[str, JSONValue],
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        unary_ids = {
            item.entity_ids[0]
            for item in evidence
            if len(item.entity_ids) == 1
        }
        groups: dict[str, list[int]] = {}
        for entity_id in unary_ids:
            label = self._labels.get(entity_id, "")
            base = _base_name(label)
            if base:
                groups.setdefault(base, []).append(entity_id)

        result: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for ids in groups.values():
            if len(ids) < 3:
                continue
            references = tuple(
                entity_id
                for entity_id in ids
                if self._preferred_state(
                    self._labels.get(entity_id, ""),
                    available_actions,
                )
            )
            targets = tuple(
                entity_id
                for entity_id in ids
                if entity_id not in references
            )
            if len(references) >= 2 and targets:
                result.append((references, targets))
        return tuple(result)

    def _preferred_state(
        self,
        label: str,
        available_actions: dict[str, JSONValue],
    ) -> bool:
        state = _state_text(label)
        if not state:
            return False
        stems: set[str] = {"success", "complete", "ready", "on"}
        if "ACTIVATE" in available_actions:
            stems.add("activat")
        if "OPEN" in available_actions:
            stems.add("open")
        return any(stem in state for stem in stems)

    def _best_subject(
        self,
        control_id: int,
        evidence: tuple[GenericScalarEvidence, ...],
    ) -> int | None:
        candidates = {
            item.entity_ids[-1]
            for item in evidence
            if len(item.entity_ids) >= 2
        }
        control_text = self._entity_text(control_id, evidence)
        ranked = sorted(
            (
                (
                    _similarity(
                        control_text,
                        self._entity_text(candidate, evidence),
                    ),
                    candidate,
                )
                for candidate in candidates
                if candidate != control_id
            ),
            reverse=True,
        )
        if not ranked or ranked[0][0] <= 0.0:
            return None
        return ranked[0][1]

    def _entity_text(
        self,
        entity_id: int,
        evidence: tuple[GenericScalarEvidence, ...],
    ) -> str:
        parts = [
            self._labels.get(entity_id, ""),
            self._descriptions.get(entity_id, ""),
        ]
        parts.extend(
            item.raw_text
            for item in evidence
            if len(item.entity_ids) == 1
            and item.entity_ids[0] == entity_id
        )
        return " ".join(parts)

    @staticmethod
    def _common_input_keys(
        subject_ids: tuple[int, ...],
        evidence: tuple[GenericScalarEvidence, ...],
    ) -> tuple[str, ...]:
        key_sets: list[set[str]] = []
        for subject_id in subject_ids:
            key_sets.append(
                {
                    item.feature_key
                    for item in evidence
                    if len(item.entity_ids) >= 2
                    and item.entity_ids[-1] == subject_id
                }
            )
        if not key_sets:
            return ()
        return tuple(sorted(set.intersection(*key_sets)))

    @staticmethod
    def _common_unary_keys(
        entity_ids: tuple[int, ...],
        evidence: tuple[GenericScalarEvidence, ...],
    ) -> tuple[str, ...]:
        key_sets: list[set[str]] = []
        for entity_id in entity_ids:
            key_sets.append(
                {
                    item.feature_key
                    for item in evidence
                    if item.entity_ids == (entity_id,)
                }
            )
        if not key_sets:
            return ()
        return tuple(sorted(set.intersection(*key_sets)))

    @staticmethod
    def _latest_evidence(
        evidence: tuple[GenericScalarEvidence, ...],
        entity_id: int,
        feature_key: str,
        *,
        binary: bool,
    ) -> GenericScalarEvidence | None:
        matches = [
            item
            for item in evidence
            if item.feature_key == feature_key
            and (
                (
                    binary
                    and len(item.entity_ids) >= 2
                    and item.entity_ids[-1] == entity_id
                )
                or (
                    not binary
                    and item.entity_ids == (entity_id,)
                )
            )
        ]
        return max(matches, key=lambda item: item.step) if matches else None

    @classmethod
    def _latest_value(
        cls,
        evidence: tuple[GenericScalarEvidence, ...],
        entity_id: int,
        feature_key: str,
    ) -> float | None:
        item = cls._latest_evidence(
            evidence,
            entity_id,
            feature_key,
            binary=False,
        )
        return item.value if item is not None else None

    @staticmethod
    def _numeric_dialog_options(
        options: tuple[tuple[int, str], ...],
    ) -> tuple[tuple[int, str, float], ...]:
        found: list[tuple[int, str, float]] = []
        for index, label in options:
            match = _NUMBER.search(label)
            if match is None:
                continue
            value = float(match.group(0).replace(",", ""))
            lowered = label.lower()
            if any(
                token in lowered
                for token in ("decrease", "lower", "down", "minus", "subtract")
            ):
                value = -abs(value)
            elif any(
                token in lowered
                for token in ("increase", "raise", "up", "plus", "add")
            ):
                value = abs(value)
            else:
                continue
            found.append((index, label, value))
        return tuple(found)

    @staticmethod
    def _transfer_source(
        context: ScientificContext,
    ) -> MechanismRecord | None:
        for mechanism in context.transfer_candidates:
            parameters = _parameters(mechanism)
            if parameters.get("transfer_key") == _TRANSFER_KEY:
                return mechanism
        return None

    @staticmethod
    def _relevance(
        observation: dict[str, JSONValue],
        entity: _Entity,
    ) -> float:
        goal = set(_tokens(_task_text(observation)))
        if not goal:
            return 0.0
        entity_tokens = set(
            _tokens(entity.name + " " + entity.description)
        )
        if not entity_tokens:
            return 0.0
        return len(goal & entity_tokens) / len(entity_tokens)

    def _next_location(
        self,
        observation: dict[str, JSONValue],
        teleport_locations: dict[str, JSONValue],
    ) -> str | None:
        candidates = [
            location
            for location in teleport_locations
            if location not in self._visited_locations
        ]
        if not candidates:
            return None
        goal = set(_tokens(_task_text(observation)))

        def score(location: str) -> tuple[float, str]:
            tokens = set(_tokens(location))
            overlap = (
                len(tokens & goal) / len(tokens)
                if tokens
                else 0.0
            )
            return (-overlap, location)

        return min(candidates, key=score)


def create_policy(
    config: dict[str, JSONValue],
) -> AutonomousScientistPolicy:
    return AutonomousScientistPolicy(config)
