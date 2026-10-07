from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from itertools import permutations

from ecsa.mechanisms import MechanismVersionRef


_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_NUMBER = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d+)?"
)
_STATE = re.compile(r"\(([^()]*)\)")
TRANSFER_KEY = "scalar-interaction-to-control"
GENERIC_DISCOVERY_ASSUMPTIONS = (
    "public-observation-only",
    "generic-scalar-discovery",
)

class ActionRole(StrEnum):
    VISIT_LOCATION = "visit_location"
    VISIT_ENTITY = "visit_entity"
    OBSERVE_UNARY = "observe_unary"
    ACQUIRE = "acquire"
    STATE_CHANGE = "state_change"
    PROBE_BINARY = "probe_binary"
    PLACE = "place"
    PASSIVE = "passive"


@dataclass(frozen=True)
class PublicEntityView:
    entity_id: int
    name: str
    description: str
    accessible: bool
    inventory: bool

    def __post_init__(self) -> None:
        if type(self.entity_id) is not int or self.entity_id < 0:
            raise ValueError("entity_id must be a nonnegative integer")
        if not self.name:
            raise ValueError("entity name is required")


@dataclass(frozen=True)
class ActionAffordance:
    action_id: str
    role: ActionRole
    arity: int

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id is required")
        if not isinstance(self.role, ActionRole):
            raise ValueError("typed action role required")
        if type(self.arity) is not int or self.arity < 0 or self.arity > 2:
            raise ValueError("action arity must be 0..2")


@dataclass(frozen=True)
class DialogOption:
    option_id: int
    label: str

    def __post_init__(self) -> None:
        if type(self.option_id) is not int or self.option_id < 0:
            raise ValueError("dialog option id must be nonnegative")
        if not self.label:
            raise ValueError("dialog option label is required")


@dataclass(frozen=True)
class AutonomousWorldView:
    goal_text: str
    entities: tuple[PublicEntityView, ...]
    actions: tuple[ActionAffordance, ...]
    locations: tuple[str, ...]
    in_dialog: bool
    dialog_options: tuple[DialogOption, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entities, tuple):
            raise ValueError("entities must be immutable")
        if not isinstance(self.actions, tuple):
            raise ValueError("actions must be immutable")
        if not isinstance(self.locations, tuple):
            raise ValueError("locations must be immutable")
        if not isinstance(self.dialog_options, tuple):
            raise ValueError("dialog options must be immutable")


@dataclass(frozen=True)
class TransferMechanismView:
    ref: MechanismVersionRef
    transfer_key: str
    preferred_degree: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, MechanismVersionRef):
            raise ValueError("typed mechanism reference required")
        if not self.transfer_key:
            raise ValueError("transfer_key is required")
        if (
            self.preferred_degree is not None
            and (
                type(self.preferred_degree) is not int
                or not 0 <= self.preferred_degree <= 4
            )
        ):
            raise ValueError("preferred_degree must be None or in 0..4")


@dataclass(frozen=True)
class GenericScalarEvidence:
    evidence_id: str
    step: int
    action_name: str
    entity_ids: tuple[int, ...]
    feature_key: str
    value: float
    raw_text: str

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.action_name or not self.feature_key:
            raise ValueError("generic evidence ids/action/feature are required")
        if type(self.step) is not int or self.step < 0:
            raise ValueError("generic evidence step must be nonnegative")
        if not isinstance(self.entity_ids, tuple) or not all(
            type(value) is int and value >= 0
            for value in self.entity_ids
        ):
            raise ValueError("generic evidence entity ids must be nonnegative integers")
        if not math.isfinite(self.value):
            raise ValueError("generic evidence value must be finite")
        if not self.raw_text:
            raise ValueError("generic evidence raw_text is required")


@dataclass(frozen=True)
class GenericNumericPrediction:
    subject_entity_id: int
    control_entity_id: int
    predicted_value: float
    frozen_step: int

    def __post_init__(self) -> None:
        if (
            type(self.subject_entity_id) is not int
            or self.subject_entity_id < 0
            or type(self.control_entity_id) is not int
            or self.control_entity_id < 0
        ):
            raise ValueError("generic prediction entity ids must be nonnegative")
        if not math.isfinite(self.predicted_value):
            raise ValueError("generic predicted value must be finite")
        if type(self.frozen_step) is not int or self.frozen_step < 0:
            raise ValueError("generic frozen_step must be nonnegative")


@dataclass(frozen=True)
class GenericNumericHypothesis:
    hypothesis_id: str
    transfer_key: str
    feature_key: str
    control_feature_key: str
    polynomial_degree: int
    coefficients: tuple[float, ...]
    source_evidence_ids: tuple[str, ...]
    predictions: tuple[GenericNumericPrediction, ...]
    source_mechanism: MechanismVersionRef | None = None

    def __post_init__(self) -> None:
        if (
            not self.hypothesis_id
            or not self.transfer_key
            or not self.feature_key
            or not self.control_feature_key
        ):
            raise ValueError("generic hypothesis identifiers are required")
        if (
            type(self.polynomial_degree) is not int
            or self.polynomial_degree < 0
            or self.polynomial_degree > 4
        ):
            raise ValueError("generic polynomial_degree must be in 0..4")
        if (
            not isinstance(self.coefficients, tuple)
            or len(self.coefficients) != self.polynomial_degree + 1
            or not all(math.isfinite(value) for value in self.coefficients)
        ):
            raise ValueError("generic coefficients must match polynomial degree")
        if (
            not isinstance(self.source_evidence_ids, tuple)
            or not self.source_evidence_ids
            or not all(
                isinstance(value, str) and value
                for value in self.source_evidence_ids
            )
            or len(set(self.source_evidence_ids))
            != len(self.source_evidence_ids)
        ):
            raise ValueError("generic source evidence ids must be unique strings")
        if (
            not isinstance(self.predictions, tuple)
            or not self.predictions
            or not all(
                isinstance(value, GenericNumericPrediction)
                for value in self.predictions
            )
        ):
            raise ValueError("generic hypothesis requires typed predictions")
        if self.source_mechanism is not None and not isinstance(
            self.source_mechanism, MechanismVersionRef
        ):
            raise ValueError("generic source_mechanism must be a MechanismVersionRef")


@dataclass(frozen=True)
class AutonomousDecision:
    action_id: str | None = None
    entity_args: tuple[int, ...] = ()
    location_arg: str | None = None
    dialog_option: int | None = None
    reasoning: str | None = None
    hypotheses: tuple[GenericNumericHypothesis, ...] = ()
    validation_hypothesis_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ordinary = self.action_id is not None
        dialog = self.dialog_option is not None
        if ordinary == dialog:
            raise ValueError("decision must choose exactly one action or dialog option")
        if not isinstance(self.entity_args, tuple) or len(self.entity_args) > 2:
            raise ValueError("entity_args must be an immutable tuple of arity <= 2")
        if self.location_arg is not None and not self.location_arg:
            raise ValueError("location_arg must be nonempty when present")


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
    return " ".join(_STATE.findall(value)).lower()


def _similarity(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    shared_numbers = {
        token for token in left_tokens & right_tokens if token.isdigit()
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


def _solve(
    matrix: list[list[float]],
    vector: list[float],
) -> tuple[float, ...] | None:
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
                matrix[row][column] += powers[row] * powers[column]
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


class AutonomousScientist:
    """Environment-neutral evidence-driven controller."""

    def __init__(
        self,
        *,
        max_polynomial_degree: int = 2,
        max_pair_trials: int = 160,
        control_tolerance: float = 1.5,
    ) -> None:
        if not 0 <= max_polynomial_degree <= 4:
            raise ValueError("max_polynomial_degree must be in 0..4")
        if max_pair_trials < 1:
            raise ValueError("max_pair_trials must be positive")
        if control_tolerance <= 0:
            raise ValueError("control_tolerance must be positive")
        self.max_degree = max_polynomial_degree
        self.max_pair_trials = max_pair_trials
        self.control_tolerance = float(control_tolerance)

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
        self._historical_preferred: set[int] = set()
        self._placed_predictions: set[tuple[str, int]] = set()

    def decide(
        self,
        view: AutonomousWorldView,
        evidence: tuple[GenericScalarEvidence, ...],
        transfer_candidates: tuple[TransferMechanismView, ...],
    ) -> AutonomousDecision:
        entity_map = {
            entity.entity_id: entity
            for entity in view.entities
        }
        for entity in view.entities:
            self._labels[entity.entity_id] = entity.name
            self._descriptions[entity.entity_id] = entity.description
            if self._preferred_state(entity.name):
                self._historical_preferred.add(entity.entity_id)

        self._resolve_pending(view)

        if view.in_dialog:
            decision = self._dialog_decision(view, evidence)
            self._step += 1
            return decision

        if self._active is None:
            candidates = self._discover_hypotheses(
                evidence,
                transfer_candidates,
                view,
            )
            if candidates:
                self._active = candidates[0]

        if self._active is not None:
            decision = self._pursue_active(
                view,
                evidence,
                entity_map,
            )
            if decision is not None:
                self._step += 1
                return decision

        decision = self._explore(
            view,
            transfer_candidates,
            entity_map,
        )
        self._step += 1
        return decision

    def _decision(
        self,
        *,
        action_id: str | None = None,
        entity_args: tuple[int, ...] = (),
        location_arg: str | None = None,
        dialog_option: int | None = None,
        reasoning: str,
        emit: GenericNumericHypothesis | None = None,
        validate: str | None = None,
    ) -> AutonomousDecision:
        hypotheses = (
            (emit,)
            if emit is not None
            and emit.hypothesis_id not in self._emitted
            else ()
        )
        if hypotheses:
            self._emitted.add(emit.hypothesis_id)
        return AutonomousDecision(
            action_id=action_id,
            entity_args=entity_args,
            location_arg=location_arg,
            dialog_option=dialog_option,
            reasoning=reasoning,
            hypotheses=hypotheses,
            validation_hypothesis_ids=(
                (validate,) if validate is not None else ()
            ),
        )

    @staticmethod
    def _by_role(
        view: AutonomousWorldView,
        role: ActionRole,
        *,
        arity: int | None = None,
    ) -> tuple[ActionAffordance, ...]:
        return tuple(
            item
            for item in view.actions
            if item.role is role
            and (arity is None or item.arity == arity)
        )

    def _explore(
        self,
        view: AutonomousWorldView,
        transfer_candidates: tuple[TransferMechanismView, ...],
        current: dict[int, PublicEntityView],
    ) -> AutonomousDecision:
        structural_prior = (
            self._transfer_source(transfer_candidates) is not None
        )

        location = self._next_location(view)
        if location is not None and not any(
            entity.accessible and self._relevance(view, entity) > 0
            for entity in current.values()
        ):
            action = self._first_role(view, ActionRole.VISIT_LOCATION, 1)
            if action is not None:
                self._visited_locations.add(location)
                return self._decision(
                    action_id=action.action_id,
                    location_arg=location,
                    reasoning="visit the most goal-relevant unexplored location",
                )

        ranked = sorted(
            (
                entity
                for entity in current.values()
                if entity.accessible
            ),
            key=lambda entity: (
                -self._relevance(view, entity),
                entity.entity_id,
            ),
        )

        role_order = [
            ActionRole.OBSERVE_UNARY,
            ActionRole.ACQUIRE,
        ]
        if not structural_prior:
            role_order.append(ActionRole.STATE_CHANGE)

        for role in role_order:
            for action in self._by_role(view, role, arity=1):
                for entity in ranked:
                    signature = (action.action_id, entity.entity_id)
                    if signature in self._attempted:
                        continue
                    self._attempted.add(signature)
                    if role is ActionRole.OBSERVE_UNARY:
                        self._dialog_entity = entity.entity_id
                    return self._decision(
                        action_id=action.action_id,
                        entity_args=(entity.entity_id,),
                        reasoning=(
                            "probe a previously untried unary affordance"
                        ),
                    )

        binary = self._first_role(view, ActionRole.PROBE_BINARY, 2)
        if (
            binary is not None
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
                        self._relevance(view, pair[0]),
                        self._relevance(view, pair[1]),
                    ),
                    pair[0].entity_id,
                    pair[1].entity_id,
                ),
            )
            for left, right in pair_candidates:
                signature = (
                    binary.action_id,
                    left.entity_id,
                    right.entity_id,
                )
                if signature in self._attempted:
                    continue
                self._attempted.add(signature)
                self._pair_trials += 1
                return self._decision(
                    action_id=binary.action_id,
                    entity_args=(
                        left.entity_id,
                        right.entity_id,
                    ),
                    reasoning=(
                        "probe an untried binary interaction for observable effects"
                    ),
                )

        visit_entity = self._first_role(
            view,
            ActionRole.VISIT_ENTITY,
            1,
        )
        if visit_entity is not None:
            nearby = sorted(
                (
                    entity
                    for entity in current.values()
                    if entity.entity_id not in self._visited_objects
                    and not entity.accessible
                    and self._relevance(view, entity) > 0
                ),
                key=lambda entity: (
                    -self._relevance(view, entity),
                    entity.entity_id,
                ),
            )
            if nearby:
                entity = nearby[0]
                self._visited_objects.add(entity.entity_id)
                return self._decision(
                    action_id=visit_entity.action_id,
                    entity_args=(entity.entity_id,),
                    reasoning="move beside a goal-relevant uninspected entity",
                )

        if location is not None:
            action = self._first_role(view, ActionRole.VISIT_LOCATION, 1)
            if action is not None:
                self._visited_locations.add(location)
                return self._decision(
                    action_id=action.action_id,
                    location_arg=location,
                    reasoning="continue environment coverage",
                )

        passive = self._first_role(view, ActionRole.PASSIVE, 0)
        if passive is None:
            passive = next(
                (
                    action
                    for action in view.actions
                    if action.arity == 0
                ),
                None,
            )
        if passive is None:
            raise RuntimeError("no safe generic fallback action is available")
        return self._decision(
            action_id=passive.action_id,
            reasoning="perform a passive observation step",
        )

    def _pursue_active(
        self,
        view: AutonomousWorldView,
        evidence: tuple[GenericScalarEvidence, ...],
        current: dict[int, PublicEntityView],
    ) -> AutonomousDecision | None:
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
        visit = self._first_role(view, ActionRole.VISIT_ENTITY, 1)

        if subject is None or not subject.accessible:
            if visit is not None:
                return self._decision(
                    action_id=visit.action_id,
                    entity_args=(prediction.subject_entity_id,),
                    reasoning="move to the predicted subject entity",
                    emit=emit,
                )

        placed_key = (
            self._active.hypothesis_id,
            prediction.control_entity_id,
        )
        acquire = self._first_role(view, ActionRole.ACQUIRE, 1)
        if (
            placed_key not in self._placed_predictions
            and acquire is not None
            and subject is not None
            and not subject.inventory
        ):
            signature = (
                acquire.action_id,
                prediction.subject_entity_id,
            )
            if signature not in self._attempted:
                self._attempted.add(signature)
                return self._decision(
                    action_id=acquire.action_id,
                    entity_args=(prediction.subject_entity_id,),
                    reasoning="make the predicted subject manipulable",
                    emit=emit,
                )

        if control is None or not control.accessible:
            if visit is not None:
                return self._decision(
                    action_id=visit.action_id,
                    entity_args=(prediction.control_entity_id,),
                    reasoning="move to the predicted control entity",
                    emit=emit,
                )

        place = self._first_role(view, ActionRole.PLACE, 2)
        put_key = placed_key
        if (
            place is not None
            and placed_key not in self._placed_predictions
            and subject is not None
            and subject.inventory
            and control is not None
            and put_key not in self._put_attempted
        ):
            self._put_attempted.add(put_key)
            self._placed_predictions.add(placed_key)
            for acquire_action in self._by_role(
                view,
                ActionRole.ACQUIRE,
                arity=1,
            ):
                self._attempted.discard(
                    (
                        acquire_action.action_id,
                        prediction.subject_entity_id,
                    )
                )
            return self._decision(
                action_id=place.action_id,
                entity_args=(
                    prediction.subject_entity_id,
                    prediction.control_entity_id,
                ),
                reasoning=(
                    "apply the predicted subject/control pairing before "
                    "prospective control validation"
                ),
                emit=emit,
            )

        observe = self._first_role(
            view,
            ActionRole.OBSERVE_UNARY,
            1,
        )
        if observe is not None:
            self._dialog_entity = prediction.control_entity_id
            return self._decision(
                action_id=observe.action_id,
                entity_args=(prediction.control_entity_id,),
                reasoning="inspect the public control interface",
                emit=emit,
            )
        return None

    def _dialog_decision(
        self,
        view: AutonomousWorldView,
        evidence: tuple[GenericScalarEvidence, ...],
    ) -> AutonomousDecision:
        if not view.dialog_options:
            raise RuntimeError("dialog mode has no public options")

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
                    evidence,
                    prediction.control_entity_id,
                    self._active.control_feature_key,
                )
                if current_value is not None:
                    delta_options = self._numeric_dialog_options(
                        view.dialog_options
                    )
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
                            dialog_option=best[0],
                            reasoning=(
                                "choose the numeric control change that most "
                                "reduces prediction error"
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
            for item in view.dialog_options
            if any(
                token in item.label.lower()
                for token in (
                    "exit",
                    "leave",
                    "done",
                    "close",
                    "back",
                )
            )
        ]
        chosen = (
            exit_options[0]
            if exit_options
            else view.dialog_options[-1]
        )
        return self._decision(
            dialog_option=chosen.option_id,
            reasoning="leave the interface after recording public state",
        )

    def _resolve_pending(self, view: AutonomousWorldView) -> None:
        if self._pending_validation is None:
            return
        hypothesis_id, entity_id = self._pending_validation
        entity = next(
            (
                item
                for item in view.entities
                if item.entity_id == entity_id
            ),
            None,
        )
        if entity is not None and self._preferred_state(entity.name):
            self._completed_predictions.add((hypothesis_id, entity_id))
        else:
            self._failed.add(hypothesis_id)
            if (
                self._active is not None
                and self._active.hypothesis_id == hypothesis_id
            ):
                self._active = None
        self._placed_predictions.discard((hypothesis_id, entity_id))
        self._pending_validation = None

    def _discover_hypotheses(
        self,
        evidence: tuple[GenericScalarEvidence, ...],
        transfer_candidates: tuple[TransferMechanismView, ...],
        view: AutonomousWorldView,
    ) -> tuple[GenericNumericHypothesis, ...]:
        if len(evidence) < 4:
            return ()

        source = self._transfer_source(transfer_candidates)
        preferred_degree = (
            source.preferred_degree
            if source is not None
            else None
        )

        hypotheses: list[tuple[float, GenericNumericHypothesis]] = []
        for reference_ids, target_ids in self._control_groups(evidence):
            pairings = {
                control_id: self._best_subject(control_id, evidence)
                for control_id in (*reference_ids, *target_ids)
            }
            if any(
                pairings[control_id] is None
                for control_id in reference_ids
            ):
                continue

            reference_subjects = tuple(
                pairings[control_id]
                for control_id in reference_ids
                if pairings[control_id] is not None
            )
            input_keys = self._common_input_keys(
                reference_subjects,
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
                        points.append((input_item.value, output_item.value))
                        source_ids.extend(
                            (
                                input_item.evidence_id,
                                output_item.evidence_id,
                            )
                        )
                    if not complete:
                        continue

                    fits: list[_Fit] = []
                    for degree in range(
                        min(self.max_degree, len(points) - 1) + 1
                    ):
                        fitted = _fit_polynomial(
                            tuple(points),
                            degree,
                        )
                        if fitted is None:
                            continue
                        coefficients, score = fitted
                        score += self._control_feature_penalty(
                            reference_ids,
                            control_key,
                            evidence,
                        )
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
                    hypotheses.append(
                        (
                            best.score,
                            GenericNumericHypothesis(
                                hypothesis_id=hypothesis_id,
                                transfer_key=TRANSFER_KEY,
                                feature_key=feature_key,
                                control_feature_key=control_key,
                                polynomial_degree=best.degree,
                                coefficients=best.coefficients,
                                source_evidence_ids=best.source_evidence_ids,
                                predictions=tuple(predictions),
                                source_mechanism=(
                                    source.ref
                                    if source is not None
                                    else None
                                ),
                            ),
                        )
                    )
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
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        unary_ids = {
            item.entity_ids[0]
            for item in evidence
            if len(item.entity_ids) == 1
        }
        groups: dict[str, list[int]] = {}
        for entity_id in unary_ids:
            base = _base_name(self._labels.get(entity_id, ""))
            if base:
                groups.setdefault(base, []).append(entity_id)

        result: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for ids in groups.values():
            if len(ids) < 3:
                continue
            references = tuple(
                entity_id
                for entity_id in ids
                if entity_id in self._historical_preferred
            )
            targets = tuple(
                entity_id
                for entity_id in ids
                if entity_id not in references
            )
            if len(references) >= 2 and targets:
                result.append((references, targets))
        return tuple(result)

    @staticmethod
    def _preferred_state(label: str) -> bool:
        state = _state_text(label)
        if not state:
            return False
        words = _tokens(state)
        return any(
            word in {"success", "successful", "complete", "completed", "ready", "open", "on"}
            or word.startswith("activat")
            for word in words
        )

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

    @classmethod
    def _control_feature_penalty(
        cls,
        entity_ids: tuple[int, ...],
        feature_key: str,
        evidence: tuple[GenericScalarEvidence, ...],
    ) -> float:
        samples = [
            item
            for item in evidence
            if item.feature_key == feature_key
            and item.entity_ids
            and item.entity_ids[0] in entity_ids
        ]
        text = " ".join(item.raw_text.lower() for item in samples)
        values = {round(item.value, 12) for item in samples}

        penalty = 0.0
        if any(
            token in text
            for token in (
                "current",
                "present",
                "setting",
                "value",
                "level",
                "state",
            )
        ):
            penalty -= 0.5
        if any(
            token in text
            for token in (
                "range",
                "minimum",
                "maximum",
                " min ",
                " max ",
                "allowable",
                "allowed",
            )
        ):
            penalty += 0.75
        if len(values) <= 1:
            penalty += 0.5
        return penalty

    @staticmethod
    def _common_input_keys(
        subject_ids: tuple[int, ...],
        evidence: tuple[GenericScalarEvidence, ...],
    ) -> tuple[str, ...]:
        key_sets = [
            {
                item.feature_key
                for item in evidence
                if len(item.entity_ids) >= 2
                and item.entity_ids[-1] == subject_id
            }
            for subject_id in subject_ids
        ]
        if not key_sets:
            return ()
        return tuple(sorted(set.intersection(*key_sets)))

    @staticmethod
    def _common_unary_keys(
        entity_ids: tuple[int, ...],
        evidence: tuple[GenericScalarEvidence, ...],
    ) -> tuple[str, ...]:
        key_sets = [
            {
                item.feature_key
                for item in evidence
                if item.entity_ids == (entity_id,)
            }
            for entity_id in entity_ids
        ]
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
        options: tuple[DialogOption, ...],
    ) -> tuple[tuple[int, str, float], ...]:
        found: list[tuple[int, str, float]] = []
        for option in options:
            match = _NUMBER.search(option.label)
            if match is None:
                continue
            value = float(match.group(0).replace(",", ""))
            lowered = option.label.lower()
            if any(
                token in lowered
                for token in (
                    "decrease",
                    "lower",
                    "down",
                    "minus",
                    "subtract",
                )
            ):
                value = -abs(value)
            elif any(
                token in lowered
                for token in (
                    "increase",
                    "raise",
                    "up",
                    "plus",
                    "add",
                )
            ):
                value = abs(value)
            else:
                continue
            found.append((option.option_id, option.label, value))
        return tuple(found)

    @staticmethod
    def _transfer_source(
        transfer_candidates: tuple[TransferMechanismView, ...],
    ) -> MechanismRecord | None:
        for mechanism in transfer_candidates:
            if (
                _parameter_map(mechanism).get("transfer_key")
                == TRANSFER_KEY
            ):
                return mechanism
        return None

    @staticmethod
    def _relevance(
        view: AutonomousWorldView,
        entity: PublicEntityView,
    ) -> float:
        goal = set(_tokens(view.goal_text))
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
        view: AutonomousWorldView,
    ) -> str | None:
        candidates = [
            location
            for location in view.locations
            if location not in self._visited_locations
        ]
        if not candidates:
            return None
        goal = set(_tokens(view.goal_text))

        def score(location: str) -> tuple[float, str]:
            tokens = set(_tokens(location))
            overlap = (
                len(tokens & goal) / len(tokens)
                if tokens
                else 0.0
            )
            return (-overlap, location)

        return min(candidates, key=score)

    @staticmethod
    def _first_role(
        view: AutonomousWorldView,
        role: ActionRole,
        arity: int,
    ) -> ActionAffordance | None:
        return next(
            (
                item
                for item in view.actions
                if item.role is role and item.arity == arity
            ),
            None,
        )
