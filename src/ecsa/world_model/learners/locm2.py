from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from typing import Hashable, Iterable

from .base import (
    AcquisitionTrace,
    ActionModelProposal,
    LearnerFailure,
)
from ..hypotheses import (
    ActionSchemaHypothesis,
    EntityTypeHypothesis,
    HypothesisStatus,
    PredicateHypothesis,
    WorldContractHypothesis,
)


@dataclass(frozen=True, order=True)
class TransitionRef:
    schema_id: str
    parameter_index: int

    def __post_init__(self) -> None:
        if not self.schema_id:
            raise ValueError("transition schema_id is required")
        if type(self.parameter_index) is not int or self.parameter_index < 0:
            raise ValueError("transition parameter_index must be nonnegative")


@dataclass(frozen=True)
class Locm2SortAnalysis:
    sort_id: str
    member_refs: tuple[str, ...]
    all_transitions: tuple[TransitionRef, ...]
    machine_transition_sets: tuple[tuple[TransitionRef, ...], ...]


@dataclass(frozen=True)
class Locm2Analysis:
    sorts: tuple[Locm2SortAnalysis, ...]


@dataclass(frozen=True)
class _SortWork:
    sort_index: int
    member_refs: tuple[str, ...]
    all_transitions: frozenset[TransitionRef]
    pairs: frozenset[tuple[TransitionRef, TransitionRef]]
    sequences: tuple[tuple[TransitionRef, ...], ...]
    machine_sets: tuple[frozenset[TransitionRef], ...]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[Hashable, Hashable] = {}

    def add(self, value: Hashable) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: Hashable) -> Hashable:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: Hashable, right: Hashable) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if repr(left_root) <= repr(right_root):
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _evidence_ids(traces: tuple[AcquisitionTrace, ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            step.evidence_id
            for trace in traces
            for step in trace.steps
        )
    )


def _state_union(
    transitions: frozenset[TransitionRef],
    pairs: frozenset[tuple[TransitionRef, TransitionRef]],
) -> _UnionFind:
    union = _UnionFind()
    for transition in transitions:
        union.add((transition, "start"))
        union.add((transition, "end"))
    for left, right in pairs:
        if left in transitions and right in transitions:
            union.union((left, "end"), (right, "start"))
    return union


def _predicted_pairs(
    transitions: frozenset[TransitionRef],
    pairs: frozenset[tuple[TransitionRef, TransitionRef]],
) -> frozenset[tuple[TransitionRef, TransitionRef]]:
    union = _state_union(transitions, pairs)
    return frozenset(
        (left, right)
        for left in transitions
        for right in transitions
        if union.find((left, "end"))
        == union.find((right, "start"))
    )


def _well_formed(
    transitions: frozenset[TransitionRef],
    pairs: frozenset[tuple[TransitionRef, TransitionRef]],
) -> bool:
    restricted = frozenset(
        (left, right)
        for left, right in pairs
        if left in transitions and right in transitions
    )
    return _predicted_pairs(transitions, restricted) == restricted


def _valid_subset(
    transitions: frozenset[TransitionRef],
    pairs: frozenset[tuple[TransitionRef, TransitionRef]],
    sequences: tuple[tuple[TransitionRef, ...], ...],
) -> bool:
    restricted = frozenset(
        (left, right)
        for left, right in pairs
        if left in transitions and right in transitions
    )
    if not _well_formed(transitions, pairs):
        return False

    union = _state_union(transitions, restricted)
    for sequence in sequences:
        projected = tuple(
            transition
            for transition in sequence
            if transition in transitions
        )
        for left, right in zip(projected, projected[1:]):
            if (
                union.find((left, "end"))
                != union.find((right, "start"))
            ):
                return False
    return True


def _holes(
    transitions: frozenset[TransitionRef],
    pairs: frozenset[tuple[TransitionRef, TransitionRef]],
) -> tuple[frozenset[TransitionRef], ...]:
    predicted = _predicted_pairs(transitions, pairs)
    missing = predicted - pairs
    unique = {
        frozenset((left, right))
        for left, right in missing
    }
    return tuple(
        sorted(
            unique,
            key=lambda values: (
                len(values),
                tuple(sorted(values)),
            ),
        )
    )


def _select_transition_sets(
    transitions: frozenset[TransitionRef],
    pairs: frozenset[tuple[TransitionRef, TransitionRef]],
    sequences: tuple[tuple[TransitionRef, ...], ...],
) -> tuple[frozenset[TransitionRef], ...]:
    if not transitions:
        return ()

    selected: list[frozenset[TransitionRef]] = []
    for hole in _holes(transitions, pairs):
        if any(hole <= candidate for candidate in selected):
            continue

        remaining = tuple(sorted(transitions - hole))
        found: frozenset[TransitionRef] | None = None
        for extra_count in range(0, len(remaining) + 1):
            for extras in combinations(remaining, extra_count):
                candidate = frozenset((*hole, *extras))
                if candidate == transitions:
                    continue
                if _valid_subset(candidate, pairs, sequences):
                    found = candidate
                    break
            if found is not None:
                break
        if found is not None:
            selected.append(found)

    selected = [
        candidate
        for candidate in selected
        if not any(
            candidate < other
            for other in selected
        )
    ]
    selected.append(transitions)
    unique = {
        candidate
        for candidate in selected
    }
    return tuple(
        sorted(
            unique,
            key=lambda values: (
                len(values) == len(transitions),
                len(values),
                tuple(sorted(values)),
            ),
        )
    )


class Locm2Learner:
    learner_id = "locm2"

    def __init__(self, *, max_transitions_per_sort: int = 16) -> None:
        if (
            type(max_transitions_per_sort) is not int
            or max_transitions_per_sort < 2
        ):
            raise ValueError(
                "max_transitions_per_sort must be an integer >= 2"
            )
        self.max_transitions_per_sort = max_transitions_per_sort

    def analyze(
        self,
        traces: tuple[AcquisitionTrace, ...],
    ) -> Locm2Analysis:
        works, _ = self._analyze(traces)
        return Locm2Analysis(
            sorts=tuple(
                Locm2SortAnalysis(
                    sort_id=f"sort-{work.sort_index}",
                    member_refs=work.member_refs,
                    all_transitions=tuple(
                        sorted(work.all_transitions)
                    ),
                    machine_transition_sets=tuple(
                        tuple(sorted(machine))
                        for machine in work.machine_sets
                    ),
                )
                for work in works
            )
        )

    def update(
        self,
        traces: tuple[AcquisitionTrace, ...],
        current_contracts: tuple[WorldContractHypothesis, ...],
    ) -> ActionModelProposal | LearnerFailure:
        del current_contracts
        try:
            works, position_sort = self._analyze(traces)
            return self._translate(
                traces=traces,
                works=works,
                position_sort=position_sort,
            )
        except Exception as exc:
            return LearnerFailure(
                self.learner_id,
                f"{type(exc).__name__}: {exc}",
            )

    def _analyze(
        self,
        traces: tuple[AcquisitionTrace, ...],
    ) -> tuple[
        tuple[_SortWork, ...],
        dict[TransitionRef, int],
    ]:
        if not traces:
            raise ValueError("at least one acquisition trace is required")

        position_union = _UnionFind()
        object_positions: dict[str, set[TransitionRef]] = defaultdict(set)
        action_arities: dict[str, int] = {}

        for trace in traces:
            for step in trace.steps:
                arity = len(step.object_refs)
                previous = action_arities.setdefault(
                    step.schema_id,
                    arity,
                )
                if previous != arity:
                    raise ValueError(
                        f"inconsistent arity for action {step.schema_id}"
                    )
                for index, object_ref in enumerate(step.object_refs):
                    position = TransitionRef(step.schema_id, index)
                    position_union.add(position)
                    object_positions[object_ref].add(position)

        for positions in object_positions.values():
            ordered = tuple(sorted(positions))
            if not ordered:
                continue
            first = ordered[0]
            for position in ordered[1:]:
                position_union.union(first, position)

        components: dict[Hashable, set[TransitionRef]] = defaultdict(set)
        for position in tuple(position_union.parent):
            components[position_union.find(position)].add(position)
        ordered_components = tuple(
            sorted(
                components.values(),
                key=lambda values: tuple(sorted(values)),
            )
        )
        position_sort = {
            position: index
            for index, values in enumerate(ordered_components)
            for position in values
        }

        sort_members: dict[int, set[str]] = defaultdict(set)
        for object_ref, positions in object_positions.items():
            sort_ids = {
                position_sort[position]
                for position in positions
            }
            if len(sort_ids) != 1:
                raise ValueError(
                    f"object {object_ref} spans inconsistent inferred sorts"
                )
            [sort_index] = sort_ids
            sort_members[sort_index].add(object_ref)

        works: list[_SortWork] = []
        for sort_index, positions in enumerate(ordered_components):
            all_transitions = frozenset(positions)
            if (
                len(all_transitions)
                > self.max_transitions_per_sort
            ):
                raise ValueError(
                    "LOCM2 transition subset search budget exceeded: "
                    f"{len(all_transitions)} > "
                    f"{self.max_transitions_per_sort}"
                )

            member_refs = tuple(sorted(sort_members[sort_index]))
            sequences: list[tuple[TransitionRef, ...]] = []
            for trace in traces:
                by_object: dict[str, list[TransitionRef]] = {
                    member: []
                    for member in member_refs
                }
                for step in trace.steps:
                    for index, object_ref in enumerate(step.object_refs):
                        position = TransitionRef(step.schema_id, index)
                        if position_sort.get(position) != sort_index:
                            continue
                        if object_ref in by_object:
                            by_object[object_ref].append(position)
                sequences.extend(
                    tuple(values)
                    for values in by_object.values()
                    if values
                )

            pairs = frozenset(
                pair
                for sequence in sequences
                for pair in zip(sequence, sequence[1:])
            )
            machine_sets = _select_transition_sets(
                all_transitions,
                pairs,
                tuple(sequences),
            )
            works.append(
                _SortWork(
                    sort_index=sort_index,
                    member_refs=member_refs,
                    all_transitions=all_transitions,
                    pairs=pairs,
                    sequences=tuple(sequences),
                    machine_sets=machine_sets,
                )
            )

        return tuple(works), position_sort

    def _translate(
        self,
        *,
        traces: tuple[AcquisitionTrace, ...],
        works: tuple[_SortWork, ...],
        position_sort: dict[TransitionRef, int],
    ) -> ActionModelProposal:
        evidence = _evidence_ids(traces)
        type_ids = {
            work.sort_index: f"locm2-type:{work.sort_index}"
            for work in works
        }
        entity_types = tuple(
            EntityTypeHypothesis(
                hypothesis_id=type_ids[work.sort_index],
                member_refs=work.member_refs,
                supporting_evidence_ids=evidence,
                contradicting_evidence_ids=(),
                confidence=0.5,
                status=HypothesisStatus.PROPOSED,
            )
            for work in works
        )

        predicate_values: list[PredicateHypothesis] = []
        state_predicates: dict[
            tuple[int, int, Hashable],
            str,
        ] = {}
        state_unions: dict[
            tuple[int, int],
            _UnionFind,
        ] = {}

        for work in works:
            for machine_index, machine in enumerate(work.machine_sets):
                restricted_pairs = frozenset(
                    (left, right)
                    for left, right in work.pairs
                    if left in machine and right in machine
                )
                union = _state_union(machine, restricted_pairs)
                state_unions[(work.sort_index, machine_index)] = union
                roots = {
                    union.find((transition, endpoint))
                    for transition in machine
                    for endpoint in ("start", "end")
                }
                ordered_roots = tuple(
                    sorted(roots, key=repr)
                )
                for state_index, root in enumerate(ordered_roots):
                    predicate_id = (
                        f"locm2-state:{work.sort_index}:"
                        f"{machine_index}:{state_index}"
                    )
                    state_predicates[
                        (work.sort_index, machine_index, root)
                    ] = predicate_id
                    predicate_values.append(
                        PredicateHypothesis(
                            hypothesis_id=predicate_id,
                            argument_type_ids=(
                                type_ids[work.sort_index],
                            ),
                            symmetric=False,
                            supporting_evidence_ids=evidence,
                            contradicting_evidence_ids=(),
                            confidence=0.5,
                            status=HypothesisStatus.PROPOSED,
                        )
                    )

        arities: dict[str, int] = {}
        for trace in traces:
            for step in trace.steps:
                arities.setdefault(
                    step.schema_id,
                    len(step.object_refs),
                )

        actions: list[ActionSchemaHypothesis] = []
        for schema_id in sorted(arities):
            arity = arities[schema_id]
            parameter_types: list[str] = []
            preconditions: set[str] = set()
            add_effects: set[str] = set()
            delete_effects: set[str] = set()

            for parameter_index in range(arity):
                transition = TransitionRef(
                    schema_id,
                    parameter_index,
                )
                sort_index = position_sort[transition]
                parameter_types.append(type_ids[sort_index])
                work = next(
                    value
                    for value in works
                    if value.sort_index == sort_index
                )
                for machine_index, machine in enumerate(
                    work.machine_sets
                ):
                    if transition not in machine:
                        continue
                    union = state_unions[
                        (sort_index, machine_index)
                    ]
                    start_root = union.find(
                        (transition, "start")
                    )
                    end_root = union.find(
                        (transition, "end")
                    )
                    start_id = state_predicates[
                        (
                            sort_index,
                            machine_index,
                            start_root,
                        )
                    ]
                    end_id = state_predicates[
                        (
                            sort_index,
                            machine_index,
                            end_root,
                        )
                    ]
                    preconditions.add(start_id)
                    if start_id != end_id:
                        delete_effects.add(start_id)
                        add_effects.add(end_id)

            actions.append(
                ActionSchemaHypothesis(
                    hypothesis_id=(
                        "locm2-action:"
                        + sha256(
                            (
                                schema_id
                                + repr(tuple(parameter_types))
                            ).encode()
                        ).hexdigest()[:20]
                    ),
                    source_schema_id=schema_id,
                    parameter_type_ids=tuple(parameter_types),
                    precondition_predicate_ids=tuple(
                        sorted(preconditions)
                    ),
                    add_effect_predicate_ids=tuple(
                        sorted(add_effects)
                    ),
                    delete_effect_predicate_ids=tuple(
                        sorted(delete_effects)
                    ),
                    numeric_effect_ids=(),
                    symmetric_parameter_groups=(),
                    supporting_evidence_ids=evidence,
                    contradicting_evidence_ids=(),
                    confidence=0.5,
                    status=HypothesisStatus.PROPOSED,
                )
            )

        return ActionModelProposal(
            learner_id=self.learner_id,
            entity_types=entity_types,
            predicates=tuple(predicate_values),
            actions=tuple(actions),
            supporting_evidence_ids=evidence,
        )


__all__ = [
    "Locm2Analysis",
    "Locm2Learner",
    "Locm2SortAnalysis",
    "TransitionRef",
]
