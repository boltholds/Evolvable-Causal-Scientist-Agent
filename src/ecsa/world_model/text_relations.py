"""Whole-observation text X/Y acquisition from ordinary world transitions.

The observation payload is retained in full; no hand-selected numeric fields,
benchmark vocabulary, or torch dependency. X contains only the BEFORE state
and action. Y contains only observations generated AFTER the action.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .contracts import InteractionTransition, RawObservation, GroundAction


def canonical_text(value: object) -> str:
    """Stable full JSON serialization; text is not interpreted or summarized."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class TextRelationSample:
    transition_id: str
    schema_id: str
    x_text: str
    y_text: str
    before_observation_id: str
    after_observation_id: str
    success: bool

    def __post_init__(self) -> None:
        if not all((self.transition_id, self.schema_id, self.x_text, self.y_text,
                    self.before_observation_id, self.after_observation_id)):
            raise ValueError("nonempty text relation evidence fields required")
        if type(self.success) is not bool:
            raise TypeError("success must be boolean")


class FullTextTransitionProjector:
    """No application-specific fields are extracted or dropped from payloads."""

    def encode_candidate(self, before: RawObservation, action: GroundAction) -> str:
        if not isinstance(before, RawObservation) or not isinstance(action, GroundAction):
            raise TypeError("raw observation and ground action are required")
        return canonical_text({
            "before": before.payload.thaw(),
            "action": {
                "schema_id": action.schema_id,
                "arguments": [item.thaw() for item in action.arguments],
            },
        })

    def encode_outcome(self, transition: InteractionTransition) -> str:
        if not isinstance(transition, InteractionTransition):
            raise TypeError("typed interaction transition required")
        return canonical_text({
            "after": transition.after.payload.thaw(),
            "outcome": {
                "success": transition.outcome.success,
                "payload": transition.outcome.payload.thaw(),
            },
        })

    def project(self, transition: InteractionTransition) -> TextRelationSample:
        if not isinstance(transition, InteractionTransition):
            raise TypeError("typed interaction transition required")
        return TextRelationSample(
            transition_id=transition.transition_id,
            schema_id=transition.action.schema_id,
            x_text=self.encode_candidate(transition.before, transition.action),
            y_text=self.encode_outcome(transition),
            before_observation_id=transition.before.observation_id,
            after_observation_id=transition.after.observation_id,
            success=transition.outcome.success,
        )


@runtime_checkable
class TextRelationSink(Protocol):
    def observe_text_pair(self, sample: TextRelationSample) -> None: ...


class TextRelationAcquisition:
    """Bounded per-schema evidence bank with optional learner sink."""

    def __init__(self, *, max_pairs_per_schema: int = 512,
                 sink: TextRelationSink | None = None) -> None:
        if type(max_pairs_per_schema) is not int or max_pairs_per_schema < 1:
            raise ValueError("positive per-schema capacity required")
        if sink is not None and not isinstance(sink, TextRelationSink):
            raise TypeError("sink must implement TextRelationSink")
        self.projector = FullTextTransitionProjector()
        self._limit = max_pairs_per_schema
        self._sink = sink
        self._seen: set[str] = set()
        self._by_schema: dict[str, list[TextRelationSample]] = {}

    def observe_transition(self, transition: InteractionTransition) -> TextRelationSample | None:
        if not isinstance(transition, InteractionTransition):
            raise TypeError("typed interaction transition required")
        if transition.transition_id in self._seen:
            return None
        sample = self.projector.project(transition)
        self._seen.add(transition.transition_id)
        group = self._by_schema.setdefault(sample.schema_id, [])
        group.append(sample)
        if len(group) > self._limit:
            del group[:len(group) - self._limit]
        if self._sink is not None:
            self._sink.observe_text_pair(sample)
        return sample

    def samples_for(self, schema_id: str) -> tuple[TextRelationSample, ...]:
        return tuple(self._by_schema.get(schema_id, ()))

    @property
    def total_samples(self) -> int:
        return sum(len(group) for group in self._by_schema.values())

    def ready_schemas(self, *, min_examples: int = 24,
                      min_distinct_x: int = 8) -> tuple[str, ...]:
        if type(min_examples) is not int or min_examples < 1:
            raise ValueError("positive min_examples required")
        if type(min_distinct_x) is not int or min_distinct_x < 1:
            raise ValueError("positive min_distinct_x required")
        return tuple(sorted(schema for schema, group in self._by_schema.items()
                            if len(group) >= min_examples
                            and len({sample.x_text for sample in group}) >= min_distinct_x))
