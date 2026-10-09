"""Domain-neutral *diagnostics* for matched ECSA policy ablations.

Repeated unsuccessful actions with vanishing action-success variance reduction
are a conservative proxy for low-information retries, not a proof that an
experiment could never reveal useful science. Never label them causal failures.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite, log2
import json


@dataclass(frozen=True)
class ProgressActionEvent:
    transition_id: str
    schema_id: str
    arguments: tuple[object, ...]
    success: bool
    prior_schema_variance_reduction: float
    context_signature: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.transition_id or not self.schema_id:
            raise ValueError("transition and action schema provenance required")
        if not isinstance(self.arguments, tuple):
            raise TypeError("action arguments must be immutable")
        if type(self.success) is not bool:
            raise TypeError("public action success must be bool")
        if (
            isinstance(self.prior_schema_variance_reduction, bool)
            or not isinstance(self.prior_schema_variance_reduction, (int, float))
            or not isfinite(self.prior_schema_variance_reduction)
        ):
            raise ValueError("finite prequential information estimate required")
        if not isinstance(self.context_signature, tuple) or any(
            not isinstance(value, str) for value in self.context_signature
        ):
            raise TypeError("public context signature must be strings")


@dataclass(frozen=True)
class ProgressEpisodeMetrics:
    total_actions: int
    distinct_schemas: int
    action_counts: tuple[tuple[str, int], ...]
    longest_exact_action_streak: int
    low_information_failed_repeats: int
    max_schema_fraction: float
    schema_entropy_bits: float


def _action_key(event: ProgressActionEvent) -> str:
    return json.dumps(
        [event.schema_id, event.arguments], sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, default=repr
    )


def summarize_progress_actions(
    events: tuple[ProgressActionEvent, ...],
    *,
    min_gain: float = 0.002,
    min_repeats: int = 2,
) -> ProgressEpisodeMetrics:
    if not isinstance(events, tuple) or any(
        not isinstance(x, ProgressActionEvent) for x in events
    ):
        raise TypeError("immutable public action events required")
    if type(min_repeats) is not int or min_repeats < 1:
        raise ValueError("min_repeats must be positive")
    if (
        isinstance(min_gain, bool) or not isinstance(min_gain, (int, float))
        or not isfinite(min_gain) or min_gain < 0
    ):
        raise ValueError("min_gain must be finite and nonnegative")

    ids = [event.transition_id for event in events]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate transition IDs in episode")

    counts = Counter(event.schema_id for event in events)
    seen: Counter[tuple[str, tuple[str, ...]]] = Counter()
    previous = None
    streak = 0
    longest = 0
    low_info = 0

    for item in events:
        key = _action_key(item)
        contextual = (key, item.context_signature)
        if (
            seen[contextual] >= min_repeats
            and not item.success
            and item.prior_schema_variance_reduction <= min_gain
        ):
            low_info += 1
        seen[contextual] += 1

        streak = streak + 1 if key == previous else 1
        longest = max(longest, streak)
        previous = key

    total = len(events)
    maximum = max(counts.values()) / total if total else 0.0
    entropy = -sum(
        (v / total) * log2(v / total) for v in counts.values()
    ) if total else 0.0
    return ProgressEpisodeMetrics(
        total_actions=total,
        distinct_schemas=len(counts),
        action_counts=tuple(sorted(counts.items())),
        longest_exact_action_streak=longest,
        low_information_failed_repeats=low_info,
        max_schema_fraction=maximum,
        schema_entropy_bits=entropy,
    )
