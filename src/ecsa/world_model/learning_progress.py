"""Domain-independent evidence of epistemic progress and sterile cycles.

A changed public observation, a successful API call or a new evidence ID is
not by itself scientific knowledge. This ledger tracks externally evaluated
prediction improvement, posterior uncertainty reduction and independent
hypothesis tests, without naming any feature/action as important.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite

from .contracts import GroundAction


def _action_signature(action: GroundAction) -> str:
    return repr((action.schema_id, tuple(arg.thaw() for arg in action.arguments)))


def _strings(values: tuple[str, ...], *, label: str) -> None:
    if not isinstance(values, tuple) or not all(
        isinstance(value, str) and bool(value) for value in values
    ):
        raise ValueError(f"{label} must contain nonempty strings")


@dataclass(frozen=True)
class ProgressEvidence:
    transition_id: str
    action: GroundAction
    context_signature: tuple[str, ...]
    before_signature: tuple[str, ...]
    after_signature: tuple[str, ...]
    predictive_gain: float | None
    uncertainty_reduction: float
    confirmed_hypothesis_ids: tuple[str, ...]
    contradicted_hypothesis_ids: tuple[str, ...]
    action_cost: float
    independent_trial_group: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.transition_id, str) or not self.transition_id:
            raise ValueError("transition provenance required")
        if not isinstance(self.action, GroundAction):
            raise TypeError("typed public action required")
        for label in ("context_signature", "before_signature", "after_signature",
                      "confirmed_hypothesis_ids", "contradicted_hypothesis_ids"):
            _strings(getattr(self, label), label=label)
        if self.predictive_gain is not None and (
            isinstance(self.predictive_gain, bool) or
            not isinstance(self.predictive_gain, (int, float)) or
            not isfinite(self.predictive_gain)
        ):
            raise ValueError("predictive gain must be finite or unknown")
        for label in ("uncertainty_reduction", "action_cost"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.action_cost < 0:
            raise ValueError("action cost must be nonnegative")
        if self.independent_trial_group is not None and (
            not isinstance(self.independent_trial_group, str) or
            not self.independent_trial_group
        ):
            raise ValueError("independent trial group must be a nonempty string")


@dataclass(frozen=True)
class ProgressAssessment:
    action_signature: str
    expected_gain: float | None
    repeated_uninformative_count: int
    stagnation_penalty: float
    independent_confirmation_count: int


@dataclass(frozen=True)
class CycleEvidence:
    action_signatures: tuple[str, ...]
    context_signatures: tuple[tuple[str, ...], ...]
    repeated_occurrences: int
    epistemically_verified: bool = False


class LearningProgressLedger:
    """Finite episodic history; duplicate protection spans the retained window.

    Long-term exact duplicate detection requires external durable provenance
    storage. This in-memory ledger intentionally does not grow forever.
    """

    def __init__(self, max_recent: int = 256, stagnation_horizon: int = 8,\n                 min_progress_gain: float = 0.002):
        if type(max_recent) is not int or max_recent < 1:
            raise ValueError("max_recent must be positive")
        if type(stagnation_horizon) is not int or stagnation_horizon < 1:
            raise ValueError("stagnation_horizon must be positive")
        if not isfinite(min_progress_gain) or min_progress_gain < 0:\n            raise ValueError('invalid gain threshold')\n        self.min_progress_gain = min_progress_gain\n        self.max_recent = max_recent
        self.stagnation_horizon = stagnation_horizon
        self._records: deque[ProgressEvidence] = deque()
        self._recent_ids: set[str] = set()
        self.total_observed = 0

    @property
    def recent(self) -> tuple[ProgressEvidence, ...]:
        return tuple(self._records)

    def observe(self, signal: ProgressEvidence) -> None:
        if not isinstance(signal, ProgressEvidence):
            raise TypeError("ProgressEvidence required")
        if signal.transition_id in self._recent_ids:
            raise ValueError("duplicate transition provenance")
        if len(self._records) >= self.max_recent:
            oldest = self._records.popleft()
            self._recent_ids.remove(oldest.transition_id)
        self._records.append(signal)
        self._recent_ids.add(signal.transition_id)
        self.total_observed += 1

    @staticmethod
    def _value(signal: ProgressEvidence) -> float:
        confirmed = bool(signal.confirmed_hypothesis_ids or signal.contradicted_hypothesis_ids)
        qualified = confirmed and signal.independent_trial_group is not None
        predictive = max(float(signal.predictive_gain or 0.0), 0.0)
        uncertainty = max(signal.uncertainty_reduction, 0.0)
        return predictive + uncertainty + (0.25 if qualified else 0.0)

    def assess(self, action: GroundAction, *,
               context_signature: tuple[str, ...]) -> ProgressAssessment:
        if not isinstance(action, GroundAction):
            raise TypeError("GroundAction required")
        _strings(context_signature, label="context_signature")
        signature = _action_signature(action)
        group = [
            record for record in self._records
            if _action_signature(record.action) == signature and
            record.context_signature == context_signature
        ]
        if not group:
            return ProgressAssessment(signature, None, 0, 0.0, 0)
        seen_groups: set[str] = set()
        gains: list[float] = []
        idle = 0
        confirmations = 0
        for record in group:
            value = self._value(record)
            if record.independent_trial_group is not None and (
                record.confirmed_hypothesis_ids or record.contradicted_hypothesis_ids
            ):
                if record.independent_trial_group not in seen_groups:
                    confirmations += 1
                    seen_groups.add(record.independent_trial_group)
                else:
                    # Repeated notices for a single trial must not yield
                    # repeated confirmation credit.
                    value -= 0.25
            value = max(0.0, value)
            gains.append(value)
            if value > self.min_progress_gain:
                idle = 0
            else:
                idle += 1
        penalty = min(1.0, idle / self.stagnation_horizon)
        return ProgressAssessment(
            action_signature=signature,
            expected_gain=sum(gains) / len(gains),
            repeated_uninformative_count=idle,
            stagnation_penalty=penalty,
            independent_confirmation_count=confirmations,
        )

    def detect_cycle(self) -> CycleEvidence:
        if not self._records:
            return CycleEvidence((), (), 0)
        signatures: dict[tuple[tuple[str, ...], str, tuple[str, ...]], int] = {}
        for record in self._records:
            key = (record.before_signature, _action_signature(record.action),
                   record.after_signature)
            signatures[key] = signatures.get(key, 0) + 1
        best = max(signatures, key=signatures.get)
        count = signatures[best]
        if count < self.stagnation_horizon:
            return CycleEvidence((), (), 0)
        return CycleEvidence((best[1],), (best[0], best[2]), count)
