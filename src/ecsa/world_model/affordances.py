from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .contracts import FrozenRawValue, GroundAction


@dataclass(frozen=True)
class ArgumentEvidence:
    """Empirical applicability of a value at one argument position.

    This is observational evidence, not a semantic type assertion.
    """
    schema_id: str
    arity: int
    position: int
    value: FrozenRawValue
    successes: int
    failures: int

    @property
    def probability(self) -> float:
        return (self.successes + 1.0) / (self.successes + self.failures + 2.0)

    @property
    def uncertainty(self) -> float:
        alpha = self.successes + 1.0
        beta = self.failures + 1.0
        return alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))


@dataclass(frozen=True)
class ContrastiveEvidence:
    """A controlled argument substitution with opposite outcomes."""
    schema_id: str
    position: int
    successful: FrozenRawValue
    unsuccessful: FrozenRawValue


class ActiveAffordanceLearner:
    """Outcome-driven positional/joint argument learner without world vocabulary.

    State-dependent failures are not treated as proof of invalid argument types.
    Successful trials provide positive evidence; contrasting trials in the same
    observed state provide stronger evidence about argument positions.
    """

    def __init__(self) -> None:
        self._positive: Counter[tuple[str, int, int, FrozenRawValue]] = Counter()
        self._negative: Counter[tuple[str, int, int, FrozenRawValue]] = Counter()
        self._joint_success: Counter[GroundAction] = Counter()
        self._joint_failure: Counter[GroundAction] = Counter()
        self._observations: dict[tuple[str, str], list[tuple[GroundAction, bool]]] = {}

    def observe(
        self,
        action: GroundAction,
        *,
        success: bool,
        state_id: str | None = None,
    ) -> None:
        if not isinstance(action, GroundAction):
            raise TypeError("action must be GroundAction")
        if type(success) is not bool:
            raise TypeError("success must be bool")
        if state_id is not None and not isinstance(state_id, str):
            raise TypeError("state_id must be str or None")
        arity = len(action.arguments)
        for position, argument in enumerate(action.arguments):
            key = (action.schema_id, arity, position, argument)
            if success:
                self._positive[key] += 1
            else:
                self._negative[key] += 1
        if success:
            self._joint_success[action] += 1
        else:
            self._joint_failure[action] += 1
        if state_id is not None:
            self._observations.setdefault((action.schema_id, state_id), []).append(
                (action, success)
            )

    def evidence(self, action: GroundAction, position: int) -> ArgumentEvidence:
        if not 0 <= position < len(action.arguments):
            raise IndexError("argument position out of range")
        value = action.arguments[position]
        key = (action.schema_id, len(action.arguments), position, value)
        return ArgumentEvidence(
            action.schema_id, len(action.arguments), position, value,
            self._positive[key], self._negative[key],
        )

    def contrasts(self) -> tuple[ContrastiveEvidence, ...]:
        found: set[ContrastiveEvidence] = set()
        for (schema_id, _), observations in self._observations.items():
            for index, (left, left_ok) in enumerate(observations):
                for right, right_ok in observations[index + 1:]:
                    if left_ok == right_ok or len(left.arguments) != len(right.arguments):
                        continue
                    differing = [
                        i for i, (a, b) in enumerate(zip(left.arguments, right.arguments))
                        if a != b
                    ]
                    if len(differing) != 1:
                        continue
                    position = differing[0]
                    successful = left if left_ok else right
                    unsuccessful = right if left_ok else left
                    found.add(ContrastiveEvidence(
                        schema_id, position,
                        successful.arguments[position],
                        unsuccessful.arguments[position],
                    ))
        return tuple(sorted(found, key=lambda c: (
            c.schema_id, c.position, repr(c.successful), repr(c.unsuccessful)
        )))

    def score(self, action: GroundAction) -> tuple[float, float, float]:
        """Rank uncertain argument substitutions and promising combinations.

        A schema is still selected by the scheduler; this is a within-schema
        signal. No outcome is assumed to identify a precondition by itself.
        """
        if not action.arguments:
            return (0.0, 0.0, 0.0)
        evidence = tuple(self.evidence(action, i) for i in range(len(action.arguments)))
        positive = sum(value.successes for value in evidence)
        trials = self._joint_success[action] + self._joint_failure[action]
        # Prioritize probing unexplored combinations containing successful values.
        return (
            1.0 / (1.0 + trials),
            float(positive) / (1.0 + sum(v.successes + v.failures for v in evidence)),
            sum(v.uncertainty for v in evidence) / len(evidence),
        )
