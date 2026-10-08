from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .contracts import FrozenRawValue, GroundAction
from .perception.base import PerceptualObservation


class PreconditionStatus(StrEnum):
    CANDIDATE = "candidate"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class FeatureCondition:
    feature_key: str
    value: FrozenRawValue


@dataclass(frozen=True)
class PreconditionHypothesis:
    schema_id: str
    arguments: tuple[FrozenRawValue, ...]
    condition: FeatureCondition
    supporting_successes: int
    contradicting_successes: int
    failing_matches: int
    status: PreconditionStatus


def observable_conditions(perception: PerceptualObservation) -> frozenset[FeatureCondition]:
    """Public feature observations only; never infer semantic predicate names."""
    features = [
        FeatureCondition("global:" + feature.feature_id, feature.value)
        for feature in perception.global_features
    ]
    for entity in perception.entities:
        identity = entity.source_identity or entity.local_ref
        features.extend(
            FeatureCondition("entity:" + identity + ":" + f.feature_id, f.value)
            for f in entity.features
        )
    return frozenset(features)


class ActivePreconditionLearner:
    """Generate and falsify necessary-condition candidates for grounded actions.

    Successful transitions are evidence for necessary conditions. Failure alone
    never establishes necessity; partial observability remains explicit.
    """

    def __init__(self) -> None:
        self._records: list[tuple[GroundAction, bool, frozenset[FeatureCondition]]] = []

    def observe(self, action: GroundAction, *, success: bool, before: PerceptualObservation) -> None:
        if not isinstance(action, GroundAction) or not isinstance(before, PerceptualObservation):
            raise TypeError("typed action and perception required")
        if type(success) is not bool:
            raise TypeError("success must be bool")
        self._records.append((action, success, observable_conditions(before)))

    def hypotheses(self, action: GroundAction) -> tuple[PreconditionHypothesis, ...]:
        records = [(success, features) for candidate, success, features in self._records if candidate == action]
        successful = [features for success, features in records if success]
        if not successful:
            return ()
        candidate_conditions = set().union(*successful)
        results = []
        for condition in candidate_conditions:
            support = sum(condition in features for features in successful)
            contradiction = len(successful) - support
            failing_matches = sum(condition in features for success, features in records if not success)
            results.append(PreconditionHypothesis(
                action.schema_id, action.arguments, condition,
                support, contradiction, failing_matches,
                PreconditionStatus.CANDIDATE if not contradiction else PreconditionStatus.CONTRADICTED,
            ))
        return tuple(sorted(results, key=lambda item: (item.condition.feature_key, repr(item.condition.value))))

    def state_score(self, action: GroundAction, perception: PerceptualObservation) -> float:
        conditions = observable_conditions(perception)
        hypotheses = self.hypotheses(action)
        viable = [hyp for hyp in hypotheses if hyp.status is PreconditionStatus.CANDIDATE]
        if not viable:
            return 0.0
        return sum(hyp.condition in conditions for hyp in viable) / len(viable)
