from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import exp, log

from ecsa.contracts import PredictiveDistribution, TheoryPosterior

from .contracts import FrozenRawValue, GroundAction
from .perception.base import PerceptualObservation


@dataclass(frozen=True)
class BoundEntityFeature:
    """Observed entity feature, deliberately excluding the entity's identity."""

    feature_id: str
    value: FrozenRawValue

    def __post_init__(self) -> None:
        if not self.feature_id or not isinstance(self.value, FrozenRawValue):
            raise ValueError("typed observable entity feature required")


def lifted_argument_features(
    action: GroundAction,
    perception: PerceptualObservation,
) -> tuple[frozenset[BoundEntityFeature] | None, ...]:
    """Bind entity observations to positional action args, never to names.

    None means the parameter was NOT observed as an entity. It must not be
    interpreted as evidence of feature absence.
    """
    if not isinstance(action, GroundAction):
        raise TypeError("GroundAction required")
    if not isinstance(perception, PerceptualObservation):
        raise TypeError("PerceptualObservation required")

    found: list[frozenset[BoundEntityFeature] | None] = []
    for argument in action.arguments:
        raw = argument.thaw()
        if type(raw) not in (str, int, float):
            found.append(None)
            continue
        reference = str(raw)
        matches = tuple(
            entity
            for entity in perception.entities
            if reference in (entity.local_ref, entity.source_identity)
        )
        # Ambiguous identity links are not evidence.
        if len(matches) != 1:
            found.append(None)
            continue
        entity = matches[0]
        identity_tokens = {reference, entity.local_ref}
        if entity.source_identity is not None:
            identity_tokens.add(entity.source_identity)

        found.append(frozenset(
            BoundEntityFeature(feature.feature_id, feature.value)
            for feature in entity.features
            if not (
                feature.value.kind.value == "scalar"
                and feature.value.thaw() is not None
                and str(feature.value.thaw()) in identity_tokens
            )
        ))
    return tuple(found)


@dataclass(frozen=True)
class LiftedCondition:
    argument_index: int
    feature_id: str
    value: FrozenRawValue
    present: bool = True

    def __post_init__(self) -> None:
        if type(self.argument_index) is not int or self.argument_index < 0:
            raise ValueError("nonnegative positional argument index required")
        if not self.feature_id or not isinstance(self.value, FrozenRawValue):
            raise ValueError("typed observed feature required")
        if type(self.present) is not bool:
            raise ValueError("present must be bool")

    def evaluate(
        self,
        argument_features: tuple[frozenset[BoundEntityFeature] | None, ...],
    ) -> bool | None:
        if self.argument_index >= len(argument_features):
            raise ValueError("literal index outside action signature")
        slot = argument_features[self.argument_index]
        if slot is None:
            return None
        exists = BoundEntityFeature(self.feature_id, self.value) in slot
        return exists if self.present else not exists


class LiftedRuleKind(StrEnum):
    ALWAYS = "always"
    NEVER = "never"
    ARGUMENT_FEATURE = "argument_feature"
    CONJUNCTION = "conjunction"


@dataclass(frozen=True)
class LiftedApplicabilityHypothesis:
    schema_id: str
    arity: int
    kind: LiftedRuleKind
    conditions: tuple[LiftedCondition, ...] = ()

    def __post_init__(self) -> None:
        if not self.schema_id or type(self.arity) is not int or self.arity < 0:
            raise ValueError("valid action schema and arity required")
        if not isinstance(self.kind, LiftedRuleKind):
            raise TypeError("LiftedRuleKind required")
        expected = {
            LiftedRuleKind.ALWAYS: 0,
            LiftedRuleKind.NEVER: 0,
            LiftedRuleKind.ARGUMENT_FEATURE: 1,
            LiftedRuleKind.CONJUNCTION: 2,
        }[self.kind]
        if (
            not isinstance(self.conditions, tuple)
            or len(self.conditions) != expected
            or len(set(self.conditions)) != len(self.conditions)
            or any(
                not isinstance(condition, LiftedCondition)
                or condition.argument_index >= self.arity
                for condition in self.conditions
            )
        ):
            raise ValueError("typed, nonredundant lifted literals required")

    @property
    def theory_id(self) -> str:
        payload = (
            self.schema_id, self.arity, self.kind.value,
            tuple(sorted(self.conditions, key=repr)),
        )
        return "lifted-applicability:" + sha256(repr(payload).encode()).hexdigest()

    def applicable(
        self,
        action: GroundAction,
        features: tuple[frozenset[BoundEntityFeature] | None, ...],
    ) -> bool | None:
        if action.schema_id != self.schema_id or len(action.arguments) != self.arity:
            raise ValueError("action does not match hypothesis signature")
        if self.kind is LiftedRuleKind.ALWAYS:
            return True
        if self.kind is LiftedRuleKind.NEVER:
            return False
        values = tuple(condition.evaluate(features) for condition in self.conditions)
        # With partially observed arguments, do not infer absence or
        # impossibility from missing entity observations.
        if any(value is None for value in values):
            return None
        return all(values)

    def success_probability(
        self,
        action: GroundAction,
        features: tuple[frozenset[BoundEntityFeature] | None, ...],
        *,
        reliability: float,
    ) -> float:
        applicable = self.applicable(action, features)
        if applicable is None:
            return 0.5
        return reliability if applicable else 1.0 - reliability


@dataclass(frozen=True)
class LiftedApplicabilityBelief:
    hypotheses: tuple[LiftedApplicabilityHypothesis, ...]
    posterior: TheoryPosterior
    evidence_count: int


@dataclass(frozen=True)
class _Trial:
    action: GroundAction
    success: bool
    features: tuple[frozenset[BoundEntityFeature] | None, ...]


class LiftedApplicabilityLearner:
    """Bayesian, identity-invariant candidate mechanism learner.

    Conditions describe features of argument roles (arg[i]), not grounded
    entity IDs. Two distinct positive bindings and a negative contrast are
    required before a lifted rule can enter the posterior. These conditions
    remain falsifiable correlations; hidden preconditions remain possible.
    """

    _PRIOR = {
        LiftedRuleKind.ALWAYS: 0.17,
        LiftedRuleKind.NEVER: 0.17,
        LiftedRuleKind.ARGUMENT_FEATURE: 0.50,
        LiftedRuleKind.CONJUNCTION: 0.16,
    }

    def __init__(
        self,
        *,
        reliability: float = 0.9,
        max_history_per_schema: int = 256,
        max_features_per_position: int = 32,
        max_literal_rules: int = 24,
        max_conjunction_rules: int = 12,
    ) -> None:
        if type(reliability) is not float or not 0.5 < reliability < 1.0:
            raise ValueError("reliability must be a float strictly between 0.5 and 1")
        for capacity in (
            max_history_per_schema,
            max_features_per_position,
            max_literal_rules,
            max_conjunction_rules,
        ):
            if type(capacity) is not int or capacity < 1:
                raise ValueError("capacity limits must be positive integers")
        self.reliability = reliability
        self.max_history_per_schema = max_history_per_schema
        self.max_features_per_position = max_features_per_position
        self.max_literal_rules = max_literal_rules
        self.max_conjunction_rules = max_conjunction_rules
        self._history: dict[tuple[str, int], list[_Trial]] = defaultdict(list)

    def observe(
        self,
        action: GroundAction,
        *,
        success: bool,
        before: PerceptualObservation,
    ) -> None:
        if type(success) is not bool:
            raise TypeError("success must be bool")
        features = lifted_argument_features(action, before)
        key = (action.schema_id, len(action.arguments))
        trials = self._history[key]
        trials.append(_Trial(action, success, features))
        if len(trials) > self.max_history_per_schema:
            del trials[:len(trials) - self.max_history_per_schema]

    @staticmethod
    def _evaluate_all(
        conditions: tuple[LiftedCondition, ...],
        trial: _Trial,
    ) -> bool | None:
        results = tuple(condition.evaluate(trial.features) for condition in conditions)
        if any(result is None for result in results):
            return None
        return all(results)

    def _admissible(
        self,
        conditions: tuple[LiftedCondition, ...],
        successes: list[_Trial],
        failures: list[_Trial],
    ) -> tuple[int, int] | None:
        supporting = [trial for trial in successes if self._evaluate_all(conditions, trial) is True]
        if len(supporting) < 2:
            return None
        for position in {condition.argument_index for condition in conditions}:
            refs = {trial.action.arguments[position] for trial in supporting}
            if len(refs) < 2:
                return None
        contrast = sum(self._evaluate_all(conditions, trial) is False for trial in failures)
        if not contrast:
            return None
        return (len(supporting), contrast)

    def belief(self, schema_id: str, arity: int) -> LiftedApplicabilityBelief | None:
        if not schema_id or type(arity) is not int or arity < 0:
            raise ValueError("valid action signature required")
        trials = self._history.get((schema_id, arity), [])
        successes = [trial for trial in trials if trial.success]
        failures = [trial for trial in trials if not trial.success]
        if len(successes) < 2 or not failures or arity == 0:
            return None

        observed: list[Counter[BoundEntityFeature]] = [Counter() for _ in range(arity)]
        for trial in trials:
            for index, slot in enumerate(trial.features):
                if slot is not None:
                    observed[index].update(slot)

        proposed: list[tuple[int, int, LiftedCondition]] = []
        for index, counts in enumerate(observed):
            for feature in sorted(counts, key=lambda value: (-counts[value], repr(value)))[:self.max_features_per_position]:
                for present in (True, False):
                    literal = LiftedCondition(index, feature.feature_id, feature.value, present)
                    support = self._admissible((literal,), successes, failures)
                    if support is not None:
                        proposed.append((support[0], support[1], literal))
        if not proposed:
            return None
        proposed.sort(key=lambda pair: (-pair[0], -pair[1], repr(pair[2])))
        literals = tuple(value for _, _, value in proposed[:self.max_literal_rules])
        unary = tuple(
            LiftedApplicabilityHypothesis(
                schema_id, arity, LiftedRuleKind.ARGUMENT_FEATURE, (literal,),
            )
            for literal in literals
        )
        # Conjunctions are kept bounded and must explain at least two
        # distinct successful argument bindings plus a negative contrast.
        conjunctions: list[tuple[int, int, LiftedApplicabilityHypothesis]] = []
        for index, left in enumerate(literals):
            for right in literals[index + 1:]:
                if left.argument_index == right.argument_index and left.feature_id == right.feature_id:
                    continue
                conditions = (left, right)
                support = self._admissible(conditions, successes, failures)
                if support is not None:
                    conjunctions.append((
                        support[0], support[1],
                        LiftedApplicabilityHypothesis(
                            schema_id, arity, LiftedRuleKind.CONJUNCTION, conditions,
                        ),
                    ))
        conjunctions.sort(key=lambda item: (-item[0], -item[1], item[2].theory_id))
        hypotheses = (
            LiftedApplicabilityHypothesis(schema_id, arity, LiftedRuleKind.ALWAYS),
            LiftedApplicabilityHypothesis(schema_id, arity, LiftedRuleKind.NEVER),
            *unary,
            *(item[2] for item in conjunctions[:self.max_conjunction_rules]),
        )
        family_count = Counter(h.kind for h in hypotheses)
        total_prior = sum(self._PRIOR[kind] for kind in family_count)
        log_weights: list[float] = []
        for hypothesis in hypotheses:
            prior = self._PRIOR[hypothesis.kind] / (total_prior * family_count[hypothesis.kind])
            weight = log(prior)
            for trial in trials:
                probability = hypothesis.success_probability(
                    trial.action, trial.features, reliability=self.reliability,
                )
                weight += log(probability if trial.success else 1.0 - probability)
            log_weights.append(weight)
        max_weight = max(log_weights)
        weights = tuple(exp(value - max_weight) for value in log_weights)
        denominator = sum(weights)
        return LiftedApplicabilityBelief(
            hypotheses=tuple(hypotheses),
            posterior=TheoryPosterior(tuple(
                (h.theory_id, w / denominator)
                for h, w in zip(hypotheses, weights)
            )),
            evidence_count=len(trials),
        )

    def predictions(
        self,
        belief: LiftedApplicabilityBelief,
        *,
        experiment_id: str,
        action: GroundAction,
        before: PerceptualObservation,
    ) -> tuple[PredictiveDistribution, ...]:
        if not isinstance(belief, LiftedApplicabilityBelief):
            raise TypeError("LiftedApplicabilityBelief required")
        if not experiment_id:
            raise ValueError("experiment_id required")
        features = lifted_argument_features(action, before)
        return tuple(
            PredictiveDistribution(
                theory_id=hypothesis.theory_id,
                experiment_id=experiment_id,
                probabilities=(
                    (("success",), probability),
                    (("failure",), 1.0 - probability),
                ),
            )
            for hypothesis in belief.hypotheses
            for probability in (
                hypothesis.success_probability(
                    action, features, reliability=self.reliability,
                ),
            )
        )
