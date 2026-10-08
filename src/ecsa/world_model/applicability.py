from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import exp, log

from ecsa.contracts import PredictiveDistribution, TheoryPosterior

from .contracts import FrozenRawValue, GroundAction
from .perception.base import PerceptualObservation
from .preconditions import FeatureCondition, observable_conditions


class ApplicabilityRule(StrEnum):
    ALWAYS = "always"
    NEVER = "never"
    ARGUMENT_EQUALS = "argument_equals"
    FEATURE_PRESENT = "feature_present"
    FEATURE_ABSENT = "feature_absent"
    ARGUMENT_AND_FEATURE = "argument_and_feature"


@dataclass(frozen=True)
class ApplicabilityHypothesis:
    """An observable, falsifiable rule over an opaque action signature."""
    schema_id: str
    arity: int
    rule: ApplicabilityRule
    argument_index: int | None = None
    argument_value: FrozenRawValue | None = None
    condition: FeatureCondition | None = None

    def __post_init__(self) -> None:
        if not self.schema_id or type(self.arity) is not int or self.arity < 0:
            raise ValueError("valid schema identifier and arity required")
        if not isinstance(self.rule, ApplicabilityRule):
            raise TypeError("typed applicability rule required")
        uses_argument = self.rule in (
            ApplicabilityRule.ARGUMENT_EQUALS,
            ApplicabilityRule.ARGUMENT_AND_FEATURE,
        )
        uses_feature = self.rule in (
            ApplicabilityRule.FEATURE_PRESENT,
            ApplicabilityRule.FEATURE_ABSENT,
            ApplicabilityRule.ARGUMENT_AND_FEATURE,
        )
        if uses_argument:
            if (
                type(self.argument_index) is not int
                or not 0 <= self.argument_index < self.arity
                or not isinstance(self.argument_value, FrozenRawValue)
            ):
                raise ValueError("argument rule requires indexed frozen value")
        elif self.argument_index is not None or self.argument_value is not None:
            raise ValueError("unexpected argument constraint")
        if uses_feature:
            if not isinstance(self.condition, FeatureCondition):
                raise ValueError("feature rule requires an observed condition")
        elif self.condition is not None:
            raise ValueError("unexpected feature constraint")

    @property
    def theory_id(self) -> str:
        signature = (
            self.schema_id,
            self.arity,
            self.rule.value,
            self.argument_index,
            self.argument_value,
            self.condition,
        )
        return "applicability:" + sha256(repr(signature).encode()).hexdigest()

    def predicts_applicable(
        self,
        action: GroundAction,
        conditions: frozenset[FeatureCondition],
    ) -> bool:
        if action.schema_id != self.schema_id or len(action.arguments) != self.arity:
            raise ValueError("action does not match applicability signature")
        if self.rule is ApplicabilityRule.ALWAYS:
            return True
        if self.rule is ApplicabilityRule.NEVER:
            return False
        if self.rule is ApplicabilityRule.FEATURE_PRESENT:
            return self.condition in conditions
        if self.rule is ApplicabilityRule.FEATURE_ABSENT:
            return self.condition not in conditions
        assert self.argument_index is not None
        argument_matches = action.arguments[self.argument_index] == self.argument_value
        if self.rule is ApplicabilityRule.ARGUMENT_EQUALS:
            return argument_matches
        return argument_matches and self.condition in conditions

    def success_probability(
        self,
        action: GroundAction,
        conditions: frozenset[FeatureCondition],
        *,
        reliability: float = 0.9,
    ) -> float:
        return reliability if self.predicts_applicable(action, conditions) else 1.0 - reliability


@dataclass(frozen=True)
class ApplicabilityBelief:
    hypotheses: tuple[ApplicabilityHypothesis, ...]
    posterior: TheoryPosterior
    evidence_count: int


@dataclass(frozen=True)
class _Trial:
    action: GroundAction
    success: bool
    conditions: frozenset[FeatureCondition]


class ActiveApplicabilityLearner:
    """Bounded Bayesian version-space over grounded applicability mechanisms.

    Builds rules only from observed successful and failed experiments.
    Probabilities are a deliberately noisy working likelihood, not causal proof.
    """

    _FAMILY_PRIOR = {
        ApplicabilityRule.ALWAYS: 0.20,
        ApplicabilityRule.NEVER: 0.20,
        ApplicabilityRule.ARGUMENT_EQUALS: 0.25,
        ApplicabilityRule.FEATURE_PRESENT: 0.20,
        ApplicabilityRule.FEATURE_ABSENT: 0.05,
        ApplicabilityRule.ARGUMENT_AND_FEATURE: 0.10,
    }

    def __init__(
        self,
        *,
        reliability: float = 0.9,
        max_history_per_schema: int = 256,
        max_argument_rules: int = 12,
        max_feature_rules: int = 12,
        max_joint_rules: int = 12,
    ) -> None:
        if (
            type(reliability) is not float
            or not 0.5 < reliability < 1.0
        ):
            raise ValueError("reliability must be float in (0.5, 1)")
        for value in (
            max_history_per_schema, max_argument_rules,
            max_feature_rules, max_joint_rules,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("all capacity limits must be positive integers")
        self.reliability = reliability
        self.max_history_per_schema = max_history_per_schema
        self.max_argument_rules = max_argument_rules
        self.max_feature_rules = max_feature_rules
        self.max_joint_rules = max_joint_rules
        self._history: dict[tuple[str, int], list[_Trial]] = defaultdict(list)

    def observe(
        self,
        action: GroundAction,
        *,
        success: bool,
        before: PerceptualObservation,
    ) -> None:
        if not isinstance(action, GroundAction) or not isinstance(before, PerceptualObservation):
            raise TypeError("typed action and pre-action perception required")
        if type(success) is not bool:
            raise TypeError("success must be bool")
        key = (action.schema_id, len(action.arguments))
        trials = self._history[key]
        trials.append(_Trial(action, success, observable_conditions(before)))
        if len(trials) > self.max_history_per_schema:
            del trials[: len(trials) - self.max_history_per_schema]

    def _rules(self, schema_id: str, arity: int, trials: list[_Trial]) -> tuple[ApplicabilityHypothesis, ...]:
        positive = [trial for trial in trials if trial.success]
        negative = [trial for trial in trials if not trial.success]
        base = (
            ApplicabilityHypothesis(schema_id, arity, ApplicabilityRule.ALWAYS),
            ApplicabilityHypothesis(schema_id, arity, ApplicabilityRule.NEVER),
        )
        if not positive or not negative:
            return ()

        arguments: Counter[tuple[int, FrozenRawValue]] = Counter()
        features: Counter[FeatureCondition] = Counter()
        negative_features: Counter[FeatureCondition] = Counter()
        joints: Counter[tuple[int, FrozenRawValue, FeatureCondition]] = Counter()
        for trial in positive:
            for index, argument in enumerate(trial.action.arguments):
                arguments[(index, argument)] += 1
                for condition in trial.conditions:
                    joints[(index, argument, condition)] += 1
            features.update(trial.conditions)
        for trial in negative:
            negative_features.update(trial.conditions)

        def rank(counter: Counter):
            return sorted(counter, key=lambda key: (-counter[key], repr(key)))

        arg_rules = tuple(
            ApplicabilityHypothesis(
                schema_id, arity, ApplicabilityRule.ARGUMENT_EQUALS, index, value,
            )
            for index, value in rank(arguments)[:self.max_argument_rules]
        )
        feature_rules = tuple(
            ApplicabilityHypothesis(
                schema_id, arity, ApplicabilityRule.FEATURE_PRESENT, condition=condition,
            )
            for condition in rank(features)[:self.max_feature_rules]
        )
        # Presence in a failed state and absence from all successful states
        # is a plausible negative precondition, not a proven one.
        absent_rules = tuple(
            ApplicabilityHypothesis(
                schema_id, arity, ApplicabilityRule.FEATURE_ABSENT, condition=condition,
            )
            for condition in rank(negative_features)
            if condition not in features
        )[:self.max_feature_rules]
        joint_rules = tuple(
            ApplicabilityHypothesis(
                schema_id, arity, ApplicabilityRule.ARGUMENT_AND_FEATURE,
                index, value, condition,
            )
            for index, value, condition in rank(joints)[:self.max_joint_rules]
        )
        return base + arg_rules + feature_rules + absent_rules + joint_rules

    def belief(self, schema_id: str, arity: int) -> ApplicabilityBelief | None:
        if not schema_id or type(arity) is not int or arity < 0:
            raise ValueError("valid schema and arity required")
        trials = self._history.get((schema_id, arity), [])
        hypotheses = self._rules(schema_id, arity, trials)
        if len(hypotheses) < 3:
            return None

        families: Counter[ApplicabilityRule] = Counter(hyp.rule for hyp in hypotheses)
        active_mass = sum(self._FAMILY_PRIOR[rule] for rule in families)
        log_weights = []
        for hypothesis in hypotheses:
            family_mass = self._FAMILY_PRIOR[hypothesis.rule]
            prior = family_mass / (active_mass * families[hypothesis.rule])
            likelihood = log(prior)
            for trial in trials:
                p = hypothesis.success_probability(
                    trial.action, trial.conditions, reliability=self.reliability,
                )
                likelihood += log(p if trial.success else 1.0 - p)
            log_weights.append(likelihood)
        maximum = max(log_weights)
        weights = tuple(exp(value - maximum) for value in log_weights)
        normalizer = sum(weights)
        posterior = TheoryPosterior(
            tuple(
                (hypothesis.theory_id, weight / normalizer)
                for hypothesis, weight in zip(hypotheses, weights)
            )
        )
        return ApplicabilityBelief(hypotheses, posterior, len(trials))

    def predictions(
        self,
        belief: ApplicabilityBelief,
        *,
        experiment_id: str,
        action: GroundAction,
        before: PerceptualObservation,
    ) -> tuple[PredictiveDistribution, ...]:
        if not isinstance(belief, ApplicabilityBelief):
            raise TypeError("applicability belief required")
        if not experiment_id:
            raise ValueError("experiment identifier required")
        if not isinstance(before, PerceptualObservation):
            raise TypeError("pre-action perception required")
        conditions = observable_conditions(before)
        result = []
        for hypothesis in belief.hypotheses:
            success = hypothesis.success_probability(
                action, conditions, reliability=self.reliability,
            )
            result.append(PredictiveDistribution(
                theory_id=hypothesis.theory_id,
                experiment_id=experiment_id,
                probabilities=(
                    (("success",), success),
                    (("failure",), 1.0 - success),
                ),
            ))
        return tuple(result)
