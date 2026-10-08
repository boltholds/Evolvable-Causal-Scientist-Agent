"""Identity-disjoint transfer evaluation for generic applicability learning.

The oracle is restricted to fixture generation and evaluation: policy inputs
contain opaque action names, argument IDs and public pre-action features only.
All three predictors see exactly the same training transitions. Held-out
outcomes are never passed back to the learners during evaluation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from random import Random

from ecsa.world_model.applicability import ActiveApplicabilityLearner
from ecsa.world_model.candidate_generation import ExperimentHistory
from ecsa.world_model.contracts import GroundAction, freeze_raw_value
from ecsa.world_model.lifted_applicability import LiftedApplicabilityLearner
from ecsa.world_model.perception.base import (
    EntityObservation,
    ObservedFeature,
    PerceptualObservation,
)


@dataclass(frozen=True)
class TransferEvaluation:
    seed: int
    train_actions: int
    heldout_actions: int
    identity_overlap: int
    model_coverage: int
    schema_brier: float
    grounded_brier: float
    lifted_brier: float


@dataclass(frozen=True)
class _Example:
    action: GroundAction
    perception: PerceptualObservation
    success: bool


def _example(
    *,
    entity_id: str,
    enabled: bool,
    nuisance: bool,
    unrelated: bool,
) -> _Example:
    action = GroundAction("A0", (freeze_raw_value(entity_id),))
    observation = PerceptualObservation(
        observation_id="obs:" + entity_id,
        entities=(
            EntityObservation(
                local_ref="ref:" + entity_id,
                source_identity=entity_id,
                features=(
                    ObservedFeature("v0", freeze_raw_value(enabled), "public"),
                    ObservedFeature("v1", freeze_raw_value(nuisance), "public"),
                    ObservedFeature("object-ref", freeze_raw_value(entity_id), "public"),
                ),
                provenance_id="obs:" + entity_id,
            ),
            EntityObservation(
                local_ref="unrelated:" + entity_id,
                source_identity="unrelated:" + entity_id,
                features=(
                    ObservedFeature("v0", freeze_raw_value(unrelated), "public"),
                ),
                provenance_id="obs:" + entity_id,
            ),
        ),
        global_features=(),
    )
    # The oracle is deliberately NOT passed to the learner.
    return _Example(action, observation, enabled)


def _examples(seed: int, *, heldout: bool, per_class: int) -> tuple[_Example, ...]:
    random = Random(seed + (100_000 if heldout else 0))
    prefix = "deployment-" if heldout else "laboratory-"
    entries = [
        _example(
            entity_id=f"{prefix}{seed}-{index}-{int(enabled)}",
            enabled=enabled,
            nuisance=bool(random.getrandbits(1)),
            unrelated=bool(random.getrandbits(1)),
        )
        for enabled in (True, False)
        for index in range(per_class)
    ]
    random.shuffle(entries)
    return tuple(entries)


def _predict_mixture(
    learner: ActiveApplicabilityLearner | LiftedApplicabilityLearner,
    *,
    action: GroundAction,
    before: PerceptualObservation,
    fallback: float,
) -> tuple[float, bool]:
    belief = learner.belief(action.schema_id, len(action.arguments))
    if belief is None:
        return fallback, False
    candidates = learner.predictions(
        belief,
        experiment_id="heldout:" + str(action.arguments[0].thaw()),
        action=action,
        before=before,
    )
    predicted = {
        item.theory_id: item.probability(("success",))
        for item in candidates
    }
    return (
        sum(
            mass * predicted[hypothesis]
            for hypothesis, mass in belief.posterior.probabilities
        ),
        True,
    )


def evaluate_heldout(
    seed: int,
    *,
    train_per_class: int = 8,
    heldout_per_class: int = 12,
) -> TransferEvaluation:
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if (
        type(train_per_class) is not int or train_per_class < 2
        or type(heldout_per_class) is not int or heldout_per_class < 1
    ):
        raise ValueError("positive training and evaluation class counts required")

    train = _examples(seed, heldout=False, per_class=train_per_class)
    heldout = _examples(seed, heldout=True, per_class=heldout_per_class)
    train_ids = {example.action.arguments[0] for example in train}
    test_ids = {example.action.arguments[0] for example in heldout}
    overlap = len(train_ids & test_ids)
    if overlap:
        raise AssertionError("train and evaluation identities must be disjoint")

    baseline = ExperimentHistory()
    grounded = ActiveApplicabilityLearner()
    lifted = LiftedApplicabilityLearner()
    for example in train:
        baseline.record(example.action, success=example.success)
        grounded.observe(
            example.action, success=example.success, before=example.perception,
        )
        lifted.observe(
            example.action, success=example.success, before=example.perception,
        )

    scores = [0.0, 0.0, 0.0]
    covered = 0
    for example in heldout:
        # Strictly prequential: no observation from held-out outcomes is
        # written into any model until AFTER this study (never here).
        guess = baseline.schema_success_probability(
            example.action.schema_id, len(example.action.arguments),
        )
        grounded_guess, _ = _predict_mixture(
            grounded,
            action=example.action,
            before=example.perception,
            fallback=guess,
        )
        lifted_guess, usable = _predict_mixture(
            lifted,
            action=example.action,
            before=example.perception,
            fallback=guess,
        )
        covered += int(usable)
        label = float(example.success)
        for index, predicted in enumerate((guess, grounded_guess, lifted_guess)):
            scores[index] += (predicted - label) ** 2
    total = len(heldout)
    return TransferEvaluation(
        seed=seed,
        train_actions=len(train),
        heldout_actions=total,
        identity_overlap=overlap,
        model_coverage=covered,
        schema_brier=scores[0] / total,
        grounded_brier=scores[1] / total,
        lifted_brier=scores[2] / total,
    )


def evaluate_suite(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> tuple[TransferEvaluation, ...]:
    if not seeds:
        raise ValueError("at least one seed required")
    return tuple(evaluate_heldout(seed) for seed in seeds)


if __name__ == "__main__":
    results = evaluate_suite()
    print(json.dumps([asdict(result) for result in results], sort_keys=True))
    if not all(
        result.identity_overlap == 0
        and result.model_coverage == result.heldout_actions
        and result.lifted_brier < result.schema_brier
        and result.lifted_brier < result.grounded_brier
        for result in results
    ):
        raise SystemExit("identity-held-out transfer acceptance gate failed")
