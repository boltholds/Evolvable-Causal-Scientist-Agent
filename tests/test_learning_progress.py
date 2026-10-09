"""Contract-first tests for world-independent epistemic progress."""
import pytest

from ecsa.world_model.contracts import GroundAction, freeze_raw_value
from ecsa.world_model.learning_progress import (
    ProgressEvidence, LearningProgressLedger,
)


def action(name="opaque-A", *arguments):
    return GroundAction(name, tuple(freeze_raw_value(v) for v in arguments))


def evidence(index, *, name="opaque-A", before=("stable",), after=("stable",),
             context=("room",), predictive_gain=None, uncertainty=0.0,
             confirmed=(), contradicted=(), trial=None, cost=1.0):
    return ProgressEvidence(
        transition_id=f"public:{index}", action=action(name),
        context_signature=context, before_signature=before, after_signature=after,
        predictive_gain=predictive_gain, uncertainty_reduction=uncertainty,
        confirmed_hypothesis_ids=confirmed,
        contradicted_hypothesis_ids=contradicted,
        action_cost=cost, independent_trial_group=trial,
    )


def test_duplicate_transition_provenance_rejected():
    ledger = LearningProgressLedger(max_recent=10)
    datum = evidence(1)
    ledger.observe(datum)
    with pytest.raises(ValueError, match="duplicate"):
        ledger.observe(datum)


def test_unique_observation_identifiers_not_credit():
    ledger = LearningProgressLedger(max_recent=16, stagnation_horizon=3)
    for index in range(9):
        ledger.observe(evidence(index, before=(f"id-{index}",), after=(f"id-{index+1}",)))
    score = ledger.assess(action(), context_signature=("room",))
    assert score.expected_gain == 0.0
    assert score.stagnation_penalty > 0
    assert score.repeated_uninformative_count >= 3


def test_unknown_progress_is_not_improvement():
    ledger = LearningProgressLedger()
    ledger.observe(evidence(1))
    current = ledger.assess(action(), context_signature=("room",))
    assert current.expected_gain == 0.0
    assert current.independent_confirmation_count == 0
    assert current.repeated_uninformative_count == 1


def test_progress_memory_is_bounded():
    ledger = LearningProgressLedger(max_recent=7)
    for i in range(32):
        ledger.observe(evidence(i))
    assert len(ledger.recent) == 7
    assert ledger.total_observed == 32
    assert all(item.transition_id.startswith("public:") for item in ledger.recent)


def test_cycle_detection_uses_context_not_private_ids():
    ledger = LearningProgressLedger(max_recent=12, stagnation_horizon=3)
    for i in range(6):
        ledger.observe(evidence(i, name="unknown-retry", before=("unchanged",), after=("unchanged",)))
    cycle = ledger.detect_cycle()
    assert cycle.repeated_occurrences >= 3
    assert cycle.action_signatures
    assert not cycle.epistemically_verified


def test_learning_progress_requires_finite_evidence():
    with pytest.raises(ValueError):
        evidence(1, predictive_gain=float("nan"))
    with pytest.raises(ValueError):
        evidence(1, uncertainty=float("inf"))
    with pytest.raises(ValueError):
        evidence(1, cost=-1.0)


def test_independent_confirmations_do_not_count_as_stagnation():
    ledger = LearningProgressLedger(max_recent=16)
    for i in range(6):
        ledger.observe(evidence(i, trial=f"independent-{i}", confirmed=("h1",)))
    assessment = ledger.assess(action(), context_signature=("room",))
    assert assessment.independent_confirmation_count == 6
    assert assessment.stagnation_penalty == 0
    assert assessment.expected_gain > 0
