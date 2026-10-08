"""Ablation: objects are disjoint, outcomes withheld during evaluation."""

from ecsa.benchmarks.lifted_transfer import evaluate_heldout, evaluate_suite


def test_lifted_transfer_beats_schema_and_grounded_on_unseen_objects() -> None:
    results = evaluate_suite((0, 1, 2))
    assert len(results) == 3
    for result in results:
        assert result.train_actions == 16
        assert result.heldout_actions == 24
        assert result.identity_overlap == 0
        assert result.model_coverage == result.heldout_actions
        assert result.lifted_brier < result.schema_brier
        assert result.lifted_brier < result.grounded_brier


def test_heldout_evaluation_is_deterministic() -> None:
    assert evaluate_heldout(7) == evaluate_heldout(7)


def test_bad_dataset_parameters_fail() -> None:
    try:
        evaluate_heldout(7, train_per_class=1)
    except ValueError:
        pass
    else:
        raise AssertionError("positive support gate must be enforced")
