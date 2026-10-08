"""Optional reproducible benchmark tests: pytest -q tests/test_relation_discovery.py."""
import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from ecsa.contracts import PredictiveDistribution, TheoryPosterior
from ecsa.science import ScienceKernel
from ecsa.experimental.relation_discovery import (
    Dataset, ExperimentConfig, HeadKind, Law, RelationModel,
    evaluate, experiment, fit, make_dataset, mutual_information,
    oracle_compatibility, parameters, probability,
)


def test_parameter_matched_heads_and_independent_encoders():
    torch.manual_seed(0)
    mlp = RelationModel(HeadKind.MLP)
    kan = RelationModel(HeadKind.SPLINE_KAN)
    assert abs(parameters(mlp) - parameters(kan)) <= 2
    assert mlp.encoder.x is not mlp.encoder.y
    for knots, hidden in ((5, 12), (7, 9), (9, 7)):
        variant = RelationModel(HeadKind.SPLINE_KAN, kan_knots=knots, kan_hidden=hidden)
        assert abs(parameters(mlp) - parameters(variant)) / parameters(mlp) < .03
    assert mlp(torch.randn(5, 3), torch.randn(5, 2)).shape == (5,)
    assert kan(torch.randn(5, 3), torch.randn(5, 2)).shape == (5,)


@pytest.mark.parametrize("law", list(Law))
def test_oracle_consistency_and_asymmetric_sensor_surfaces(law):
    data = make_dataset(law, 32, seed=11, radius=.9)
    assert oracle_compatibility(law, data.x, data.y).mean() == 1.
    assert data.x_view.shape == (32, 3)
    assert data.y_view.shape == (32, 2)


def test_science_kernel_agrees_with_vectorized_binary_eig():
    torch.set_num_threads(1)
    data = make_dataset(Law.ADDITIVE, 48, seed=10, radius=.9)
    models = []
    for i in range(3):
        torch.manual_seed(i)
        model = RelationModel(HeadKind.SPLINE_KAN)
        fit(model, data, seed=20+i, steps=45, lr=.012)
        models.append(model)
    models = tuple(models)
    scores = mutual_information(models, data.x_view, data.y_view)
    assert scores.shape == (48,)
    assert np.all(scores >= 0.)
    assert np.all(scores <= 1.000001)
    assert probability(models[0], data.x_view, data.y_view).shape == (48,)

    posterior = TheoryPosterior(tuple((f"h{i}", 1./len(models)) for i in range(len(models))))
    for idx in (0, 3, 7, 15):
        distributions = tuple(
            PredictiveDistribution(
                theory_id=f"h{i}",
                experiment_id=f"query-{idx}",
                probabilities=(
                    (("success",), float(probability(model, data.x_view[idx:idx+1], data.y_view[idx:idx+1])[0])),
                    (("failure",), float(1. - probability(model, data.x_view[idx:idx+1], data.y_view[idx:idx+1])[0])),
                ),
            )
            for i, model in enumerate(models)
        )
        direct = ScienceKernel().score_experiment(posterior, distributions)
        assert math.isclose(
            direct.information_gain_bits, float(scores[idx]), abs_tol=1e-6
        )


def test_shuffled_observations_do_not_look_like_true_pairs():
    torch.set_num_threads(1)
    data = make_dataset(Law.MULTIPLICATIVE, 96, seed=101, radius=.9)
    shuffled = Dataset(
        data.x, data.y, data.x_view,
        data.y_view[np.random.default_rng(899).permutation(len(data.y_view))],
    )
    torch.manual_seed(20)
    model = RelationModel(HeadKind.SPLINE_KAN)
    fit(model, shuffled, seed=10, steps=95, lr=.012)
    quality = evaluate(model, Law.MULTIPLICATIVE, 701, n=96, radius=1.05)
    assert quality["auc"] < .75


def test_short_benchmark_and_equal_active_query_budgets():
    c = ExperimentConfig(
        train_pairs=36, test_pairs=48, steps=35,
        active_queries=4, active_steps=10,
    )
    rows = experiment(Law.ADDITIVE, seed=2, config=c, active=True)
    assert {row.model for row in rows} == {
        "mlp", "spline_kan", "spline_kan_ensemble",
        "spline_kan_eig_batch", "spline_kan_random_batch",
    }
    assert all(0 <= row.auc <= 1 and row.training_seconds >= 0 for row in rows)
    assert rows[-1].queries == rows[-2].queries == 4
