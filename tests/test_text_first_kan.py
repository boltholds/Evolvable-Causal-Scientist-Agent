"""Reproducible text / numeric / hybrid KAN feature ablation gates."""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecsa.experimental.text_first_kan import (
    DualEncoderRelation, FixtureLaw, FullTextFeatureProjector,
    HashedTextFeatures, InputRepresentation, TextKANConfig,
    _matrix, _simulation, benchmark_one,
)
from ecsa.experimental.relation_discovery import HeadKind


def test_text_encoder_captures_negations_numbers_and_utf8():
    encoder = HashedTextFeatures(64)
    assert np.array_equal(encoder.encode("открыт +7.5"), encoder.encode("открыт +7.5"))
    assert not np.array_equal(encoder.encode("not open 1.5"), encoder.encode("open 1.5"))
    assert not np.array_equal(encoder.encode("-10.5"), encoder.encode("-10.6"))
    assert encoder.encode("дверь закрыта").shape == (64,)


def test_text_only_numeric_only_and_hybrid_have_identical_vector_dimensions():
    sample = _simulation(FixtureLaw.CATEGORICAL, 51, 5, heldout=False)[0]
    feature = FullTextFeatureProjector(dimension=64)
    for mode in InputRepresentation:
        x = feature.encode(sample.x_text, mode)
        y = feature.encode(sample.y_text, mode)
        assert x.shape == y.shape == (128,)
        if mode is InputRepresentation.TEXT:
            assert np.any(x[0:64] != 0.0)
            assert np.all(x[64:] == 0.0)
        elif mode is InputRepresentation.NUMERIC:
            assert np.all(x[:64] == 0.0)
            assert np.any(x[64:] != 0.0)
        else:
            assert np.any(x[:64] != 0.0) and np.any(x[64:] != 0.0)


def test_numeric_only_does_not_read_categorical_hue():
    feature = FullTextFeatureProjector(dimension=64)
    one = '{"before":{"mode":"amber","u":0.4},"action":{"schema_id":"opaque"}}'
    two = '{"before":{"mode":"violet","u":0.4},"action":{"schema_id":"opaque"}}'
    a = feature.encode(one, InputRepresentation.NUMERIC)
    b = feature.encode(two, InputRepresentation.NUMERIC)
    assert np.array_equal(a, b)
    assert not np.array_equal(feature.encode(one, InputRepresentation.TEXT), feature.encode(two, InputRepresentation.TEXT))


def test_two_towers_are_separate_and_heads_are_parameter_matched_across_representations():
    torch.manual_seed(11)
    kan = DualEncoderRelation(HeadKind.SPLINE_KAN, 128)
    mlp = DualEncoderRelation(HeadKind.MLP, 128)
    assert kan.encoder.x is not kan.encoder.y
    assert kan(torch.randn(6, 128), torch.randn(6, 128)).shape == (6,)
    assert mlp(torch.randn(6, 128), torch.randn(6, 128)).shape == (6,)
    assert sum(t.numel() for t in kan.parameters()) > 0


def test_fixture_observations_have_no_cross_split_object_ids_or_future_in_x():
    train = _simulation(FixtureLaw.INTERACTION, 50, 14, heldout=False)
    test = _simulation(FixtureLaw.INTERACTION, 51, 14, heldout=True)
    ids_train = {json.loads(s.x_text)["before"]["object"]["identity"] for s in train}
    ids_test = {json.loads(s.x_text)["before"]["object"]["identity"] for s in test}
    assert not ids_train.intersection(ids_test)
    assert all("measurement" not in s.x_text for s in train)
    assert all("identity" not in s.y_text for s in train)


def test_short_smoke_on_three_representations_same_data_and_budget():
    cfg = TextKANConfig(samples=32, holdout=16, feature_size=32, training_steps=20)
    rows = [benchmark_one(FixtureLaw.CATEGORICAL, 0, cfg, r, HeadKind.SPLINE_KAN)
            for r in InputRepresentation]
    assert len({row.parameter_count for row in rows}) == 1
    assert all(0 <= row.auroc <= 1 and row.train_observations == 32 for row in rows)
    assert all(0 <= row.paired_accuracy <= 1 for row in rows)
    assert all(row.train_object_id_overlap == 0 for row in rows)


def test_configuration_rejects_invalid_values():
    with pytest.raises(ValueError):
        TextKANConfig(feature_size=1)
