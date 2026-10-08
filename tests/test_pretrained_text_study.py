"""Offline acceptance gates: no model weight download required."""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecsa.experimental.pretrained_text_study import (
    ComparableFeatures, EncoderArm, EvalDomain, PretrainedStudyConfig,
    SentenceTransformerTextEncoder, _hard_negatives, _paraphrase,
    _opposite_mode_outcomes, _move_model_features,
    benchmark, fit_lora_on_observed_pairs,
)
from ecsa.experimental.text_first_kan import FixtureLaw, HashedTextFeatures, _simulation


class FakeSentenceTransformer:
    """Deterministic stub, not a claimed pretrained model."""
    def __init__(self, dimension=32):
        self.max_seq_length = None
        self.calls = []
        self._hash = HashedTextFeatures(dimension)

    def encode(self, texts, **kwargs):
        self.calls.append((tuple(texts), kwargs))
        return np.stack([self._hash.encode(text) for text in texts])


def test_encoder_is_batched_and_cache_is_revision_scoped():
    mock = FakeSentenceTransformer(32)
    port = SentenceTransformerTextEncoder(
        "fake", dimension=32, revision="r1", max_seq_length=128,
        model=mock, batch_size=3,
    )
    a = port.encode_many(("one", "two", "one"))
    assert a.shape == (3, 32)
    assert np.allclose(a[0], a[2])
    assert len(mock.calls) == 1
    assert len(mock.calls[0][0]) == 2
    assert mock.calls[0][1]["truncate_dim"] == 32
    assert mock.max_seq_length == 128
    port.encode_many(("two", "one"))
    assert len(mock.calls) == 1
    port.clear_cache()
    port.encode_many(("two",))
    assert len(mock.calls) == 2


def test_inconsistent_model_dimensionality_fails_loudly():
    mock = FakeSentenceTransformer(32)
    port = SentenceTransformerTextEncoder("fake", dimension=64, model=mock)
    with pytest.raises(ValueError, match="embedding dimensions"):
        port.encode_many(("test",))


def test_same_KAN_input_width_across_all_four_representations():
    samples = _simulation(FixtureLaw.INTERACTION, 0, 5, heldout=False)
    port = SentenceTransformerTextEncoder("fake", dimension=32, model=FakeSentenceTransformer())
    features = ComparableFeatures(32, pretrained=port)
    vectors = {}
    for arm in (EncoderArm.HASH, EncoderArm.FROZEN, EncoderArm.HYBRID, EncoderArm.NUMERIC):
        x, y = features.make(samples, arm)
        assert x.shape == y.shape == (5, 64)
        vectors[arm] = (x, y)
    assert np.any(vectors[EncoderArm.FROZEN][0][:, :32])
    assert not np.any(vectors[EncoderArm.FROZEN][0][:, 32:])
    assert np.any(vectors[EncoderArm.HYBRID][0][:, 32:])
    assert not np.any(vectors[EncoderArm.NUMERIC][0][:, :32])


def test_paraphrase_only_modifies_wording_not_numbers_or_outcomes():
    pair = _simulation(FixtureLaw.NUMERIC, 1, 5, heldout=True)[0]
    modified = _paraphrase(pair)
    before = json.loads(pair.x_text)
    after = json.loads(modified.x_text)
    assert before["before"]["controls"] == after["before"]["controls"]
    assert before["before"]["object"] == after["before"]["object"]
    assert before["before"]["scene"] != after["before"]["scene"]
    assert json.loads(pair.y_text)["after"]["measurement"] == json.loads(modified.y_text)["after"]["measurement"]
    assert pair.transition_id == modified.transition_id


def test_hard_negatives_use_only_evaluation_samples_and_nontrivial_disagreement():
    test = _simulation(FixtureLaw.NUMERIC, 11, 16, heldout=True)
    j = _hard_negatives(test)
    assert len(j) == len(test)
    y = np.array([float(json.loads(s.y_text)["after"]["measurement"]) for s in test])
    assert np.all(np.abs(y - y[j]) >= 0.18)


def test_four_arms_two_eval_domains_have_same_budgets_and_holdout_ids():
    cfg = PretrainedStudyConfig(train_samples=24, heldout_samples=16, feature_size=32,
                                train_steps=22, max_seq_length=128)
    rows = benchmark(
        model_id="fake", config=cfg, seeds=(0,), laws=(FixtureLaw.INTERACTION,),
        sentence_model=FakeSentenceTransformer(),
    )
    assert len(rows) == 12
    assert {r.arm for r in rows} == {
        EncoderArm.HASH, EncoderArm.FROZEN, EncoderArm.HYBRID, EncoderArm.NUMERIC,
    }
    assert {r.eval_domain for r in rows} == set(EvalDomain)
    assert len({r.kan_parameters for r in rows}) == 1
    assert all(r.train_pairs == 24 and r.eval_pairs == 16 for r in rows)
    assert all(r.train_test_identity_overlap == 0 for r in rows)
    assert all(0 <= r.auroc <= 1 and 0 <= r.brier <= 1 for r in rows)
    assert all(r.lora_steps == 0 for r in rows)


def test_hash_numeric_can_run_without_sentence_transformers():
    cfg = PretrainedStudyConfig(train_samples=24, heldout_samples=16,
                                feature_size=32, train_steps=20)
    rows = benchmark(
        config=cfg, seeds=(2,), laws=(FixtureLaw.CATEGORICAL,),
        arms=(EncoderArm.HASH, EncoderArm.NUMERIC),
    )
    assert len(rows) == 6


def test_lora_must_be_last_and_single_law_seed_per_model():
    cfg = PretrainedStudyConfig(train_samples=24, heldout_samples=16,
                                feature_size=32, train_steps=16)
    with pytest.raises(ValueError, match="LoRA must run last"):
        benchmark(
            config=cfg, seeds=(0,), laws=(FixtureLaw.CATEGORICAL,),
            arms=(EncoderArm.LORA, EncoderArm.FROZEN),
            sentence_model=FakeSentenceTransformer(),
        )
    with pytest.raises(ValueError, match="one law/seed"):
        benchmark(
            config=cfg, seeds=(0, 1), laws=(FixtureLaw.CATEGORICAL,),
            arms=(EncoderArm.LORA,), sentence_model=FakeSentenceTransformer(),
        )


def test_train_never_receives_eval_outcomes_and_backbone_lazy():
    import ecsa.experimental.pretrained_text_study as module
    src = inspect.getsource(module.benchmark)
    assert "fit_lora_on_observed_pairs(\n                        encoder, train," in src
    assert "heldout" not in inspect.signature(fit_lora_on_observed_pairs).parameters
    assert "sentence_transformers" not in inspect.getsource(module.ComparableFeatures)


def test_cli_offline_baseline_produces_json(tmp_path, monkeypatch, capsys):
    import sys
    from ecsa.experimental.pretrained_text_study import main
    path = tmp_path / "baseline.json"
    monkeypatch.setattr(sys, "argv", [
        "pretrained_text_study", "--arms", "hash_text", "numeric_only",
        "--laws", "categorical", "--seeds", "0",
        "--train-samples", "24", "--heldout-samples", "16",
        "--feature-size", "32", "--steps", "12",
        "--output", str(path),
    ])
    main()
    report = json.loads(path.read_text(encoding="utf-8"))
    assert len(report["results"]) == 6
    assert json.loads(capsys.readouterr().out)["runs"] == 6


def test_opposite_mode_counterfactual_preserves_numeric_X_and_flips_Y():
    for law in (FixtureLaw.CATEGORICAL, FixtureLaw.INTERACTION):
        samples = _simulation(law, 11, 20, heldout=True)
        opposite = _opposite_mode_outcomes(samples, law)
        assert len(opposite) == len(samples)
        for original, other in zip(samples, opposite):
            assert original.x_text == other.x_text
            measurement = float(json.loads(original.y_text)["after"]["measurement"])
            alternative = float(json.loads(other.y_text)["after"]["measurement"])
            assert abs(measurement + alternative) < 0.2
        assert any(a.y_text != b.y_text for a, b in zip(samples, opposite))
    with pytest.raises(ValueError):
        _opposite_mode_outcomes(
            _simulation(FixtureLaw.NUMERIC, 1, 12, heldout=True),
            FixtureLaw.NUMERIC,
        )


def test_tokenizer_metadata_survives_device_transfer():
    original={"input_ids":torch.tensor([[3,5]]), "prompt": "search", "n": 2}
    moved=_move_model_features(original, torch.device("cpu"))
    assert torch.equal(moved["input_ids"], original["input_ids"])
    assert moved["prompt"] == "search"
    assert moved["n"] == 2
