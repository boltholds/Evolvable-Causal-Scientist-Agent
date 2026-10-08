"""RED->GREEN gates for hypothesis-to-encoder negative feedback."""
import inspect
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from ecsa.experimental.feedback_encoder import (
    FeedbackConfig,
    FeedbackMode,
    FeedbackRelationSystem,
    _collapse_penalty,
    _empirical_hypothesis_weights,
    _feedback_fit,
    _fit_warmup,
    _feedback_benchmark,
    _accept_checkpoint,
)
from ecsa.experimental.text_first_kan import FixtureLaw, _simulation
from ecsa.experimental.pretrained_text_study import TextEmbeddingPort


class SyntheticFrozenEncoder:
    """Offline replacement for a frozen model, deliberately not a real LLM."""
    def __init__(self, dimension=32):
        self.dimension = dimension
        self.calls = 0
        self.table = {}

    def encode_many(self, texts):
        assert isinstance(texts, tuple)
        self.calls += 1
        from ecsa.experimental.text_first_kan import HashedTextFeatures
        hasher = HashedTextFeatures(self.dimension)
        return np.stack([hasher.encode(t) for t in texts]).astype(np.float32)


def test_hypothesis_ensemble_and_independent_adapters():
    torch.manual_seed(11)
    model = FeedbackRelationSystem(input_dim=32)
    x = torch.randn(7, 32)
    y = torch.randn(7, 32)
    logits = model.hypothesis_logits(x, y)
    assert logits.shape == (3, 7)
    assert model.x_adapter is not model.y_adapter
    assert len({id(h) for h in model.hypotheses}) == 3
    assert (model.probabilities(x, y) >= 0).all()
    assert (model.probabilities(x, y) <= 1).all()


def test_hypothesis_weights_derived_only_from_calibration():
    model = FeedbackRelationSystem(input_dim=32)
    x = torch.randn(9, 32)
    y = torch.randn(9, 32)
    weights = _empirical_hypothesis_weights(model, x, y)
    assert weights.shape == (3,)
    assert weights.sum().item() == pytest.approx(1.)
    assert weights.min() > 0
    assert model.hypothesis_version == 0
    assert model.adapter_version == 0


def test_zero_variance_is_penalized_but_nonzero_is_finite():
    collapsed = _collapse_penalty(torch.ones(24, 3), min_std=.15)
    spread = _collapse_penalty(torch.randn(24, 3), min_std=.15)
    assert collapsed > spread
    assert torch.isfinite(collapsed)
    assert torch.isfinite(spread)


def test_guard_rejects_bad_calibration_and_latent_collapse():
    assert _accept_checkpoint(before_nll=.7, after_nll=.72, before_spread=.25,
                              after_spread=.23, tolerance=.05, min_spread=.15)
    assert not _accept_checkpoint(before_nll=.7, after_nll=1.7, before_spread=.25,
                                  after_spread=.23, tolerance=.05, min_spread=.15)
    assert not _accept_checkpoint(before_nll=.7, after_nll=.65, before_spread=.25,
                                  after_spread=.005, tolerance=.05, min_spread=.15)


def test_mode_updates_exact_parameter_groups_and_versions():
    cfg = FeedbackConfig(bootstrap=24, adaptation=20, calibration=12, heldout=12,
                         feature_size=32, warmup_steps=10, feedback_steps=10,
                         checkpoint_every=5)
    samples = _simulation(FixtureLaw.CATEGORICAL, 34, 60, heldout=False)
    encoder = SyntheticFrozenEncoder(32)
    x, y = encoder.encode_many(tuple(s.x_text for s in samples)), encoder.encode_many(tuple(s.y_text for s in samples))
    warm = _fit_warmup(x[:24], y[:24], cfg, seed=4)
    assert warm.adapter_version == warm.hypothesis_version == 0
    start_adapter = [p.clone() for p in warm.x_adapter.parameters()]
    start_head = [p.clone() for p in warm.hypotheses[0].parameters()]
    results = {}
    for mode in (FeedbackMode.KAN_ONLY, FeedbackMode.ADAPTER_ONLY, FeedbackMode.ALTERNATING):
        learned, audit = _feedback_fit(warm, x[:44], y[:44], x[44:56], y[44:56],
                                       cfg, mode, seed=4)
        results[mode] = (learned, audit)
        assert audit.optimizer_steps == 10
        assert audit.epochs == 2
    kan, kan_audit = results[FeedbackMode.KAN_ONLY]
    adapter, adapter_audit = results[FeedbackMode.ADAPTER_ONLY]
    alternating, alt_audit = results[FeedbackMode.ALTERNATING]
    assert all(torch.equal(p, before) for p,before in zip(kan.x_adapter.parameters(), start_adapter))
    assert all(torch.equal(p, before) for p,before in zip(adapter.hypotheses[0].parameters(), start_head))
    assert kan.hypothesis_version > 0 and kan.adapter_version == 0
    assert adapter.hypothesis_version == 0 and adapter.adapter_version > 0
    assert alternating.adapter_version > 0 and alternating.hypothesis_version > 0
    assert kan_audit.kan_optimizer_steps == 10 and kan_audit.adapter_optimizer_steps == 0
    assert adapter_audit.kan_optimizer_steps == 0 and adapter_audit.adapter_optimizer_steps == 10
    assert alt_audit.kan_optimizer_steps == alt_audit.adapter_optimizer_steps == 5


def test_three_arms_identical_bootstrap_and_no_heldout_leakage():
    cfg = FeedbackConfig(bootstrap=24, adaptation=20, calibration=12, heldout=12,
                         feature_size=32, warmup_steps=12, feedback_steps=12,
                         checkpoint_every=6)
    rows = _feedback_benchmark(law=FixtureLaw.INTERACTION, seed=0,
                               config=cfg, encoder=SyntheticFrozenEncoder(32))
    assert {r.mode for r in rows} == set(FeedbackMode)
    assert {r.domain for r in rows} >= {"heldout", "paraphrase", "counterfactual_mode"}
    for mode in FeedbackMode:
        r = next(x for x in rows if x.mode is mode and x.domain == 'heldout')
        assert r.bootstrap == 24 and r.adaptation == 20 and r.calibration == 12
        assert r.heldout == 12 and r.identity_overlap == 0
        assert 0 <= r.auroc <= 1
    assert {r.init_auc for r in rows if r.domain == 'heldout'}.__len__() == 1
    assert 'heldout' not in inspect.signature(_feedback_fit).parameters
    assert 'test' not in inspect.signature(_feedback_fit).parameters


def test_feedback_adapter_has_real_gradient_through_frozen_kan():
    cfg = FeedbackConfig(bootstrap=24, adaptation=20, calibration=12, heldout=12,
                         feature_size=32, warmup_steps=6, feedback_steps=2,
                         checkpoint_every=1)
    samples = _simulation(FixtureLaw.INTERACTION, 100, 56, heldout=False)
    encoder = SyntheticFrozenEncoder(32)
    xx = torch.from_numpy(encoder.encode_many(tuple(s.x_text for s in samples)))
    yy = torch.from_numpy(encoder.encode_many(tuple(s.y_text for s in samples)))
    model = _fit_warmup(xx.numpy()[:24], yy.numpy()[:24], cfg, seed=5)
    for p in model.hypotheses.parameters():
        p.requires_grad_(False)
    for p in (*model.x_adapter.parameters(), *model.y_adapter.parameters()):
        p.requires_grad_(True)
    logits = model.hypothesis_logits(xx[24:44], yy[24:44]); loss = torch.nn.functional.softplus(-logits).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.x_adapter.parameters())
    assert all(p.grad is None for p in model.hypotheses.parameters())


def test_malformed_port_and_config_rejected():
    with pytest.raises(ValueError): FeedbackConfig(feature_size=4)
    with pytest.raises(ValueError): FeedbackConfig(calibration=0)
    with pytest.raises(TypeError): FeedbackRelationSystem(input_dim='32')


def test_rejected_epoch_restores_full_model_and_retains_version(monkeypatch):
    from ecsa.experimental import feedback_encoder as feedback
    cfg = FeedbackConfig(bootstrap=20, adaptation=12, calibration=8, heldout=8,
                         feature_size=32, warmup_steps=4, feedback_steps=4,
                         checkpoint_every=2)
    samples = _simulation(FixtureLaw.CATEGORICAL, 201, 40, heldout=False)
    encoder = SyntheticFrozenEncoder(32)
    x = encoder.encode_many(tuple(s.x_text for s in samples))
    y = encoder.encode_many(tuple(s.y_text for s in samples))
    model = _fit_warmup(x[:20], y[:20], cfg, seed=2)
    frozen, _ = _feedback_fit(model, x[:32], y[:32], x[32:40], y[32:40],
                              cfg, FeedbackMode.FROZEN, seed=2)
    monkeypatch.setattr(feedback, "_accept_checkpoint", lambda **kwargs: False)
    rejected, audit = _feedback_fit(model, x[:32], y[:32], x[32:40], y[32:40],
                                    cfg, FeedbackMode.ALTERNATING, seed=2)
    assert audit.accepted_epochs == 0 and audit.rejected_epochs == 2
    assert audit.representation_version == audit.mechanism_version == 0
    assert all(torch.equal(frozen.state_dict()[k],v)
               for k,v in rejected.state_dict().items())
