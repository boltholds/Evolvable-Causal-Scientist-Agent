"""Only KAN hypotheses learn; X/Y token encoders remain permanently frozen."""
from __future__ import annotations

import inspect
import json
from hashlib import sha256

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from ecsa.experimental.token_state_relations import (
    DEFAULT_ARMS, EvaluationDomain, StateArm, TokenRelationConfig,
    TokenStates, TensorStates, TokenKANSystem, StateProjector,
    _allow_grad, _losses, _batch_for, fit_kan_only,
    train_warmup, benchmark_one, _counterfactual_decimal,
)
from ecsa.experimental.verbalization_study import StudyLaw, _simulation


class FakePort:
    dimension = 96
    def __init__(self):
        self.forward_calls = 0
        self.max_observed_length = 0
        self.verified_length_count = 0

    def encode_many(self, texts: tuple[str, ...]) -> TokenStates:
        self.forward_calls += 1
        length = 6
        tokens = np.zeros((len(texts), length, self.dimension), np.float32)
        mask = np.zeros((len(texts), length), bool)
        sentence = np.zeros((len(texts), self.dimension), np.float32)
        for i, text in enumerate(texts):
            chars = [text[k:k+24] for k in range(0, len(text), 24)][:length]
            if not chars: chars = [""]
            for j, chunk in enumerate(chars):
                digest = sha256((chunk+str(j)).encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                tokens[i,j] = rng.normal(0, 1, self.dimension)
            mask[i,:len(chars)] = True
            sentence[i] = tokens[i,:len(chars)].mean(0)
            sentence[i] /= max(np.linalg.norm(sentence[i]), 1e-10)
        self.max_observed_length = max(self.max_observed_length, length)
        self.verified_length_count += len(texts)
        return TokenStates(tokens,mask,sentence)


def config() -> TokenRelationConfig:
    return TokenRelationConfig(
        bootstrap=12, adaptation=8, calibration=8, heldout=9,
        warmup_steps=10, extra_kan_steps=6, checkpoint_every=3,
    )


def test_state_arms_have_no_encoder_feedback():
    assert len(DEFAULT_ARMS) == 6
    assert all("feedback" not in arm.value for arm in StateArm)
    with pytest.raises(ValueError):
        StateArm("attention_feedback")


def test_token_states_require_mask_and_finite_vectors():
    valid = FakePort().encode_many(("one", "two"))
    assert valid.width == 96
    assert valid.mask.shape == valid.tokens.shape[:2]
    with pytest.raises(ValueError):
        TokenStates(valid.tokens,valid.mask.astype(np.int8),valid.sentence)


@pytest.mark.parametrize("arm", list(StateArm))
def test_encoder_parameters_never_train_from_kan(arm):
    torch.manual_seed(12)
    model = TokenKANSystem(arm,96)
    assert not any(p.requires_grad for p in model.x_encoder.parameters())
    assert not any(p.requires_grad for p in model.y_encoder.parameters())
    rows = _simulation(StudyLaw.OPERATOR,3,14,heldout=False)
    x,y = _batch_for(FakePort(),rows)
    _allow_grad(model,heads=True)
    _losses(model,x,y,1).mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.hypotheses.parameters())
    assert all(p.grad is None for p in model.x_encoder.parameters())
    assert all(p.grad is None for p in model.y_encoder.parameters())


def test_masked_mean_does_not_depend_on_padding():
    port = FakePort()
    sample = port.encode_many(("red one", "green two"))
    padded = np.pad(sample.tokens,((0,0),(0,4),(0,0)))
    mask = np.pad(sample.mask,((0,0),(0,4)))
    states = TensorStates.from_array(sample)
    extra = TensorStates.from_array(TokenStates(padded,mask,sample.sentence))
    for arm in (StateArm.TOKEN_MEAN,StateArm.TOKEN_ATTENTION,StateArm.GATED_ATTENTION):
        torch.manual_seed(11)
        model = StateProjector(arm,96)
        assert torch.allclose(model(states),model(extra),atol=1e-6)


def test_kan_only_fit_does_not_change_either_encoder():
    rows = _simulation(StudyLaw.PRECISION,2,28,heldout=False)
    x,y = _batch_for(FakePort(),rows)
    cfg = config()
    warm = train_warmup(StateArm.GATED_ATTENTION,x.take(0,12),y.take(0,12),cfg,seed=1)
    prior_x = {k:v.clone() for k,v in warm.x_encoder.state_dict().items()}
    prior_y = {k:v.clone() for k,v in warm.y_encoder.state_dict().items()}
    updated, audit = fit_kan_only(
        warm,x.take(0,20),y.take(0,20),x.take(20,28),y.take(20,28),
        cfg,mode=StateArm.GATED_KAN_UPDATE,seed=1,
    )
    assert audit.extra_kan_steps in (0,6)
    assert audit.hypothesis_version >= 0
    assert not hasattr(audit,"encoder_updates")
    assert all(torch.equal(v,updated.x_encoder.state_dict()[k]) for k,v in prior_x.items())
    assert all(torch.equal(v,updated.y_encoder.state_dict()[k]) for k,v in prior_y.items())
    assert not any(p.requires_grad for p in updated.x_encoder.parameters())


def test_bootstrap_trains_heads_only():
    rows = _simulation(StudyLaw.SIGN,11,12,heldout=False)
    x,y = _batch_for(FakePort(),rows)
    cfg = config()
    torch.manual_seed(7)
    initial = TokenKANSystem(StateArm.TOKEN_ATTENTION,96)
    torch.manual_seed(11*1000+59)
    ref = TokenKANSystem(StateArm.TOKEN_ATTENTION,96)
    trained = train_warmup(StateArm.TOKEN_ATTENTION,x,y,cfg,seed=11)
    for k,v in ref.x_encoder.state_dict().items():
        if k in ("warmup_center","warmup_scale"): continue
        assert torch.equal(v,trained.x_encoder.state_dict()[k])
    assert not any(p.requires_grad for p in trained.y_encoder.parameters())


def test_pair_generation_has_no_future_leakage():
    data = _simulation(StudyLaw.PRECISION,20,12,heldout=True)
    negative = _counterfactual_decimal(data)
    assert all(a.x_text == b.x_text and a.y_text != b.y_text
               for a,b in zip(data,negative))
    assert "heldout" not in inspect.signature(fit_kan_only).parameters
    assert "_negative_law_outcomes" not in inspect.getsource(fit_kan_only)


def test_benchmark_runs_no_feedback_and_disjoint_identifiers():
    source = FakePort()
    cfg = config()
    rows = benchmark_one(StudyLaw.OPERATOR,0,cfg,source,
                         model_id="offline",revision="stub")
    assert len(rows) == len(DEFAULT_ARMS)*3
    assert {row.arm for row in rows} == set(DEFAULT_ARMS)
    assert {row.domain for row in rows} == set(EvaluationDomain)
    assert all(row.identity_overlap == 0 for row in rows)
    assert all(row.model_width == 96 for row in rows)
    assert len({row.head_parameters for row in rows}) == 1
    assert all(row.audit.hypothesis_version >= 0 for row in rows)


def test_gated_and_mean_share_initial_projection_and_kan():
    torch.manual_seed(19)
    mean = TokenKANSystem(StateArm.TOKEN_MEAN,96)
    torch.manual_seed(19)
    gated = TokenKANSystem(StateArm.GATED_ATTENTION,96)
    for key in (
        "x_encoder.projection.0.weight",
        "y_encoder.projection.0.weight",
        "hypotheses.0.first.coefficients",
    ):
        assert torch.equal(mean.state_dict()[key],gated.state_dict()[key])
    assert not gated.x_encoder.residual_attention_gate.requires_grad


def test_constant_states_skip_further_hypothesis_updates():
    raw = np.ones((12,3,96),dtype=np.float32)
    states = TensorStates.from_array(TokenStates(
        raw,np.ones((12,3),bool),np.ones((12,96),dtype=np.float32)))
    cfg = config()
    model = train_warmup(StateArm.TOKEN_ATTENTION,states,states,cfg,seed=0)
    assert model.warmup_adequate_spread is False
    updated,audit = fit_kan_only(model,states,states,states,states,cfg,
                                 mode=StateArm.ATTENTION_KAN_UPDATE,seed=0)
    assert audit.skipped_due_to_low_spread
    assert audit.extra_kan_steps == 0
    assert all(torch.equal(updated.state_dict()[k],v)
               for k,v in model.state_dict().items() if k!="weights")
