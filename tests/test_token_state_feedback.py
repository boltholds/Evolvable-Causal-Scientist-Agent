"""TDD gates for full-context Qwen tokens and KAN-to-attention feedback."""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from torch import nn
from torch.nn import functional as F

from ecsa.experimental.token_state_feedback import (
    TokenStates, TokenStatePort, TensorStates, SentenceTransformerTokenPort,
    StateArm, EvaluationDomain, TokenFeedbackConfig, TokenKANSystem,
    StateProjector, _losses, _allow_grad, train_warmup, fit_feedback,
    benchmark_one, _batch_for,
)
from ecsa.experimental.verbalization_study import StudyLaw, _simulation


class FakeTokenizer:
    def __call__(self, value, *, truncation=False, add_special_tokens=True):
        return {"input_ids": list(range(len(value.split()) + 1))}


class FakeTransformer(nn.Module):
    """Deterministic fixed frozen-token model, NOT a pretrained embedding."""
    def __init__(self, width=96):
        super().__init__()
        self.width=width
        torch.manual_seed(7)
        self.embedding=nn.Embedding(2048,width)
        self.embedding.weight.requires_grad_(False)
        self.max_seq_length=256
        self.tokenizer=FakeTokenizer()
        self.forward_count=0

    def get_sentence_embedding_dimension(self): return self.width
    def __getitem__(self,index):
        if index != 0: raise IndexError(index)
        return self
    def tokenize(self,texts):
        # Hash individual characters rather than parse environment data.
        tokenized=[]
        for text in texts:
            result=[min(2047,sum(text[i:i+4].encode('utf8'))%2047+1)
                    for i in range(0,len(text),4)]
            tokenized.append(result[:self.max_seq_length] or [1])
        pad=max(len(x) for x in tokenized)
        ids=torch.zeros((len(texts),pad),dtype=torch.long)
        mask=torch.zeros_like(ids)
        for i,vec in enumerate(tokenized):
            ids[i,:len(vec)]=torch.as_tensor(vec,dtype=torch.long)
            mask[i,:len(vec)]=1
        return {"input_ids":ids,"attention_mask":mask}

    def forward(self, features):
        self.forward_count+=1
        emb=self.embedding(features['input_ids'])
        length=features['attention_mask'].sum(1)
        # Last-token state includes an average summary of preceding tokens.
        sent=(emb[torch.arange(len(emb)),length-1]+
              .5*(emb*features['attention_mask'].unsqueeze(-1)).sum(1)/length.unsqueeze(1))
        return {'token_embeddings':emb, 'attention_mask':features['attention_mask'],
                'sentence_embedding':F.normalize(sent,dim=1)}


def port(max_seq=256,batch=5):
    return SentenceTransformerTokenPort('offline-fake',model=FakeTransformer(96),
         max_seq_length=max_seq,batch_size=batch,require_audited_lengths=False)


def config():
    return TokenFeedbackConfig(bootstrap=12,adaptation=8,calibration=8,
        heldout=9,warmup_steps=10,feedback_steps=6,checkpoint_every=3,
        tolerance=10.,min_spread=.001)


def test_port_produces_full_embeddings_masks_64_dim_projection_and_caches():
    source=port()
    assert isinstance(source, TokenStatePort)
    texts=('one two three','four five','one two three')
    data=source.encode_many(texts)
    assert data.tokens.shape[0]==3 and data.tokens.shape[2]==96
    assert data.sentence.shape==(3,96)
    assert data.mask.dtype==np.bool_
    assert np.allclose(data.tokens[0],data.tokens[2])
    assert not data.mask[1,-1] or len(data.mask[1])==len(data.mask[0])
    count=source.forward_calls
    source.encode_many(texts)
    assert count==source.forward_calls
    assert source.model.forward_count==count
    assert source.verified_length_count == 2
    assert np.any(data.sentence[0,64:]!=0)


def test_token_length_preflight_rejects_silent_truncation():
    source=port(max_seq=8)
    with pytest.raises(ValueError, match='exceeds max_seq_length'):
        source.encode_many(('too many individual words to fit within a short sequence here',))
    assert source.forward_calls==0


def test_token_mask_removes_padding_from_mean_and_learned_attention():
    src=TokenStates(np.random.default_rng(3).normal(size=(2,4,96)).astype(np.float32),
                    np.array([[1,1,0,0],[1,0,0,0]],dtype=bool),
                    np.random.default_rng(5).normal(size=(2,96)).astype(np.float32))
    one=TensorStates.from_array(src)
    edited=TensorStates(one.tokens.clone(),one.mask,one.sentence)
    edited.tokens[~edited.mask]=1000.
    for arm in (StateArm.TOKEN_MEAN,StateArm.TOKEN_ATTENTION):
        torch.manual_seed(4)
        model=StateProjector(arm,96)
        left=model(one)
        right=model(edited)
        assert torch.allclose(left,right,atol=1e-6)
        assert torch.all(model.weights(one)[~one.mask]==0)


def test_all_attention_head_dimensions_and_modes_share_kan_hypotheses():
    a=TokenKANSystem(StateArm.SENTENCE_64,96)
    b=TokenKANSystem(StateArm.SENTENCE_FULL,96)
    c=TokenKANSystem(StateArm.TOKEN_ATTENTION,96)
    assert a.x_encoder is not a.y_encoder
    assert a.x_encoder.projection[0].in_features==64
    assert b.x_encoder.projection[0].in_features==96
    assert c.x_encoder.token_attention is not None
    assert len({id(k) for k in c.hypotheses})==3
    assert sum(p.numel() for p in a.hypotheses.parameters())==sum(p.numel() for p in b.hypotheses.parameters())
    assert sum(p.numel() for p in a.hypotheses.parameters())==sum(p.numel() for p in c.hypotheses.parameters())


def test_kan_feedback_backpropagates_to_attention_not_frozen_hypotheses():
    rows=_simulation(StudyLaw.OPERATOR, seed=9,n=12,heldout=False)
    x,y=_batch_for(port(),rows)
    model=TokenKANSystem(StateArm.TOKEN_ATTENTION,96)
    _allow_grad(model,heads=False,encoders=True)
    loss=_losses(model,x,y,1).mean()
    loss.backward()
    assert model.x_encoder.token_attention.weight.grad is not None
    assert model.x_encoder.token_attention.weight.grad.abs().sum()>0
    assert model.y_encoder.token_attention.weight.grad.abs().sum()>0
    assert all(v.grad is None for v in model.hypotheses.parameters())


def test_model_versions_control_parameter_mutations_and_reject_gates(monkeypatch):
    rows=_simulation(StudyLaw.PRECISION,seed=6,n=28,heldout=False)
    xx,yy=_batch_for(port(),rows)
    cfg=config()
    warm=train_warmup(StateArm.TOKEN_ATTENTION,xx.take(0,12),yy.take(0,12),cfg,seed=1)
    initial_head=[v.detach().clone() for v in warm.hypotheses.parameters()]
    initial_attention=warm.x_encoder.token_attention.weight.detach().clone()
    adapted,meta=fit_feedback(warm,xx.take(0,20),yy.take(0,20),
                               xx.take(20,28),yy.take(20,28),cfg,
                               mode=StateArm.ATTENTION_FEEDBACK,seed=1)
    assert meta.feedback_steps==6 and meta.head_updates==0 and meta.encoder_updates==6
    assert meta.encoder_version>0 and meta.hypothesis_version==0
    assert torch.equal(initial_head[0],next(adapted.hypotheses.parameters()))
    assert not torch.equal(initial_attention,adapted.x_encoder.token_attention.weight)
    updated,other=fit_feedback(warm,xx.take(0,20),yy.take(0,20),
                               xx.take(20,28),yy.take(20,28),cfg,
                               mode=StateArm.ATTENTION_KAN_UPDATE,seed=1)
    assert other.head_updates==6 and other.encoder_updates==0
    assert other.hypothesis_version>0 and other.encoder_version==0
    assert torch.equal(initial_attention,updated.x_encoder.token_attention.weight)
    # Strict calibration gate rejects any change; rollback is exact.
    from ecsa.experimental import token_state_feedback as module
    monkeypatch.setattr(module,'_calibration_loss',lambda *args: float('nan'))
    rolled,info=fit_feedback(warm,xx.take(0,20),yy.take(0,20),
                               xx.take(20,28),yy.take(20,28),cfg,
                               mode=StateArm.ATTENTION_FEEDBACK,seed=1)
    assert info.accepted_checkpoints==0 and info.rejected_checkpoints==2
    assert info.encoder_version==0
    assert all(torch.equal(warm.state_dict()[k],v)
               for k,v in rolled.state_dict().items() if k != "weights")


def test_no_heldout_leakage_in_training_and_eval_domains():
    assert 'heldout' not in inspect.signature(fit_feedback).parameters
    assert '_negative_law_outcomes' not in inspect.getsource(fit_feedback)
    source=port()
    cfg=config()
    rows=benchmark_one(StudyLaw.OPERATOR,seed=0,config=cfg,port=source,
                       model_id='fake',revision='fake')
    assert len(rows)==21
    assert {r.arm for r in rows}==set(StateArm)
    assert {r.domain for r in rows}==set(EvaluationDomain)
    assert {r.arm for r in rows if r.arm is StateArm.NUMERIC_CONTROL}=={StateArm.NUMERIC_CONTROL}
    assert all(r.identity_overlap==0 and r.train_pairs==20 and r.validation_pairs==8 for r in rows)
    assert all(r.head_parameters==rows[0].head_parameters for r in rows)
    assert all(r.model_width==96 for r in rows)
    assert all(0<=r.auroc<=1 and 0<=r.brier<=1 for r in rows)
    assert next(r for r in rows if r.arm is StateArm.ATTENTION_FEEDBACK).audit.encoder_updates==6


def test_outcome_changes_do_not_modify_candidate_x():
    from ecsa.experimental.token_state_feedback import _counterfactual_decimal
    samples=_simulation(StudyLaw.PRECISION,seed=20,n=12,heldout=True)
    changed=_counterfactual_decimal(samples)
    assert all(x.x_text==y.x_text and x.y_text!=y.y_text for x,y in zip(samples,changed))


def test_reject_malformed_token_masks_and_mode():
    with pytest.raises(ValueError):
        TokenStates(np.zeros((1,2,96),np.float32),np.zeros((1,2),bool),np.zeros((1,96),np.float32))
    with pytest.raises(TypeError): TokenKANSystem('attention',96)


def test_sentence_truncation_and_full_context_have_distinct_information_gates():
    """Last-64 vs full-1024 is a testable bottleneck, not an assertion about Qwen."""
    width=96
    text=torch.zeros((2,2,width))
    mask=torch.ones((2,2),dtype=torch.bool)
    sentence=torch.zeros((2,width))
    sentence[0,70]=1.
    sentence[1,70]=-1.
    states=TensorStates(text,mask,sentence)
    torch.manual_seed(44)
    tiny=StateProjector(StateArm.SENTENCE_64,width)
    full=StateProjector(StateArm.SENTENCE_FULL,width)
    assert torch.allclose(tiny(states)[0],tiny(states)[1])
    assert not torch.allclose(full(states)[0],full(states)[1])
    # Same full sentence vector, different token context -> attention sees it.
    token1=text.clone()
    token1[0,0,23]=5.
    token1[1,0,23]=-5.
    probe=StateProjector(StateArm.TOKEN_ATTENTION,width)
    altered=TensorStates(token1,mask,torch.zeros_like(sentence))
    assert not torch.allclose(probe(altered)[0],probe(altered)[1])


def test_warmup_only_whitening_rescues_clustered_full_embeddings():
    """Regression: original projection collapsed Qwen full/token representations."""
    n,d=16,96
    base=np.linspace(-2.,2.,d,dtype=np.float32)
    sentence=np.broadcast_to(base,(n,d)).copy()
    rng=np.random.default_rng(101)
    sentence += rng.normal(0,0.002,(n,d)).astype(np.float32)
    sentence[:,72] += np.linspace(-.006,.006,n).astype(np.float32)
    tensor=TokenStates(sentence[:,None,:].copy(),np.ones((n,1),dtype=bool),
                       sentence.copy())
    state=TensorStates.from_array(tensor)
    torch.manual_seed(1)
    model=StateProjector(StateArm.SENTENCE_FULL,d)
    model.fit_warmup_normalizer(state)
    projected=model(state)
    assert float(projected.detach().std(0,unbiased=False).mean()) > .02
    assert torch.all(model.warmup_scale>=.001)
    center=model.warmup_center.detach().clone()
    # A different evaluation distribution cannot change training moments.
    model(state.shifted(1))
    assert torch.equal(model.warmup_center,center)


def test_attention_retains_content_when_special_token_is_shared():
    """Prevent the real-model bug: selecting a common token removes all variance."""
    torch.manual_seed(21)
    n,width=16,96
    token=torch.zeros((n,3,width))
    token[:,0,10]=torch.linspace(-1,1,n)
    token[:,1,20]=torch.linspace(.1,.8,n)
    token[:,2,30]=12.  # identical strongly activated common token
    mask=torch.ones((n,3),dtype=torch.bool)
    states=TensorStates(token,mask,torch.zeros((n,width)))
    model=StateProjector(StateArm.TOKEN_ATTENTION,width)
    with torch.no_grad():
        model.token_attention.weight.zero_()
        model.token_attention.weight[0,30]=6.
    # Bounded attention retains nonzero probability for content tokens.
    masses=model.weights(states)
    assert float(masses[:,:2].detach().sum(dim=1).min()) > .01
    model.fit_warmup_normalizer(states)
    projected=model(states).detach()
    assert float(projected.std(0,unbiased=False).mean()) > .02


def test_genuinely_constant_observations_are_flagged_without_fake_learning():
    """No observed variation must not be reported as successful KAN feedback."""
    width=96
    token=np.ones((12,3,width),dtype=np.float32)
    observed=TokenStates(token,np.ones((12,3),dtype=bool),
                         np.ones((12,width),dtype=np.float32))
    samples=TensorStates.from_array(observed)
    cfg=TokenFeedbackConfig(bootstrap=12,adaptation=8,calibration=8,
                            heldout=9,warmup_steps=8,feedback_steps=4,
                            checkpoint_every=2)
    model=train_warmup(StateArm.TOKEN_ATTENTION,samples,samples,cfg,seed=0)
    assert model.warmup_adequate_spread is False
    updated,audit=fit_feedback(model,samples,samples,samples,samples,cfg,
                               mode=StateArm.ATTENTION_FEEDBACK,seed=0)
    assert audit.skipped_due_to_low_spread is True
    assert audit.feedback_steps==audit.encoder_updates==0
    assert audit.encoder_version==0
    assert all(torch.equal(model.state_dict()[k],v)
               for k,v in updated.state_dict().items() if k!='weights')
