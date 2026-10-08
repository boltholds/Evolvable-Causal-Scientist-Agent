"""Offline RED/GREEN gates for frozen pretrained world-model encoder ports.

The fake model is NEVER reported as a pretrained BehR run.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch=pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

from ecsa.experimental.world_model_encoders import (
    CausalLMConfig, Candidate, FrozenCausalLMTokenPort,
    SPECS, _token_ids, next_state_prompt,
    evaluate_next_state_likelihood,
)
from ecsa.experimental.world_model_encoder_benchmark import (
    run_study, _same_heldout,
)
from ecsa.experimental.token_state_relations import StateArm,TokenRelationConfig
from ecsa.experimental.verbalization_study import StudyLaw


class TinyTokenizer:
    """Stable fixed-width byte chunks. NOT a pretrained tokenizer."""
    def __call__(self,text,*,add_special_tokens=True,truncation=False,**kwargs):
        assert truncation is False
        chunks=[text[i:i+6] for i in range(0,len(text),6)]
        ids=[(sum(chunk.encode("utf8"))%97)+2 for chunk in chunks]
        return {"input_ids":([1] if add_special_tokens else []) + (ids or [2])}


class TinyLM(nn.Module):
    """Tiny deterministic causal token model with real gradients disabled."""
    def __init__(self,width=96,vocab=120):
        super().__init__()
        torch.manual_seed(17)
        self.embedding=nn.Embedding(vocab,width)
        self.output=nn.Linear(width,vocab,bias=False)
        self.config=SimpleNamespace(hidden_size=width,_commit_hash="fake-commit")
        self.calls=0

    def forward(self,input_ids,attention_mask=None,output_hidden_states=False,use_cache=False):
        assert attention_mask is None or attention_mask.shape==input_ids.shape
        self.calls+=1
        h=self.embedding(input_ids)
        logits=self.output(h)
        return SimpleNamespace(
            hidden_states=(h,) if output_hidden_states else None,
            logits=logits,
        )


def fake_port(max_length=512):
    return FrozenCausalLMTokenPort(
        CausalLMConfig(
            candidate=Candidate.SMALL_QWEN25,device="cpu",
            max_length=max_length,
        ), model=TinyLM(),tokenizer=TinyTokenizer(),
    )


def fixture_config():
    return TokenRelationConfig(
        bootstrap=8,adaptation=5,calibration=4,heldout=8,
        warmup_steps=8,extra_kan_steps=4,checkpoint_every=2,
    )


def test_pretrained_model_registry_and_agentworld_separation():
    assert SPECS[Candidate.BEHR_TEXTWORLD].model_id=="Ricardo-H/BehR-WorldModel-Textworld-Qwen2.5-7B"
    assert SPECS[Candidate.BEHR_TEXTWORLD].kind=="causal_hidden_states"
    assert SPECS[Candidate.AGENTWORLD].kind=="remote_simulator_only"
    with pytest.raises(ValueError):
        FrozenCausalLMTokenPort(CausalLMConfig(candidate=Candidate.AGENTWORLD),
                               model=TinyLM(),tokenizer=TinyTokenizer())


def test_large_cpu_checkpoint_is_guarded_before_hf_downloads():
    with pytest.raises(RuntimeError,match="7B CPU"):
        FrozenCausalLMTokenPort(CausalLMConfig(
            candidate=Candidate.BEHR_TEXTWORLD,device="cpu"))
    with pytest.raises(ValueError):
        CausalLMConfig(quantized_4bit=True,device="cpu")
    with pytest.raises(ValueError):
        CausalLMConfig(max_length=10)
    with pytest.raises(RuntimeError,match="CUDA"):
        if not torch.cuda.is_available():
            FrozenCausalLMTokenPort(CausalLMConfig(
                candidate=Candidate.BEHR_TEXTWORLD,device="cuda"))
        else:
            raise RuntimeError("CUDA")


def test_hidden_state_port_is_frozen_batched_and_caches_repeated_texts():
    port=fake_port()
    token=port.encode_many(("first", "a longer observation", "first"))
    assert token.tokens.ndim==3
    assert token.width==96
    assert token.mask.dtype==np.bool_
    assert token.mask[0].sum()<token.mask[1].sum()
    assert np.array_equal(token.tokens[0],token.tokens[2])
    assert port.model.calls==2
    assert port.backbone_forward_batches==2
    assert port.revision=="fake-commit"
    assert all(not parameter.requires_grad for parameter in port.model.parameters())
    assert all(parameter.grad is None for parameter in port.model.parameters())


def test_token_mask_no_future_leak_and_no_silent_truncation():
    port=fake_port(max_length=50)
    with pytest.raises(ValueError,match="truncation"):
        port.encode_many(("x"*350,))
    raw='{"before":{"pressure":7},"action":{"schema_id":"move"}}'
    prompt=next_state_prompt(raw)
    assert raw in prompt
    assert '"after"' not in prompt
    with pytest.raises(ValueError,match="max_length"):
        port.conditional_nll(raw,'{"after":{"payload":"'+'x'*300+'"}}')


def test_target_nll_is_shifted_teacher_forced_not_self_prediction():
    port=fake_port(max_length=512)
    x='{"before":{"state":4},"action":{"schema_id":"press"}}'
    y='{"after":{"result":5}}'
    result=port.conditional_nll(x,y)
    prefix=_token_ids(port.tokenizer,next_state_prompt(x),specials=True)
    target=_token_ids(port.tokenizer,y,specials=False)
    with torch.inference_mode():
        ids=torch.tensor([prefix+target],dtype=torch.long)
        logits=port.model(ids).logits[0,len(prefix)-1:len(prefix)+len(target)-1].float()
        expected=F.cross_entropy(logits,torch.tensor(target),reduction="mean").item()
    assert result==pytest.approx(expected,rel=1e-6)
    assert port.conditional_nll(x,y)==result
    assert np.isfinite(result) and result>0
    assert all(param.grad is None for param in port.model.parameters())


def test_likelihood_evaluator_filters_identical_and_rejects_modified_x():
    port=fake_port(max_length=512)
    positive,negative=_same_heldout(StudyLaw.OPERATOR,0,8)
    report=evaluate_next_state_likelihood(
        port,positive,negative,law="operator",seed=0,maximum=3)
    assert 1<=report.pairs_examined<=3
    assert report.pair_accuracy is not None
    assert 0<=report.pair_accuracy<=1
    assert report.mean_true_nll is not None
    assert port._nll_cache
    from dataclasses import replace
    modified=tuple(replace(row,x_text=row.x_text+" bad") for row in negative)
    with pytest.raises(ValueError,match="pre-action"):
        evaluate_next_state_likelihood(
            port,positive,modified,law="operator",seed=0,maximum=2)


def test_causal_lm_port_runs_kan_only_same_data_without_trained_encoder():
    port=fake_port(max_length=512)
    c=fixture_config()
    rows=run_study(
        port,config=c,laws=(StudyLaw.OPERATOR,),seeds=(0,),
        arms=(StateArm.SENTENCE_FULL,StateArm.NUMERIC_CONTROL),
        include_likelihood=False,
    )
    assert rows["candidate"]=="small_qwen25"
    assert rows["is_actual_behR"] is False
    assert len(rows["scores"])==2*3
    assert all(row["identity_overlap"]==0 for row in rows["scores"])
    assert len({row["head_parameters"] for row in rows["scores"]})==1
    assert port.model.training is False
    assert all(not p.requires_grad for p in port.model.parameters())
    assert port.model.calls>0


def test_knows_this_is_not_an_embedding_model_or_causal_proof():
    port=fake_port()
    rows=run_study(port,config=fixture_config(),laws=(StudyLaw.OPERATOR,),
                   seeds=(1,),arms=(StateArm.SENTENCE_FULL,),
                   include_latents=False,include_likelihood=True,
                   max_likelihood_pairs=2)
    assert len(rows["generative_likelihood"])==1
    assert rows["generative_likelihood"][0]["pairs_examined"]<=2
    assert rows["scores"]==[]
    assert "is_actual_behR" in rows
    assert any("not a pretrained symmetric embedding" in warning
               for warning in rows["limitations"])


def test_signature_and_rejection_of_wrong_study_flags():
    port=fake_port()
    with pytest.raises(ValueError,match="enable latent"):
        run_study(port,config=fixture_config(),laws=(StudyLaw.OPERATOR,),
                  seeds=(0,),arms=(StateArm.SENTENCE_FULL,),
                  include_latents=False,include_likelihood=False)
    with pytest.raises(ValueError,match="unique"):
        run_study(port,config=fixture_config(),laws=(StudyLaw.OPERATOR,),
                  seeds=(0,),arms=(StateArm.SENTENCE_FULL,StateArm.SENTENCE_FULL))
