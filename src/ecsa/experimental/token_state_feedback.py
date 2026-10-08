"""Frozen Qwen BPE-token states -> attention pooling -> competing spline KAN.

A controlled experiment: retain all contextual token vectors, compare against
Qwen's full or 64-d sentence embeddings, and train a *small* token-attention
aggregator from frozen KAN hypotheses. The backbone is never fine-tuned.

All decisions and checkpoint gates use training/calibration observations only.
Counterfactual heldout outcomes are generated in the BENCHMARK evaluator and
are never passed to warmup, feedback optimization or acceptance gates.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .pretrained_text_study import DEFAULT_MODEL
from .relation_discovery import SplineKANHead
from .text_first_kan import _auc, _numeric_hash
from .verbalization_study import StudyLaw, _simulation, _negative_law_outcomes, _paraphrase
from ecsa.world_model.text_relations import TextRelationSample


DEFAULT_QWEN_REVISION = "3d106eabb5535a84de3ae88f45887a78259b52de"


class StateArm(StrEnum):
    SENTENCE_64 = "sentence_64"
    NUMERIC_CONTROL = "numeric_control"
    SENTENCE_FULL = "sentence_full"
    TOKEN_MEAN = "token_mean"
    TOKEN_ATTENTION = "token_attention"
    ATTENTION_KAN_UPDATE = "attention_kan_update"
    ATTENTION_FEEDBACK = "attention_feedback"


class EvaluationDomain(StrEnum):
    COUNTERFACTUAL = "counterfactual"
    PARAPHRASE = "paraphrase"
    SMALL_NUMERIC_DELTA = "small_numeric_delta"


@dataclass(frozen=True)
class TokenStates:
    """Full final-layer contextual embeddings, original mask and full sentence vector."""
    tokens: np.ndarray
    mask: np.ndarray
    sentence: np.ndarray

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3 or self.mask.shape != self.tokens.shape[:2]:
            raise ValueError("token tensor and mask shape mismatch")
        if self.sentence.shape != (len(self.tokens), self.tokens.shape[-1]):
            raise ValueError("sentence embeddings must match full token width")
        if self.mask.dtype != np.bool_:
            raise ValueError("boolean token mask required")
        if len(self.mask) and not self.mask.any(axis=1).all():
            raise ValueError("empty token sequence")
        if not np.isfinite(self.tokens).all() or not np.isfinite(self.sentence).all():
            raise ValueError("nonfinite token or sentence representations")

    @property
    def width(self) -> int:
        return int(self.tokens.shape[-1])

    def take(self, indices: slice | np.ndarray) -> "TokenStates":
        return TokenStates(self.tokens[indices], self.mask[indices], self.sentence[indices])


@runtime_checkable
class TokenStatePort(Protocol):
    @property
    def dimension(self) -> int: ...
    def encode_many(self, texts: tuple[str, ...]) -> TokenStates: ...


class SentenceTransformerTokenPort:
    """One frozen SentenceTransformer pass yields both sentence and token vectors.

    Inject `model=` for offline tests; a real Qwen SentenceTransformer is loaded
    only when explicitly constructing the production-backed port. A full-token
    preflight rejects truncation rather than silently dropping numeric evidence.
    """

    def __init__(
        self, model_id: str = DEFAULT_MODEL, *, revision: str | None = None,
        device: str = "cpu", max_seq_length: int = 256, batch_size: int = 4,
        model: object | None = None, require_audited_lengths: bool = True,
    ) -> None:
        if not model_id or type(max_seq_length) is not int or max_seq_length < 8:
            raise ValueError("valid model ID and token limit are required")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("positive batch size required")
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_id, revision=revision,
                                        device=device, trust_remote_code=False)
        self.model = model
        self.model.max_seq_length = max_seq_length
        self.model.eval()
        self.model_id = model_id
        self.revision = revision
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.require_audited_lengths = require_audited_lengths
        self._dimension = int(self.model.get_sentence_embedding_dimension())
        if self._dimension < 64:
            raise ValueError("token model needs at least 64 dimensions")
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.max_observed_length = 0
        self.forward_calls = 0
        self.verified_length_count = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def _preflight(self, texts: tuple[str, ...]) -> None:
        try:
            tokenizer = self.model[0].tokenizer
        except (TypeError, IndexError, AttributeError):
            if self.require_audited_lengths:
                raise RuntimeError("token-length audit unavailable; refusing silent truncation")
            return
        for text in texts:
            sequence = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
            length = len(sequence)
            self.max_observed_length = max(length, self.max_observed_length)
            self.verified_length_count += 1
            if length > self.max_seq_length:
                raise ValueError(f"token sequence {length} exceeds max_seq_length={self.max_seq_length}")

    def encode_many(self, texts: tuple[str, ...]) -> TokenStates:
        if not isinstance(texts, tuple) or any(type(x) is not str for x in texts):
            raise TypeError("immutable tuple of texts required")
        if not texts:
            raise ValueError("cannot encode an empty collection")
        fresh = tuple(dict.fromkeys(t for t in texts if t not in self._cache))
        if fresh:
            self._preflight(fresh)
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            for offset in range(0, len(fresh), self.batch_size):
                items = fresh[offset:offset+self.batch_size]
                raw = self.model.tokenize(list(items))
                features = {key: value.to(device) if isinstance(value, Tensor) else value
                            for key, value in raw.items()}
                with torch.inference_mode():
                    encoded = self.model(features)
                self.forward_calls += 1
                token = encoded["token_embeddings"].detach().float().cpu().numpy()
                mask = encoded["attention_mask"].detach().cpu().numpy().astype(bool)
                sentence = encoded["sentence_embedding"].detach().float().cpu().numpy()
                if token.shape[:2] != mask.shape or token.shape[-1] != self.dimension:
                    raise ValueError("invalid contextual-token shape")
                if sentence.shape != (len(items), self.dimension):
                    raise ValueError("invalid full-sentence shape")
                for j, text in enumerate(items):
                    valid = int(mask[j].sum())
                    if valid < 1 or valid > self.max_seq_length:
                        raise ValueError("invalid token mask or truncation")
                    self._cache[text] = (
                        token[j, mask[j]].copy(),
                        F.normalize(torch.from_numpy(sentence[j].copy()),dim=0).numpy(),
                    )
        length = max(len(self._cache[t][0]) for t in texts)
        result = np.zeros((len(texts),length,self.dimension), dtype=np.float32)
        mask = np.zeros((len(texts),length), dtype=bool)
        sentence = np.zeros((len(texts),self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            token_vec, sent = self._cache[text]
            n = len(token_vec)
            result[i,:n] = token_vec
            mask[i,:n] = True
            sentence[i] = sent
        return TokenStates(result,mask,sentence)


@dataclass(frozen=True)
class TokenFeedbackConfig:
    bootstrap: int = 24
    adaptation: int = 16
    calibration: int = 12
    heldout: int = 24
    warmup_steps: int = 45
    feedback_steps: int = 24
    checkpoint_every: int = 8
    warmup_lr: float = 0.012
    feedback_lr: float = 0.005
    preservation_weight: float = 0.35
    variance_weight: float = 0.12
    min_spread: float = 0.006
    tolerance: float = 0.06

    def __post_init__(self) -> None:
        for key in ("bootstrap", "adaptation", "calibration", "heldout", "warmup_steps",
                    "feedback_steps", "checkpoint_every"):
            val = getattr(self,key)
            if type(val) is not int or val < 2:
                raise ValueError(f"{key} needs >= 2")
        if self.checkpoint_every > self.feedback_steps:
            raise ValueError("feedback needs at least one checkpoint")
        for key in ("warmup_lr","feedback_lr", "preservation_weight","variance_weight", "min_spread", "tolerance"):
            val = getattr(self,key)
            if not np.isfinite(val) or val < 0 or ("lr" in key and val == 0):
                raise ValueError(f"invalid {key}")


@dataclass(frozen=True)
class TensorStates:
    tokens: Tensor
    mask: Tensor
    sentence: Tensor

    @classmethod
    def from_array(cls, states: TokenStates) -> "TensorStates":
        return cls(torch.from_numpy(states.tokens.copy()),
                   torch.from_numpy(states.mask.copy()),
                   torch.from_numpy(states.sentence.copy()))

    def take(self, start: int, stop: int) -> "TensorStates":
        return TensorStates(self.tokens[start:stop], self.mask[start:stop],
                            self.sentence[start:stop])

    def shifted(self, shift: int) -> "TensorStates":
        return TensorStates(torch.roll(self.tokens,shift,0),
                            torch.roll(self.mask,shift,0),
                            torch.roll(self.sentence,shift,0))

    def __len__(self) -> int:
        return len(self.sentence)


class StateProjector(nn.Module):
    def __init__(self, mode: StateArm, width: int) -> None:
        super().__init__()
        self.mode = mode
        self.width = width
        self.token_attention = (
            nn.Linear(width, 1, bias=False) if mode in (
                StateArm.TOKEN_ATTENTION, StateArm.ATTENTION_KAN_UPDATE,
                StateArm.ATTENTION_FEEDBACK
            ) else None
        )
        self.projection = nn.Sequential(
            nn.Linear(64 if mode in (StateArm.SENTENCE_64, StateArm.NUMERIC_CONTROL) else width, 12),
            nn.Tanh(), nn.Linear(12, 2), nn.Tanh(),
        )

    def weights(self, states: TensorStates) -> Tensor:
        if self.token_attention is None:
            return states.mask.float() / states.mask.float().sum(1,keepdim=True)
        logits = self.token_attention(F.layer_norm(states.tokens,(self.width,))).squeeze(-1)
        logits = logits.masked_fill(~states.mask, torch.finfo(logits.dtype).min)
        return torch.softmax(logits,dim=1)

    def forward(self, states: TensorStates) -> Tensor:
        if self.mode in (StateArm.SENTENCE_64, StateArm.NUMERIC_CONTROL):
            vector = F.normalize(states.sentence[:,:64],dim=1)
        elif self.mode is StateArm.SENTENCE_FULL:
            vector = states.sentence
        else:
            attention = self.weights(states)
            vector = torch.einsum("bt,btd->bd",attention,states.tokens)
            vector = F.normalize(vector,dim=1)
        vector = F.layer_norm(vector,(vector.shape[-1],))
        return self.projection(vector)


class TokenKANSystem(nn.Module):
    def __init__(self, mode: StateArm, width: int) -> None:
        super().__init__()
        if not isinstance(mode,StateArm):
            raise TypeError("typed state-encoder arm required")
        self.x_encoder = StateProjector(mode,width)
        self.y_encoder = StateProjector(mode,width)
        self.hypotheses = nn.ModuleList((
            SplineKANHead(knots=5,hidden=12),
            SplineKANHead(knots=7,hidden=9),
            SplineKANHead(knots=9,hidden=7),
        ))
        self.register_buffer("weights", torch.full((3,),1/3))
        self.encoder_version = 0
        self.hypothesis_version = 0

    def logits(self,x:TensorStates,y:TensorStates) -> Tensor:
        joint = torch.cat((self.x_encoder(x),self.y_encoder(y)),dim=1)
        return torch.stack([head(joint).flatten() for head in self.hypotheses])

    @torch.no_grad()
    def probability(self,x:TensorStates,y:TensorStates) -> Tensor:
        return torch.einsum("h,hb->b",self.weights,torch.sigmoid(self.logits(x,y)))

    @torch.no_grad()
    def min_spread(self,x:TensorStates,y:TensorStates) -> float:
        return float(min(self.x_encoder(x).std(0,unbiased=False).mean(),
                         self.y_encoder(y).std(0,unbiased=False).mean()))


def _losses(model: TokenKANSystem, x:TensorStates, y:TensorStates, shift:int) -> Tensor:
    if len(x) < 2 or len(y) != len(x):
        raise ValueError("matched observed pairs >= 2 are required")
    positive = model.logits(x,y)
    negative = model.logits(x,y.shifted(shift))
    return (F.softplus(-positive).mean(1)+F.softplus(negative).mean(1))/2


@torch.no_grad()
def _refresh_weights(model:TokenKANSystem,x:TensorStates,y:TensorStates) -> None:
    nll = _losses(model,x,y,1)
    posterior = (-(nll-nll.min())*3).softmax(dim=0)
    model.weights.copy_(posterior*0.9 + 0.1/len(posterior))


@torch.no_grad()
def _calibration_loss(model:TokenKANSystem,x:TensorStates,y:TensorStates) -> float:
    p1=model.probability(x,y).clamp(1e-6,1-1e-6)
    p0=model.probability(x,y.shifted(1)).clamp(1e-6,1-1e-6)
    return float((-torch.log(p1).mean()-torch.log1p(-p0).mean())/2)


def _collapse(model:TokenKANSystem,x:TensorStates,y:TensorStates, threshold:float) -> Tensor:
    za,zb=model.x_encoder(x),model.y_encoder(y)
    return (F.relu(threshold-za.std(0,unbiased=False)).square().mean()
            +F.relu(threshold-zb.std(0,unbiased=False)).square().mean())


@dataclass(frozen=True)
class FitAudit:
    warmup_steps: int
    feedback_steps: int
    head_updates: int
    encoder_updates: int
    accepted_checkpoints: int
    rejected_checkpoints: int
    initial_calibration_loss: float
    final_calibration_loss: float
    initial_spread: float
    final_spread: float
    encoder_version: int
    hypothesis_version: int


def train_warmup(mode:StateArm,x:TensorStates,y:TensorStates,config:TokenFeedbackConfig,
                 seed:int) -> TokenKANSystem:
    torch.set_num_threads(1)
    torch.manual_seed(seed*1000+59)
    model=TokenKANSystem(mode,x.tokens.shape[-1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.warmup_lr)
    rng=np.random.default_rng(seed*1000+65)
    for _ in range(config.warmup_steps):
        shift=int(rng.integers(1,len(x)))
        losses=_losses(model,x,y,shift)
        objective=losses.mean()+config.variance_weight*_collapse(model,x,y,0.1)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    model.eval()
    return model


def _allow_grad(model:TokenKANSystem, *, heads:bool, encoders:bool) -> None:
    for item in model.hypotheses.parameters():
        item.requires_grad_(heads); item.grad=None
    for item in (*model.x_encoder.parameters(),*model.y_encoder.parameters()):
        item.requires_grad_(encoders); item.grad=None


def fit_feedback(warmup:TokenKANSystem,x:TensorStates,y:TensorStates,
                 cx:TensorStates,cy:TensorStates,config:TokenFeedbackConfig,
                 *,mode:StateArm,seed:int) -> tuple[TokenKANSystem,FitAudit]:
    """No evaluation states or counterfactual generator are accepted."""
    model=copy.deepcopy(warmup)
    _refresh_weights(model,cx,cy)
    before=_calibration_loss(model,cx,cy)
    before_spread=model.min_spread(cx,cy)
    head_only=mode is StateArm.ATTENTION_KAN_UPDATE
    encoder_only=mode is StateArm.ATTENTION_FEEDBACK
    if not (head_only or encoder_only):
        _allow_grad(model,heads=False,encoders=False)
        return model,FitAudit(config.warmup_steps,0,0,0,0,0,before,before,
                              before_spread,before_spread,0,0)
    if warmup.x_encoder.mode is not StateArm.TOKEN_ATTENTION:
        raise ValueError("feedback controls require attention-pool warmup")
    _allow_grad(model,heads=head_only,encoders=encoder_only)
    params=[p for p in model.parameters() if p.requires_grad]
    reference=copy.deepcopy(warmup)
    _allow_grad(reference,heads=False,encoders=False)
    with torch.no_grad():
        anchor_x=reference.x_encoder(x);anchor_y=reference.y_encoder(y)
    rng=np.random.default_rng(seed*1000+103)
    accepted=rejected=0
    for start in range(0,config.feedback_steps,config.checkpoint_every):
        n=min(config.checkpoint_every,config.feedback_steps-start)
        snapshot=copy.deepcopy(model.state_dict())
        base_nll=_calibration_loss(model,cx,cy)
        old_spread=model.min_spread(cx,cy)
        optimizer=torch.optim.AdamW(params,lr=config.feedback_lr)
        for _ in range(n):
            loss=(_losses(model,x,y,int(rng.integers(1,len(x))))*model.weights.detach()).sum()
            if encoder_only:
                loss=(loss + config.preservation_weight*
                      (F.mse_loss(model.x_encoder(x),anchor_x)+F.mse_loss(model.y_encoder(y),anchor_y))/2
                      +config.variance_weight*_collapse(model,x,y,0.10))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params,1.)
            optimizer.step()
        _refresh_weights(model,cx,cy)
        candidate_nll=_calibration_loss(model,cx,cy)
        candidate_spread=model.min_spread(cx,cy)
        if (np.isfinite(candidate_nll) and np.isfinite(candidate_spread)
            and candidate_nll<=base_nll+config.tolerance
            and candidate_spread >= min(config.min_spread,old_spread*.5)):
            accepted+=1
            if encoder_only: model.encoder_version+=1
            else: model.hypothesis_version+=1
        else:
            rejected+=1
            model.load_state_dict(snapshot)
    _allow_grad(model,heads=False,encoders=False)
    model.eval()
    return model, FitAudit(
        config.warmup_steps,config.feedback_steps,
        config.feedback_steps if head_only else 0,
        config.feedback_steps if encoder_only else 0,
        accepted,rejected,before,_calibration_loss(model,cx,cy),
        before_spread,model.min_spread(cx,cy),
        model.encoder_version,model.hypothesis_version,
    )


def _batch_for(port:TokenStatePort,samples:tuple[TextRelationSample,...]) -> tuple[TensorStates,TensorStates]:
    if not samples:
        raise ValueError("nonempty sample collection required")
    text=tuple(row.x_text for row in samples)+tuple(row.y_text for row in samples)
    states=port.encode_many(text)
    n=len(samples)
    return (TensorStates.from_array(states.take(slice(0,n))),
            TensorStates.from_array(states.take(slice(n,None))))


def _numeric_batch(samples:tuple[TextRelationSample,...], width:int) -> tuple[TensorStates,TensorStates]:
    """Existing numeric-hash baseline; no benchmark-specific field selection."""
    text=tuple(row.x_text for row in samples)+tuple(row.y_text for row in samples)
    arr=np.zeros((len(text),width),dtype=np.float32)
    arr[:,:64]=np.stack([_numeric_hash(x,64) for x in text])
    data=TokenStates(arr[:,None,:].copy(),np.ones((len(text),1),dtype=bool),arr)
    n=len(samples)
    return TensorStates.from_array(data.take(slice(0,n))), TensorStates.from_array(data.take(slice(n,None)))


def _counterfactual_decimal(samples:tuple[TextRelationSample,...]) -> tuple[TextRelationSample,...]:
    """Post-split evaluation-only measurement perturbations; same X exactly."""
    from dataclasses import replace
    out=[]
    for sample in samples:
        payload=json.loads(sample.y_text)
        old=float(payload['after']['measurement'])
        payload['after']['measurement']=float(f"{(old + (.20 if old>=0 else -.20)):.5f}")
        out.append(replace(sample,y_text=json.dumps(payload,sort_keys=True,
                                            ensure_ascii=False,separators=(',',':'))))
    return tuple(out)


@dataclass(frozen=True)
class Score:
    law:str
    seed:int
    arm:StateArm
    domain:EvaluationDomain
    auroc:float
    paired_accuracy:float
    brier:float
    train_pairs:int
    validation_pairs:int
    heldout_pairs:int
    identity_overlap:int
    pretrained_model:str
    pretrained_revision:str | None
    model_width:int
    trainable_parameters:int
    attention_parameters:int
    head_parameters:int
    max_input_tokens:int
    length_audited:int
    audit:FitAudit


@torch.no_grad()
def evaluate_pair_scores(model:TokenKANSystem, x:TensorStates,y:TensorStates,
                         neg:TensorStates) -> tuple[float,float,float]:
    p1=model.probability(x,y).detach().numpy()
    p0=model.probability(x,neg).detach().numpy()
    label=np.r_[np.ones(len(x)),np.zeros(len(x))]
    p=np.r_[p1,p0]
    return (_auc(p,label),float((p1>p0).mean()),float(((p-label)**2).mean()))


def benchmark_one(law:StudyLaw,seed:int,config:TokenFeedbackConfig,
                  port:TokenStatePort,*,model_id:str,revision:str | None,
                  arms:tuple[StateArm,...]=tuple(StateArm)) -> tuple[Score,...]:
    if not isinstance(law,StudyLaw) or not all(isinstance(x,StateArm) for x in arms):
        raise TypeError("typed law and arms required")
    total=config.bootstrap+config.adaptation+config.calibration
    train=_simulation(law,seed*101+9,total,heldout=False)
    heldout=_simulation(law,seed*101+109,config.heldout,heldout=True)
    ids_train={json.loads(x.x_text)['before']['object']['identity'] for x in train}
    ids_heldout={json.loads(x.x_text)['before']['object']['identity'] for x in heldout}
    overlap=len(ids_train&ids_heldout)
    if overlap:
        raise ValueError("training and heldout share entity identities")
    xx,yy=_batch_for(port,train)
    num_x,num_y=_numeric_batch(train,port.dimension)
    ox,oy=_batch_for(port,heldout)
    negative=_negative_law_outcomes(heldout,law)
    nx,ny=_batch_for(port,negative)
    if not torch.equal(ox.sentence,nx.sentence):
        raise RuntimeError("benchmark evaluator changed counterfactual X")
    phr=_paraphrase(heldout)
    px,py=_batch_for(port,phr)
    paraphrase_negative=_negative_law_outcomes(phr,law)
    pnx,pny=_batch_for(port,paraphrase_negative)
    if not torch.equal(px.sentence,pnx.sentence):
        raise RuntimeError("benchmark evaluator changed paraphrased X")
    dec=_counterfactual_decimal(heldout)
    dx,dy=_batch_for(port,dec)
    if not torch.equal(ox.sentence,dx.sentence):
        raise RuntimeError("benchmark evaluator changed numeric X")
    data_eval=((EvaluationDomain.COUNTERFACTUAL,ox,oy,ny),
               (EvaluationDomain.PARAPHRASE,px,py,pny),
               (EvaluationDomain.SMALL_NUMERIC_DELTA,ox,oy,dy))
    warmx,warmy=xx.take(0,config.bootstrap),yy.take(0,config.bootstrap)
    adapt_end=config.bootstrap+config.adaptation
    trainx,trainy=xx.take(0,adapt_end), yy.take(0,adapt_end)
    calx,caly=xx.take(adapt_end,total), yy.take(adapt_end,total)
    attn_warmup=None
    rows=[]
    numeric_eval=(
        (EvaluationDomain.COUNTERFACTUAL, *_numeric_batch(heldout,port.dimension),
         _numeric_batch(negative,port.dimension)[1]),
        (EvaluationDomain.PARAPHRASE, *_numeric_batch(phr,port.dimension),
         _numeric_batch(paraphrase_negative,port.dimension)[1]),
        (EvaluationDomain.SMALL_NUMERIC_DELTA, *_numeric_batch(heldout,port.dimension),
         _numeric_batch(dec,port.dimension)[1]),
    )
    for arm in arms:
        arm_x,arm_y=(num_x,num_y) if arm is StateArm.NUMERIC_CONTROL else (xx,yy)
        arm_eval=numeric_eval if arm is StateArm.NUMERIC_CONTROL else data_eval
        if arm in (StateArm.ATTENTION_KAN_UPDATE, StateArm.ATTENTION_FEEDBACK):
            if attn_warmup is None:
                attn_warmup=train_warmup(StateArm.TOKEN_ATTENTION,warmx,warmy,config,seed)
            warm=attn_warmup
        else:
            warm=train_warmup(arm,arm_x.take(0,config.bootstrap),arm_y.take(0,config.bootstrap),config,seed)
            if arm is StateArm.TOKEN_ATTENTION:
                attn_warmup=warm
        trained,audit=fit_feedback(warm,
            arm_x.take(0,adapt_end),arm_y.take(0,adapt_end),
            arm_x.take(adapt_end,total),arm_y.take(adapt_end,total),
            config,mode=arm,seed=seed)
        nparameters=sum(p.numel() for p in trained.parameters())
        nheads=sum(p.numel() for p in trained.hypotheses.parameters())
        nattn=sum(p.numel() for v in (trained.x_encoder.token_attention,trained.y_encoder.token_attention)
                  if v is not None for p in v.parameters())
        for domain, x,y,n in arm_eval:
            auroc,paired,brier=evaluate_pair_scores(trained,x,y,n)
            rows.append(Score(law.value,seed,arm,domain,auroc,paired,brier,
                              adapt_end,config.calibration,config.heldout,overlap,
                              model_id,revision,port.dimension,nparameters,nattn,nheads,
                              getattr(port,'max_observed_length',0),
                              getattr(port,'verified_length_count',0), audit))
    return tuple(rows)


def run_benchmark(*, port:TokenStatePort, config:TokenFeedbackConfig,
                  laws:tuple[StudyLaw,...],seeds:tuple[int,...],
                  model_id:str,revision:str | None) -> tuple[Score,...]:
    return tuple(result for law in laws for seed in seeds
                 for result in benchmark_one(law,seed,config,port,
                                             model_id=model_id,revision=revision))


def main() -> None:
    parser=argparse.ArgumentParser(description="Full Qwen token-state versus sentence embedding KAN study")
    parser.add_argument('--model',default=DEFAULT_MODEL)
    parser.add_argument('--revision',default='')
    parser.add_argument('--laws',nargs='+',choices=[x.value for x in StudyLaw],
                        default=[StudyLaw.PRECISION.value,StudyLaw.OPERATOR.value])
    parser.add_argument('--seeds',nargs='+',type=int,default=[0])
    parser.add_argument('--bootstrap',type=int,default=24)
    parser.add_argument('--adaptation',type=int,default=16)
    parser.add_argument('--calibration',type=int,default=12)
    parser.add_argument('--heldout',type=int,default=24)
    parser.add_argument('--warmup-steps',type=int,default=45)
    parser.add_argument('--feedback-steps',type=int,default=24)
    parser.add_argument('--max-seq-length',type=int,default=256)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',default='')
    args=parser.parse_args()
    config=TokenFeedbackConfig(bootstrap=args.bootstrap, adaptation=args.adaptation,
                               calibration=args.calibration,heldout=args.heldout,
                               warmup_steps=args.warmup_steps,feedback_steps=args.feedback_steps)
    revision=args.revision or (DEFAULT_QWEN_REVISION if args.model==DEFAULT_MODEL else None)
    port=SentenceTransformerTokenPort(args.model,revision=revision,
                                       device=args.device,max_seq_length=args.max_seq_length,
                                       batch_size=args.batch_size)
    result=run_benchmark(port=port,config=config,laws=tuple(StudyLaw(x) for x in args.laws),
                         seeds=tuple(args.seeds),model_id=args.model,revision=revision)
    data={'model':args.model,'revision':revision,'config':asdict(config),
          'max_seen_length':port.max_observed_length,
          'audited_texts':port.verified_length_count,'backbone_forward_batches':port.forward_calls,
          'scores':[asdict(x) for x in result],
          'limitations':[
             'Synthetic observations, counterfactual heldout evaluation only.',
             'Qwen frozen. Feedback changes only lightweight pooling and projections.',
             'Pair discriminator sigmoid is NOT calibrated p(Y|do(X)).',
             'Hypothesis weights from calibration are NOT Bayesian physical-law posterior.',
             '64/1024-vs-full arms differ in trainable projection size; report counts.',
             'Numeric control uses the existing tanh-compressed generic numeric hash; not exact numerical measurements.',
             'No symbolic law extraction or agent-controlled intervention.',
          ]}
    payload=json.dumps(data,indent=2,ensure_ascii=False)
    if args.output:
        target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(payload+'\n',encoding='utf8')
    print(json.dumps({'total_scores':len(result),
                      'max_tokens':port.max_observed_length,
                      'audited_texts':port.verified_length_count,
                      'mean_auroc':{
                          arm.value:float(np.mean([r.auroc for r in result
                                  if r.arm is arm and r.domain is EvaluationDomain.COUNTERFACTUAL]))
                          for arm in StateArm},
                      },indent=2))


if __name__=='__main__':
    main()
