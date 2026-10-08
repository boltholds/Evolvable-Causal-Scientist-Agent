"""Frozen Qwen contextual-token representations with KAN-only learning.

The pretrained model and all X/Y pooling, attention and projection parameters
remain fixed. Only the population of spline-KAN hypotheses is trainable.
No KAN-to-encoder feedback or encoder gradient updates are permitted.
Train/calibration and evaluation data remain disjoint.
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
    GATED_ATTENTION = "gated_attention"
    GATED_KAN_UPDATE = "gated_kan_update"


# Active arms exclude historical feedback variants (see Git history).
DEFAULT_ARMS = (
    StateArm.SENTENCE_64, StateArm.NUMERIC_CONTROL,
    StateArm.SENTENCE_FULL, StateArm.TOKEN_MEAN,
    StateArm.TOKEN_ATTENTION, StateArm.ATTENTION_KAN_UPDATE,
)

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
class TokenRelationConfig:
    bootstrap: int = 24
    adaptation: int = 16
    calibration: int = 12
    heldout: int = 24
    warmup_steps: int = 45
    extra_kan_steps: int = 24
    checkpoint_every: int = 8
    warmup_lr: float = 0.012
    kan_lr: float = 0.005
    min_spread: float = 0.006
    tolerance: float = 0.06

    def __post_init__(self) -> None:
        for key in ("bootstrap", "adaptation", "calibration", "heldout", "warmup_steps",
                    "extra_kan_steps", "checkpoint_every"):
            val = getattr(self,key)
            if type(val) is not int or val < 2:
                raise ValueError(f"{key} needs >= 2")
        if self.checkpoint_every > self.extra_kan_steps:
            raise ValueError("KAN update needs at least one checkpoint")
        for key in ("warmup_lr", "kan_lr", "min_spread", "tolerance"):
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
                StateArm.TOKEN_ATTENTION, StateArm.ATTENTION_KAN_UPDATE
            ) else None
        )
        self.residual_attention_gate = None
        input_width = 64 if mode in (StateArm.SENTENCE_64, StateArm.NUMERIC_CONTROL) else width
        # Observations can occupy a tiny region of Qwen's 1024-dimensional
        # sphere. Per-sample layer normalization does not remove the shared
        # direction and previously made all learned latents identical.
        # These statistics are fitted on warmup observations ONLY.
        self.register_buffer('warmup_center', torch.zeros(input_width))
        self.register_buffer('warmup_scale', torch.ones(input_width))
        self.projection = nn.Sequential(
            nn.Linear(input_width, 24), nn.Tanh(),
            nn.Linear(24, 2), nn.Tanh(),
        )
        if mode in (
            StateArm.GATED_ATTENTION, StateArm.GATED_KAN_UPDATE,
        ):
            # Preserve the exact initial projection/head RNG state of the
            # token_mean baseline. A separately initialized attention branch
            # must not accidentally change the KAN control initialization.
            rng_state = torch.random.get_rng_state()
            self.token_attention = nn.Linear(width, 1, bias=False)
            torch.random.set_rng_state(rng_state)
            # Start almost exactly at masked mean; bounded learned residual
            # can refine it but never replace the whole representation.
            self.residual_attention_gate = nn.Parameter(torch.tensor(-5.0))

        # Encoders are frozen even during initial KAN warmup.
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def weights(self, states: TensorStates) -> Tensor:
        if self.token_attention is None:
            return states.mask.float() / states.mask.float().sum(1,keepdim=True)
        # Hard-bounded token scores prevent a single common special token
        # from monopolizing the entire attention mass during warmup.
        raw = self.token_attention(F.layer_norm(states.tokens,(self.width,))).squeeze(-1)
        logits = 1.5 * torch.tanh(raw / 1.5)
        logits = logits.masked_fill(~states.mask, torch.finfo(logits.dtype).min)
        return torch.softmax(logits,dim=1)

    def _pooled(self, states: TensorStates) -> Tensor:
        if self.mode in (StateArm.SENTENCE_64, StateArm.NUMERIC_CONTROL):
            return F.normalize(states.sentence[:,:64],dim=1)
        if self.mode is StateArm.SENTENCE_FULL:
            return states.sentence
        # The context-preserving masked-mean skip path is immutable: attention
        # can focus on predictive tokens but cannot destroy the shared state.
        mean = states.mask.float() / states.mask.float().sum(1, keepdim=True)
        pooled_mean = torch.einsum("bt,btd->bd", mean, states.tokens)
        if self.token_attention is None:
            return F.normalize(pooled_mean,dim=1)
        attention = self.weights(states)
        attended = torch.einsum("bt,btd->bd",attention,states.tokens)
        if self.residual_attention_gate is None:
            vector = 0.65 * pooled_mean + 0.35 * attended
        else:
            gate = torch.sigmoid(self.residual_attention_gate) * 0.5
            vector = pooled_mean + gate * (attended - pooled_mean)
        return F.normalize(vector,dim=1)

    @torch.no_grad()
    def fit_warmup_normalizer(self, states: TensorStates) -> None:
        if len(states) < 3:
            raise ValueError('warmup normalization requires >=3 observations')
        vec = self._pooled(states)
        center = vec.mean(0)
        stdev = vec.std(0, unbiased=False)
        if not torch.isfinite(vec).all():
            raise ValueError('nonfinite warmup representations')
        # The floor protects near-constant coordinates from exploding, while
        # the per-coordinate scale makes tiny numeric distinctions accessible.
        self.warmup_center.copy_(center)
        self.warmup_scale.copy_(stdev.clamp_min(0.001))

    def forward(self, states: TensorStates) -> Tensor:
        vector = (self._pooled(states) - self.warmup_center) / self.warmup_scale
        vector = vector.clamp(-8.,8.)
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




@dataclass(frozen=True)
class FitAudit:
    warmup_steps: int
    extra_kan_steps: int
    head_updates: int
    accepted_checkpoints: int
    rejected_checkpoints: int
    initial_calibration_loss: float
    final_calibration_loss: float
    initial_spread: float
    final_spread: float
    hypothesis_version: int
    skipped_due_to_low_spread: bool = False

def train_warmup(mode:StateArm,x:TensorStates,y:TensorStates,config:TokenRelationConfig,
                 seed:int) -> TokenKANSystem:
    torch.set_num_threads(1)
    torch.manual_seed(seed*1000+59)
    model=TokenKANSystem(mode,x.tokens.shape[-1])
    model.x_encoder.fit_warmup_normalizer(x)
    model.y_encoder.fit_warmup_normalizer(y)
    original_state = copy.deepcopy(model.state_dict())
    _allow_grad(model,heads=True)
    optimizer=torch.optim.AdamW(model.hypotheses.parameters(),lr=config.warmup_lr)
    rng=np.random.default_rng(seed*1000+65)
    # Train-only constrained checkpoint selection prevents a catastrophic
    # end-of-warmup collapse from replacing an earlier informative state.
    with torch.no_grad():
        initial_spread = model.min_spread(x,y)
        initial_score = float(_losses(model,x,y,1).mean())
    required_spread = (1e-6 if mode is StateArm.NUMERIC_CONTROL
                       else config.min_spread)
    best = copy.deepcopy(model.state_dict()) if initial_spread >= required_spread else None
    best_score = initial_score if best is not None else float('inf')
    for step in range(config.warmup_steps):
        shift=int(rng.integers(1,len(x)))
        losses=_losses(model,x,y,shift)
        objective=losses.mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%5==0 or step+1==config.warmup_steps:
            with torch.no_grad():
                spread=model.min_spread(x,y)
                score=float(_losses(model,x,y,1).mean())
            if (np.isfinite(score) and spread>=required_spread
                and score < best_score):
                best_score=score
                best=copy.deepcopy(model.state_dict())
    # An underidentified observation set is a research outcome, not a test
    # framework error. Keep the initial hypothesis and explicitly flag it;
    # further KAN-only updates are skipped when variation is insufficient.
    model.warmup_adequate_spread = best is not None
    model.load_state_dict(original_state if best is None else best)
    _allow_grad(model,heads=False)
    model.eval()
    return model


def _allow_grad(model: TokenKANSystem, *, heads: bool) -> None:
    """Freeze both encoders unconditionally; enable only KAN-head gradients."""
    for param in model.hypotheses.parameters():
        param.requires_grad_(heads)
        param.grad = None
    for param in (*model.x_encoder.parameters(), *model.y_encoder.parameters()):
        param.requires_grad_(False)
        param.grad = None


def fit_kan_only(
    warmup: TokenKANSystem, x: TensorStates, y: TensorStates,
    cx: TensorStates, cy: TensorStates, config: TokenRelationConfig,
    *, mode: StateArm, seed: int,
) -> tuple[TokenKANSystem, FitAudit]:
    """Fit spline KAN heads only; encoder representations never change."""
    model = copy.deepcopy(warmup)
    _refresh_weights(model,cx,cy)
    before = _calibration_loss(model,cx,cy)
    before_spread = model.min_spread(cx,cy)
    head_only = mode in (StateArm.ATTENTION_KAN_UPDATE, StateArm.GATED_KAN_UPDATE)
    if not getattr(warmup,'warmup_adequate_spread',True):
        _allow_grad(model,heads=False)
        model.eval()
        return model,FitAudit(config.warmup_steps,0,0,0,0,
            before,before,before_spread,before_spread,0,True)
    if not head_only:
        _allow_grad(model,heads=False)
        model.eval()
        return model,FitAudit(config.warmup_steps,0,0,0,0,
            before,before,before_spread,before_spread,0)
    if warmup.x_encoder.mode not in (StateArm.TOKEN_ATTENTION, StateArm.GATED_ATTENTION):
        raise ValueError("KAN-only update requires attention-pooling warmup")
    _allow_grad(model,heads=True)
    params = tuple(model.hypotheses.parameters())
    rng = np.random.default_rng(seed*1000+103)
    accepted=rejected=0
    for start in range(0,config.extra_kan_steps,config.checkpoint_every):
        steps=min(config.checkpoint_every,config.extra_kan_steps-start)
        snapshot=copy.deepcopy(model.state_dict())
        base_nll=_calibration_loss(model,cx,cy)
        optimizer=torch.optim.AdamW(params,lr=config.kan_lr)
        for _ in range(steps):
            loss=(_losses(model,x,y,int(rng.integers(1,len(x))))*model.weights.detach()).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params,1.)
            optimizer.step()
        _refresh_weights(model,cx,cy)
        candidate_nll=_calibration_loss(model,cx,cy)
        if np.isfinite(candidate_nll) and candidate_nll<=base_nll+config.tolerance:
            accepted+=1
            model.hypothesis_version+=1
        else:
            rejected+=1
            model.load_state_dict(snapshot)
    _allow_grad(model,heads=False)
    model.eval()
    return model, FitAudit(
        config.warmup_steps,config.extra_kan_steps,config.extra_kan_steps,
        accepted,rejected,before,_calibration_loss(model,cx,cy),
        before_spread,model.min_spread(cx,cy),model.hypothesis_version,
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
    warmup_adequate_spread: bool


@torch.no_grad()
def evaluate_pair_scores(model:TokenKANSystem, x:TensorStates,y:TensorStates,
                         neg:TensorStates) -> tuple[float,float,float]:
    p1=model.probability(x,y).detach().numpy()
    p0=model.probability(x,neg).detach().numpy()
    label=np.r_[np.ones(len(x)),np.zeros(len(x))]
    p=np.r_[p1,p0]
    return (_auc(p,label),float((p1>p0).mean()),float(((p-label)**2).mean()))


def benchmark_one(law:StudyLaw,seed:int,config:TokenRelationConfig,
                  port:TokenStatePort,*,model_id:str,revision:str | None,
                   arms:tuple[StateArm,...]=DEFAULT_ARMS) -> tuple[Score,...]:
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
    gated_warmup=None
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
        if arm is StateArm.ATTENTION_KAN_UPDATE:
            if attn_warmup is None:
                attn_warmup=train_warmup(StateArm.TOKEN_ATTENTION,warmx,warmy,config,seed)
            warm=attn_warmup
        elif arm is StateArm.GATED_KAN_UPDATE:
            if gated_warmup is None:
                gated_warmup=train_warmup(StateArm.GATED_ATTENTION,warmx,warmy,config,seed)
            warm=gated_warmup
        else:
            warm=train_warmup(arm,arm_x.take(0,config.bootstrap),arm_y.take(0,config.bootstrap),config,seed)
            if arm is StateArm.TOKEN_ATTENTION:
                attn_warmup=warm
            if arm is StateArm.GATED_ATTENTION:
                gated_warmup=warm
        trained,audit=fit_kan_only(warm,
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
                              getattr(port,'verified_length_count',0), audit,
                              bool(getattr(warm,'warmup_adequate_spread',True))))
    return tuple(rows)


def run_benchmark(*, port:TokenStatePort, config:TokenRelationConfig,
                  laws:tuple[StudyLaw,...],seeds:tuple[int,...],
                  model_id:str,revision:str | None,
                  arms:tuple[StateArm,...]=DEFAULT_ARMS) -> tuple[Score,...]:
    return tuple(result for law in laws for seed in seeds
                 for result in benchmark_one(law,seed,config,port,
                                             model_id=model_id,revision=revision,arms=arms))


def main() -> None:
    parser=argparse.ArgumentParser(description="Frozen token-state representations and KAN-only study")
    parser.add_argument('--model',default=DEFAULT_MODEL)
    parser.add_argument('--revision',default='')
    parser.add_argument('--laws',nargs='+',choices=[x.value for x in StudyLaw],
                        default=[StudyLaw.PRECISION.value,StudyLaw.OPERATOR.value])
    parser.add_argument('--arms',nargs='+',choices=[x.value for x in StateArm],
                        default=[x.value for x in DEFAULT_ARMS])
    parser.add_argument('--seeds',nargs='+',type=int,default=[0])
    parser.add_argument('--bootstrap',type=int,default=24)
    parser.add_argument('--adaptation',type=int,default=16)
    parser.add_argument('--calibration',type=int,default=12)
    parser.add_argument('--heldout',type=int,default=24)
    parser.add_argument('--warmup-steps',type=int,default=45)
    parser.add_argument('--extra-kan-steps',type=int,default=24)
    parser.add_argument('--max-seq-length',type=int,default=256)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',default='')
    args=parser.parse_args()
    config=TokenRelationConfig(bootstrap=args.bootstrap, adaptation=args.adaptation,
                               calibration=args.calibration,heldout=args.heldout,
                               warmup_steps=args.warmup_steps,extra_kan_steps=args.extra_kan_steps)
    revision=args.revision or (DEFAULT_QWEN_REVISION if args.model==DEFAULT_MODEL else None)
    port=SentenceTransformerTokenPort(args.model,revision=revision,
                                       device=args.device,max_seq_length=args.max_seq_length,
                                       batch_size=args.batch_size)
    arms=tuple(StateArm(a) for a in args.arms)
    if len(set(arms))!=len(arms):
        raise ValueError('repeated experiment arms are not permitted')
    result=run_benchmark(port=port,config=config,laws=tuple(StudyLaw(x) for x in args.laws),
                         seeds=tuple(args.seeds),model_id=args.model,revision=revision,
                         arms=arms)
    data={'model':args.model,'revision':revision,'config':asdict(config),
          'max_seen_length':port.max_observed_length,
          'audited_texts':port.verified_length_count,'backbone_forward_batches':port.forward_calls,
          'scores':[asdict(x) for x in result],
          'limitations':[
             'Synthetic observations, counterfactual heldout evaluation only.',
             'Qwen and all X/Y encoders are frozen; only spline-KAN hypotheses train.',
             'Pair discriminator sigmoid is NOT calibrated p(Y|do(X)).',
             'Hypothesis weights from calibration are NOT Bayesian physical-law posterior.',
             'Full and token states are standardized with warmup-only mean and per-feature variation.',
             '64/1024-vs-full arms differ in trainable projection size; report counts.',
             'Numeric control uses the existing tanh-compressed generic numeric hash; not exact numerical measurements.',
             'No symbolic law extraction or agent-controlled intervention.',
             'Insufficient warmup variation is flagged; further KAN updates are skipped.',
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
                          for arm in arms},
                      },indent=2))


if __name__=='__main__':
    main()
