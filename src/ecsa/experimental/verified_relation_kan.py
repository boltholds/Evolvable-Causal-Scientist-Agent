"""Train-only PCA/random-state projections and repeat-supported X/Y contrasts.

Separate X and Y encodings remain frozen. Only spline-KAN hypotheses learn.
Negatives require a *declared deterministic replay contract*, multiple
identical observed outcomes for the same pre-action X, and a second witnessed
intervention with a different outcome under the same schema/world epoch.

Repeat observations give evidence under that contract, not a causal proof
in stochastic, partially observed or changing environments. Without the
contract the miner abstains, rather than marking random Y permutations false.
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

from .relation_discovery import SplineEdgeLayer
from .text_first_kan import _auc
from .verbalization_study import StudyLaw, _simulation, _law_response
from .world_model_encoders import Candidate, CausalLMConfig, FrozenCausalLMTokenPort


class PoolKind(StrEnum):
    SENTENCE_FULL = "sentence_full"
    TOKEN_MEAN = "token_mean"


class ProjectionKind(StrEnum):
    RANDOM = "random"
    PCA = "train_only_pca"


class NegativeTraining(StrEnum):
    WITNESSED = "witnessed"
    SHUFFLED_Y_UNSAFE_CONTROL = "shuffled_y_unsafe_control"


@dataclass(frozen=True)
class ReplayWitness:
    """Independent environment replays with exactly the same observed X/action.

    'deterministic_contract' must come from an explicit environment capability,
    not from inferring determinism from a couple of samples.
    """
    experiment_id: str
    world_epoch: str
    schema_id: str
    x_text: str
    observed_y_replays: tuple[str, ...]
    deterministic_contract: bool = False

    def __post_init__(self) -> None:
        if not all((self.experiment_id, self.world_epoch, self.schema_id, self.x_text)):
            raise ValueError("witness identity, schema and X are required")
        if not self.observed_y_replays or any(not isinstance(y, str) or not y
                                               for y in self.observed_y_replays):
            raise ValueError("nonempty observed outcomes required")
        if type(self.deterministic_contract) is not bool:
            raise TypeError("deterministic_contract must be an explicit bool")


@dataclass(frozen=True)
class WitnessedPair:
    x_text: str
    y_true: str
    y_alternative: str
    positive_experiment_id: str
    alternative_experiment_id: str
    schema_id: str
    world_epoch: str

    def __post_init__(self) -> None:
        if self.y_true == self.y_alternative:
            raise ValueError("identical valid outcomes are not negative pairs")
        if self.positive_experiment_id == self.alternative_experiment_id:
            raise ValueError("negative must come from a different intervention")


@dataclass(frozen=True)
class MiningAudit:
    total_witnesses: int
    eligible_deterministic_witnesses: int
    unstable_or_unsupported_witnesses: int
    no_disjoint_alternative: int
    validated_pairs: int
    shuffled_false_negative_fraction: float | None


@dataclass(frozen=True)
class MinedPairs:
    samples: tuple[WitnessedPair, ...]
    audit: MiningAudit


def mine_repeat_supported_pairs(
    witnesses: tuple[ReplayWitness, ...], *,
    seed: int, minimum_replays: int = 3,
) -> MinedPairs:
    """No hidden world-law access, no generated Y and no random permutation labels.

    Assumptions: full relevant pre-action state, same schema/world epoch and
    an environment-declared deterministic replay capability. If those cannot
    be established (common in stochastic worlds), this returns no negatives.
    """
    if type(minimum_replays) is not int or minimum_replays < 2:
        raise ValueError("at least two independent replays required")
    if not witnesses:
        return MinedPairs((),MiningAudit(0,0,0,0,0,None))
    ids = [w.experiment_id for w in witnesses]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate intervention experiment ID")
    keys = [(w.world_epoch,w.schema_id,w.x_text) for w in witnesses]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate X/action replay group; combine its trials")
    rng = np.random.default_rng(seed)
    eligible = tuple(w for w in witnesses
                     if w.deterministic_contract
                     and len(w.observed_y_replays)>=minimum_replays
                     and len(set(w.observed_y_replays))==1)
    outcomes = tuple(w.observed_y_replays[0] for w in eligible)
    # Diagnostic only. This quantifies the exact-label contamination caused
    # by arbitrary Y permutation in these specific deterministic fixtures.
    shuffled_rate: float | None = None
    if len(eligible)>1:
        shifted = np.roll(np.asarray(outcomes,dtype=object), 1)
        shuffled_rate=float(np.mean(np.asarray(outcomes,dtype=object)==shifted))
    sampled: list[WitnessedPair] = []
    missing = 0
    for pos in eligible:
        choices = [
            other for other in eligible
            if other.experiment_id != pos.experiment_id
            and other.world_epoch == pos.world_epoch
            and other.schema_id == pos.schema_id
            and other.observed_y_replays[0] != pos.observed_y_replays[0]
        ]
        if not choices:
            missing += 1
            continue
        # Balanced, deterministic seed; never inspect any law, feature or Y magnitude.
        other = choices[int(rng.integers(len(choices)))]
        sampled.append(WitnessedPair(
            pos.x_text,pos.observed_y_replays[0],other.observed_y_replays[0],
            pos.experiment_id,other.experiment_id,pos.schema_id,pos.world_epoch,
        ))
    return MinedPairs(tuple(sampled),MiningAudit(
        len(witnesses),len(eligible),len(witnesses)-len(eligible),
        missing,len(sampled),shuffled_rate,
    ))


@runtime_checkable
class ContextStatePort(Protocol):
    @property
    def dimension(self) -> int: ...
    def encode_many(self, texts: tuple[str, ...]) -> object: ...


def extract_frozen_vectors(port: ContextStatePort, texts: tuple[str, ...],
                           kind: PoolKind) -> np.ndarray:
    if not isinstance(kind, PoolKind):
        raise TypeError("typed pool method required")
    if not isinstance(port, ContextStatePort):
        raise TypeError("context-token port required")
    states=port.encode_many(texts)
    if states.tokens.ndim!=3 or states.mask.shape!=states.tokens.shape[:2]:
        raise ValueError("invalid full hidden states or token mask")
    if states.tokens.shape[-1]!=port.dimension:
        raise ValueError("pretrained hidden state width mismatch")
    if kind is PoolKind.SENTENCE_FULL:
        vectors=np.asarray(states.sentence,dtype=np.float32)
    else:
        mask=np.asarray(states.mask,dtype=np.float32)
        counts=mask.sum(axis=1,keepdims=True)
        if (counts<=0).any():
            raise ValueError("empty mask in observed sequence")
        vectors=np.einsum("btd,bt->bd",states.tokens,mask)/counts
        norms=np.linalg.norm(vectors,axis=1,keepdims=True)
        vectors=vectors/np.maximum(norms,1e-12)
    if vectors.shape!=(len(texts),port.dimension) or not np.isfinite(vectors).all():
        raise ValueError("invalid nonfinite frozen state vectors")
    return vectors.astype(np.float32)


@dataclass(frozen=True)
class ProjectionState:
    kind: ProjectionKind
    dimension: int
    fitted_sample_count: int
    x_center: np.ndarray
    y_center: np.ndarray
    x_basis: np.ndarray
    y_basis: np.ndarray
    x_shift: np.ndarray
    y_shift: np.ndarray
    x_scale: np.ndarray
    y_scale: np.ndarray

    def apply(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        xx=_project_side(x,self.x_center,self.x_basis,self.x_shift,self.x_scale)
        yy=_project_side(y,self.y_center,self.y_basis,self.y_shift,self.y_scale)
        return xx,yy


def _project_side(vectors: np.ndarray, center: np.ndarray, basis: np.ndarray,
                  shift: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if vectors.ndim!=2 or vectors.shape[1]!=len(center):
        raise ValueError("unexpected encoder width")
    projected=(vectors-center)@basis
    return np.clip((projected-shift)/scale,-8.0,8.0).astype(np.float32)


def _fit_side(matrix: np.ndarray, kind: ProjectionKind, width: int,
              seed: int) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    if matrix.ndim!=2 or len(matrix)<3 or not np.isfinite(matrix).all():
        raise ValueError("nonempty finite train-only embedding matrix required")
    n,feature_width=matrix.shape
    if width>feature_width:
        raise ValueError("latent dimension exceeds frozen feature dimension")
    if kind is ProjectionKind.PCA and width>=n:
        raise ValueError("PCA width must be less than the training sample count")
    center=matrix.mean(axis=0).astype(np.float32)
    demeaned=np.asarray(matrix-center,dtype=np.float64)
    if kind is ProjectionKind.PCA:
        # Thin SVD: fitting moments and directions ONLY on train observations.
        _,singular,vh=np.linalg.svd(demeaned,full_matrices=False)
        rank=min(width,int(np.count_nonzero(singular>1e-10)),len(vh))
        basis=np.zeros((feature_width,width),dtype=np.float32)
        if rank:
            basis[:,:rank]=vh[:rank].T
    elif kind is ProjectionKind.RANDOM:
        rng=np.random.default_rng(seed)
        probe=rng.normal(size=(feature_width,width))
        basis=np.linalg.qr(probe,mode="reduced")[0].astype(np.float32)
    else:
        raise TypeError("typed projection strategy required")
    transformed=(matrix-center)@basis
    shift=transformed.mean(axis=0).astype(np.float32)
    scale=np.maximum(transformed.std(axis=0),0.001).astype(np.float32)
    return center,basis,shift,scale


def fit_train_only_projection(x_train: np.ndarray,y_train: np.ndarray,
                              *, kind: ProjectionKind,width: int,
                              seed: int) -> ProjectionState:
    if type(width) is not int or width<2:
        raise ValueError("latent dimension must be >=2")
    if x_train.shape!=y_train.shape or x_train.ndim!=2:
        raise ValueError("X/Y train matrices must have identical shape")
    argsx=_fit_side(x_train,kind,width,seed*17+11)
    argsy=_fit_side(y_train,kind,width,seed*17+19)
    return ProjectionState(kind,width,len(x_train),
                           argsx[0],argsy[0],argsx[1],argsy[1],
                           argsx[2],argsy[2],argsx[3],argsy[3])


class WideSplineKAN(nn.Module):
    """Same implicit relation principle, input width is 2 * latent width."""
    def __init__(self, latent_width: int, hidden: int, knots: int):
        super().__init__()
        self.first=SplineEdgeLayer(2*latent_width,hidden,knots=knots)
        self.last=SplineEdgeLayer(hidden,1,knots=knots)

    def forward(self,joint:Tensor)->Tensor:
        return self.last(torch.tanh(self.first(joint))).flatten()


class RelationPopulation(nn.Module):
    def __init__(self,latent_width:int):
        super().__init__()
        self.hypotheses=nn.ModuleList((
            WideSplineKAN(latent_width,12,5),
            WideSplineKAN(latent_width,9,7),
            WideSplineKAN(latent_width,7,9),
        ))

    def logits(self,x:Tensor,y:Tensor)->Tensor:
        joint=torch.cat((x,y),dim=1)
        return torch.stack([head(joint) for head in self.hypotheses])

    @torch.no_grad()
    def probability(self,x:Tensor,y:Tensor)->Tensor:
        return torch.sigmoid(self.logits(x,y)).mean(dim=0)


@dataclass(frozen=True)
class StudyConfig:
    train_count:int=48
    calibration_count:int=16
    heldout_count:int=24
    replays_per_x:int=3
    warmup_steps:int=100
    lr:float=0.009
    checkpoint_every:int=10

    def __post_init__(self)->None:
        for field in ("train_count","calibration_count","heldout_count",
                      "replays_per_x","warmup_steps","checkpoint_every"):
            value=getattr(self,field)
            floor=(3 if field=="replays_per_x" else
                   1 if field=="checkpoint_every" else 4)
            if type(value) is not int or value<floor:
                raise ValueError(f"{field} must be an integer >=4 (replays >=3)")
        if not 0.0<self.lr<0.1:
            raise ValueError("invalid optimizer lr")


def _vectorize(port:ContextStatePort,samples:tuple[WitnessedPair,...],
               kind:PoolKind) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if not samples:
        raise ValueError("at least one replay-supported pair required")
    xs=tuple(x.x_text for x in samples)
    ps=tuple(x.y_true for x in samples)
    ns=tuple(x.y_alternative for x in samples)
    # Shared backing-model cache deduplicates repeated strings; the learner
    # sees no ground-truth mechanism labels, only witnessed observations.
    all_vec=extract_frozen_vectors(port,xs+ps+ns,kind)
    n=len(samples)
    return all_vec[:n],all_vec[n:2*n],all_vec[2*n:]


def _project_pairs(proj:ProjectionState,
                   data:tuple[np.ndarray,np.ndarray,np.ndarray],
                   ) -> tuple[Tensor,Tensor,Tensor]:
    x,positive,negative=data
    a,b=proj.apply(x,positive)
    _,c=proj.apply(x,negative)
    return torch.from_numpy(a),torch.from_numpy(b),torch.from_numpy(c)


def _contrast_loss(model: RelationPopulation, rows:tuple[Tensor,Tensor,Tensor])->Tensor:
    x,p,n=rows
    pos=model.logits(x,p)
    neg=model.logits(x,n)
    return (F.softplus(-pos).mean()+F.softplus(neg).mean())/2


@torch.inference_mode()
def _metrics(model:RelationPopulation, data:tuple[Tensor,Tensor,Tensor]
             ) -> tuple[float,float,float,float]:
    x,p,n=data
    true=model.probability(x,p).cpu().numpy()
    wrong=model.probability(x,n).cpu().numpy()
    labels=np.r_[np.ones(len(x)),np.zeros(len(x))]
    scores=np.r_[true,wrong]
    return (_auc(scores,labels),float(np.mean(true>wrong)),
            float(((scores-labels)**2).mean()),
            float(np.mean(-np.log(np.clip(true,1e-7,1))-
                          np.log(np.clip(1-wrong,1e-7,1)))/2))


def fit_kan_with_witnesses(
    train:tuple[Tensor,Tensor,Tensor],
    calibration:tuple[Tensor,Tensor,Tensor], *,
    dimension:int,seed:int,config:StudyConfig,
) -> tuple[RelationPopulation, float]:
    torch.set_num_threads(1)
    torch.manual_seed(seed*1000+89)
    model=RelationPopulation(dimension)
    optimizer=torch.optim.AdamW(model.hypotheses.parameters(),lr=config.lr)
    best_loss=float("inf")
    best=copy.deepcopy(model.state_dict())
    for step in range(config.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        loss=_contrast_loss(model,train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.hypotheses.parameters(),1.0)
        optimizer.step()
        if (step+1)%config.checkpoint_every==0 or step+1==config.warmup_steps:
            with torch.no_grad():
                validation=float(_contrast_loss(model,calibration))
            if np.isfinite(validation) and validation<best_loss:
                best_loss=validation
                best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    model.eval()
    return model,best_loss


def fixture_replay_witnesses(
    law:StudyLaw,seed:int,count:int,*,heldout:bool,repetitions:int,
) -> tuple[ReplayWitness,...]:
    """BENCHMARK WORLD ONLY: repeated actual deterministic fixture steps.

    Each repetition independently invokes the fixture transition function.
    _law_response is not referenced by any learner/miner component.
    """
    if repetitions<2:
        raise ValueError("at least two environment replays required")
    originals=_simulation(law,seed,count,heldout=heldout)
    result=[]
    for sample in originals:
        observation=json.loads(sample.x_text)["before"]
        u=float(observation["readings"]["primary"])
        v=float(observation["readings"]["reference"])
        op=observation["predicate"]["operator"]
        outcomes=[]
        for _ in range(repetitions):
            # This is the simulated environment's controlled step, never
            # a compatibility label produced inside the KAN learner.
            measured=_law_response(law,u,v,op)
            observed=json.loads(sample.y_text)
            observed["after"]["measurement"]=float(f"{measured:.5f}")
            observed["after"]["observation"]="recorded"
            outcomes.append(json.dumps(observed,sort_keys=True,
                                       ensure_ascii=False,separators=(",",":")))
        result.append(ReplayWitness(
            sample.transition_id,
            "deterministic-synthetic-world-v1",
            sample.schema_id,
            sample.x_text,
            tuple(outcomes),
            deterministic_contract=True,
        ))
    return tuple(result)


@dataclass(frozen=True)
class StudyRow:
    law:str
    seed:int
    pool:PoolKind
    projection:ProjectionKind
    negative_training:NegativeTraining
    dimension:int
    train_pairs:int
    calibration_pairs:int
    heldout_pairs:int
    train_shuffled_false_negative_fraction:float | None
    train_no_disjoint_alternative:int
    heldout_auroc:float
    heldout_paired_accuracy:float
    heldout_brier:float
    heldout_contrastive_nll:float
    calibration_loss:float
    kan_parameters:int
    frozen_encoder_width:int
    identity_overlap:int


def benchmark_one(
    *, port:ContextStatePort,law:StudyLaw,seed:int,config:StudyConfig,
    pools:tuple[PoolKind,...]=(PoolKind.SENTENCE_FULL,),
    projections:tuple[ProjectionKind,...]=(ProjectionKind.RANDOM,ProjectionKind.PCA),
    dimensions:tuple[int,...]=(2,8,16,32),
    pairing:tuple[NegativeTraining,...]=(NegativeTraining.WITNESSED,),
) -> tuple[StudyRow,...]:
    if not pools or not projections or not dimensions or not pairing:
        raise ValueError("nonempty ablation axes required")
    if not all(isinstance(x,PoolKind) for x in pools):
        raise TypeError("PoolKind required")
    if not all(isinstance(x,ProjectionKind) for x in projections):
        raise TypeError("ProjectionKind required")
    if len(set(dimensions))!=len(dimensions):
        raise ValueError("duplicate latent dimensions")
    if len(set(pairing))!=len(pairing) or not all(isinstance(x,NegativeTraining) for x in pairing):
        raise ValueError("unique typed negative-training methods required")
    # Train, calibration and heldout are obtained by independently seeded
    # simulated environments; never reuse a pooled cross-split PCA basis.
    train_w=fixture_replay_witnesses(
        law,seed*101+9,config.train_count,heldout=False,
        repetitions=config.replays_per_x)
    cal_w=fixture_replay_witnesses(
        law,seed*101+59,config.calibration_count,heldout=False,
        repetitions=config.replays_per_x)
    test_w=fixture_replay_witnesses(
        law,seed*101+109,config.heldout_count,heldout=True,
        repetitions=config.replays_per_x)
    train=mine_repeat_supported_pairs(train_w,seed=seed*197+11,
                                      minimum_replays=config.replays_per_x)
    calibration=mine_repeat_supported_pairs(cal_w,seed=seed*197+19,
                                            minimum_replays=config.replays_per_x)
    heldout=mine_repeat_supported_pairs(test_w,seed=seed*197+29,
                                        minimum_replays=config.replays_per_x)
    if not (train.samples and calibration.samples and heldout.samples):
        raise RuntimeError(
            "insufficient witnessed incompatible outcomes; no negative labels invented")
    ids=[{w.experiment_id for w in part}
         for part in (train_w,cal_w,test_w)]
    overlap=sum(len(ids[i]&ids[j]) for i,j in ((0,1),(0,2),(1,2)))
    if overlap:
        raise ValueError("train/calibration/test intervention IDs overlap")
    rows=[]
    for pool in pools:
        train_vec=_vectorize(port,train.samples,pool)
        cal_vec=_vectorize(port,calibration.samples,pool)
        held_vec=_vectorize(port,heldout.samples,pool)
        for proj_type in projections:
            for latent in dimensions:
                if latent>port.dimension:
                    raise ValueError("latent dimension exceeds pretrained width")
                fitted=fit_train_only_projection(
                    train_vec[0],train_vec[1],kind=proj_type,width=latent,
                    seed=seed*37+int(latent),
                )
                witnessed_train=_project_pairs(fitted,train_vec)
                ca=_project_pairs(fitted,cal_vec)
                he=_project_pairs(fitted,held_vec)
                for label_scheme in pairing:
                    if label_scheme is NegativeTraining.SHUFFLED_Y_UNSAFE_CONTROL:
                        # Historical control ONLY: labels include known false negatives
                        # whenever adjacent observed Ys happen to be equal.
                        xx, positive, _=witnessed_train
                        training=(xx,positive,torch.roll(positive,1,dims=0))
                    else:
                        training=witnessed_train
                    trained,best=fit_kan_with_witnesses(
                        training,ca,dimension=latent,seed=seed,config=config)
                    auc,paired,brier,nll=_metrics(trained,he)
                    rows.append(StudyRow(
                        law.value,seed,pool,proj_type,label_scheme,latent,
                        len(train.samples),len(calibration.samples),len(heldout.samples),
                        train.audit.shuffled_false_negative_fraction,
                        train.audit.no_disjoint_alternative,
                        auc,paired,brier,nll,best,
                        sum(p.numel() for p in trained.parameters()),
                        port.dimension,overlap,
                    ))
    return tuple(rows)


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",choices=[x.value for x in Candidate],
                        default=Candidate.SMALL_QWEN25.value)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--revision",default=None)
    parser.add_argument("--4bit",dest="quantized",action="store_true")
    parser.add_argument("--max-length",type=int,default=1024)
    parser.add_argument("--laws",nargs="+",choices=[x.value for x in StudyLaw],
                        default=["precision","operator"])
    parser.add_argument("--seeds",nargs="+",type=int,default=[0])
    parser.add_argument("--pool",nargs="+",choices=[x.value for x in PoolKind],
                        default=[x.value for x in PoolKind])
    parser.add_argument("--projection",nargs="+",choices=[x.value for x in ProjectionKind],
                        default=[x.value for x in ProjectionKind])
    parser.add_argument("--dimensions",nargs="+",type=int,default=[2,8,16,32])
    parser.add_argument("--pairing",nargs="+",
                        choices=[x.value for x in NegativeTraining],
                        default=[NegativeTraining.WITNESSED.value])
    parser.add_argument("--train",type=int,default=48)
    parser.add_argument("--calibration",type=int,default=16)
    parser.add_argument("--heldout",type=int,default=24)
    parser.add_argument("--replays",type=int,default=3)
    parser.add_argument("--steps",type=int,default=100)
    parser.add_argument("--output",default="")
    args=parser.parse_args()
    config=StudyConfig(args.train,args.calibration,args.heldout,
                       args.replays,args.steps)
    if args.candidate==Candidate.AGENTWORLD.value:
        parser.error("AgentWorld simulator is not a causal LM hidden-state checkpoint in this port")
    port=FrozenCausalLMTokenPort(CausalLMConfig(
        candidate=Candidate(args.candidate),revision=args.revision,
        device=args.device,max_length=args.max_length,
        quantized_4bit=args.quantized,
    ))
    result=[
        asdict(row) for law in args.laws for seed in args.seeds
        for row in benchmark_one(
            port=port,law=StudyLaw(law),seed=seed,config=config,
            pools=tuple(PoolKind(x) for x in args.pool),
            projections=tuple(ProjectionKind(x) for x in args.projection),
            dimensions=tuple(args.dimensions),
            pairing=tuple(NegativeTraining(x) for x in args.pairing),
        )
    ]
    payload={
        "model":port.model_id,"revision":port.revision,
        "replays_are_deterministic_fixture_only":True,
        "no_oracle_label_in_negative_miner":True,
        "no_kan_to_encoder_gradient":True,
        "config":asdict(config),"rows":result,
        "limitations":[
            "Replay evidence under a declared deterministic capability is necessary; "
            "stochastic/unknown worlds abstain and require calibrated outcome support.",
            "Benchmark fixture's _law_response is used ONLY to simulate repeated "
            "environment steps, not for negative mining, PCA or KAN learning.",
            "PCA fitted only on train X/Y positives; dimension comparisons alter KAN parameter count.",
            "An observed mismatch from a different intervention is a legitimate negative "
            "only given the declared deterministic full-state environment assumption.",
            "The shuffled_y_unsafe_control arm is FOR DIAGNOSTICS ONLY and deliberately "
            "contains false negative labels; evaluation remains replay-supported.",
            "Synthetic operator outcomes are binary, making shuffled-Y invalid often.",
            "AUROC is a pair-compatibility metric, not p(Y|do(X)) or causal identification.",
        ],
    }
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({
        "model":port.model_id,"revision":port.revision,
        "rows":len(result),
        "results":[
            {"law":r["law"],"pool":r["pool"],"projection":r["projection"],
             "dim":r["dimension"],"pairing":r["negative_training"],
             "auroc":round(r["heldout_auroc"],4),
             "pairs":r["train_pairs"]}
            for r in result
        ],
    },indent=2))


if __name__=="__main__":
    main()
