"""Evidence-calibrated online routing over saved JEPA predictive mechanisms.

The router sees ONLY observed transitions, public actions, checkpoint IDs and
prechange calibration residuals. It never imports simulator law or epoch types.
Low support across ALL stored hypotheses is a proposal, NEVER automatic admission.

The discretized joint target predictive distributions make each Bayesian update
valid for ONE common observed outcome, rather than comparing model-specific
residual events. Gaussian/diagonal/bin independence remain explicit assumptions.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import erf, isfinite, sqrt
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from ecsa.contracts import (
    Observation, PopulationAnomaly, PosteriorUpdate, PredictiveDistribution,
    TheoryPosterior, TheoryRef, ExperimentScore,
)
from ecsa.science import ScienceKernel

from .action_jepa import ActionJEPA
from .jepa_world import Action, TransitionDataset, action_vector, _ACTIONS


@runtime_checkable
class PredictivePort(Protocol):
    @property
    def space_id(self) -> str: ...
    def encode(self, observations: np.ndarray) -> np.ndarray: ...
    def predict(self, before: np.ndarray, actions: np.ndarray) -> np.ndarray: ...


class JepaPredictivePort:
    """Frozen inference over a checkpoint with a reproducible encoder identity."""

    def __init__(self, model: ActionJEPA) -> None:
        if not isinstance(model, ActionJEPA):
            raise TypeError('ActionJEPA checkpoint required')
        self._model = deepcopy(model).eval()
        self._model.requires_grad_(False)
        digest = sha256()
        for name, tensor in self._model.encoder.state_dict().items():
            data = tensor.detach().cpu().contiguous().numpy()
            digest.update(name.encode())
            digest.update(str(data.shape).encode())
            digest.update(data.tobytes())
        self._space_id = 'jepa-encoder-sha256:' + digest.hexdigest()

    @property
    def space_id(self) -> str:
        return self._space_id

    def encode(self, observations: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.encode(observations), dtype=np.float64)

    def predict(self, before: np.ndarray, actions: np.ndarray) -> np.ndarray:
        values = np.asarray(before, dtype=np.float32)
        action_matrix = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or action_matrix.shape != (len(values),len(_ACTIONS)):
            raise ValueError('expected observed [N,D] states and one-hot actions')
        device = next(self._model.parameters()).device
        with torch.inference_mode():
            z = self._model.encoder(torch.as_tensor(values, device=device))
            predicted = self._model.predict_tensor(z,torch.as_tensor(action_matrix,device=device))
        return predicted.detach().float().cpu().numpy().astype(np.float64)


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    predictor: PredictivePort
    calibration: TransitionDataset
    calibration_ids: tuple[str,...]
    training_ids: tuple[str,...]

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_id,str) or not self.checkpoint_id:
            raise ValueError('checkpoint ID required')
        if not isinstance(self.predictor, PredictivePort):
            raise TypeError('typed predictive port required')
        if not isinstance(self.calibration, TransitionDataset) or len(self.calibration.before)<32:
            raise ValueError('>=32 observed calibration transitions required')
        if len(self.calibration_ids) != len(self.calibration.before):
            raise ValueError('calibration IDs must match observations')
        ids=self.calibration_ids+self.training_ids
        if any(not isinstance(v,str) or not v for v in ids) or len(ids)!=len(set(ids)):
            raise ValueError('unique nonempty calibration/training provenance required')


@dataclass(frozen=True)
class RouterConfig:
    switch_hazard: float = 0.12
    accept_probability: float = 0.75
    novelty_p: float = 0.025
    novelty_streak: int = 3
    novelty_window: int = 18
    min_confirmations: int = 24
    promotion_ratio: float = 0.75
    population_anomaly_threshold: float = 0.000001

    def __post_init__(self) -> None:
        for name in ('switch_hazard','accept_probability','novelty_p','promotion_ratio'):
            v=getattr(self,name)
            if not isinstance(v,(int,float)) or isinstance(v,bool) or not isfinite(v) or not 0<float(v)<1:
                raise ValueError(f'{name} must be in (0,1)')
        for name in ('novelty_streak','novelty_window','min_confirmations'):
            v=getattr(self,name)
            if type(v) is not int or v<1:
                raise ValueError(f'{name} must be positive')
        if self.novelty_window < self.novelty_streak:
            raise ValueError('novelty window must cover required evidence')
        if not isfinite(self.population_anomaly_threshold) or self.population_anomaly_threshold<0:
            raise ValueError('invalid population anomaly threshold')


class RouteKind(StrEnum):
    REUSE_KNOWN = 'reuse_known'
    DISAMBIGUATE = 'disambiguate'
    PROPOSE_NEW = 'propose_new'


@dataclass(frozen=True)
class NoveltyProposal:
    proposal_id: str
    evidence_ids: tuple[str,...]
    strongest_tail_probability: float


@dataclass(frozen=True)
class RouterDecision:
    transition_id: str
    kind: RouteKind
    selected_checkpoint: str | None
    posterior: TheoryPosterior
    evidence_probability: float
    smallest_tail_probability: float
    proposed_hypothesis: NoveltyProposal | None
    change_point_signal: bool


@dataclass(frozen=True)
class ConfirmationEvidence:
    hypothesis_id: str
    validation: TransitionDataset
    validation_ids: tuple[str,...]

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError('hypothesis ID required')
        if not isinstance(self.validation,TransitionDataset):
            raise TypeError('observed transition dataset required')
        if len(self.validation_ids)!=len(self.validation.before) or any(
            not isinstance(x,str) or not x for x in self.validation_ids
        ) or len(set(self.validation_ids))!=len(self.validation_ids):
            raise ValueError('unique matching validation evidence IDs required')


@dataclass(frozen=True)
class AdmissionResult:
    admitted: bool
    candidate_mse: float
    best_existing_mse: float
    evidence_count: int
    known_regime_overlap: bool


@dataclass(frozen=True)
class _Projection:
    center: np.ndarray
    basis: np.ndarray
    edges: tuple[np.ndarray,np.ndarray]

    @classmethod
    def fit(cls, first: PredictivePort, data: TransitionDataset) -> '_Projection':
        y=np.asarray(first.encode(data.after),dtype=np.float64)
        if y.ndim!=2 or y.shape[0]!=len(data.before) or y.shape[1]<2 or not np.isfinite(y).all():
            raise ValueError('latent observations require at least two finite dimensions')
        center=y.mean(axis=0)
        _,singular,vt=np.linalg.svd(y-center,full_matrices=False)
        if len(singular)<2 or singular[1]<1e-10:
            raise ValueError('calibration representation has insufficient rank')
        basis=vt[:2].T
        reduced=(y-center)@basis
        edges=tuple(np.quantile(reduced[:,i],(.12,.29,.5,.71,.88)) for i in range(2))
        for e in edges:
            if np.diff(e).min()<1e-8:
                raise ValueError('calibration quantization is degenerate')
        return cls(center,basis,(edges[0],edges[1]))

    def project(self, states: np.ndarray) -> np.ndarray:
        y=np.asarray(states,dtype=np.float64)
        if y.ndim!=2 or y.shape[1]!=len(self.center) or not np.isfinite(y).all():
            raise ValueError('invalid shared latent space')
        return (y-self.center)@self.basis

    def outcome(self, latent: np.ndarray) -> tuple[int,int]:
        z=self.project(latent.reshape(1,-1))[0]
        return tuple(int(np.searchsorted(self.edges[i],z[i],side='right')) for i in (0,1))

    def probabilities(self, mean: np.ndarray, standard_deviation: np.ndarray) -> tuple[tuple[tuple[int,int],float],...]:
        projected=self.project(mean.reshape(1,-1))[0]
        axis=[]
        for d in range(2):
            norm=(self.edges[d]-projected[d])/(sqrt(2)*standard_deviation[d])
            cumulative=np.array([0.,*[.5*(1.+erf(float(q))) for q in norm],1.])
            mass=np.maximum(np.diff(cumulative),1e-12)
            mass/=mass.sum()
            axis.append(mass)
        grid=np.outer(axis[0],axis[1]); grid/=grid.sum()
        return tuple(((i,j),float(grid[i,j])) for i in range(6) for j in range(6))


class _Density:
    def __init__(self, checkpoint: Checkpoint, projection: _Projection) -> None:
        self.checkpoint=checkpoint
        calibration=checkpoint.calibration
        pred=projection.project(checkpoint.predictor.predict(calibration.before,calibration.actions))
        true=projection.project(checkpoint.predictor.encode(calibration.after))
        residual=pred-true
        if residual.shape!=(len(calibration.before),2):
            raise ValueError('invalid calibration predictions')
        self.sigma=np.maximum(np.sqrt(np.mean(residual**2,axis=0)),.02)
        self.null_squared=np.sort(np.sum((residual/self.sigma)**2,axis=1))
        self.projection=projection

    def predict(self, before: np.ndarray, action: np.ndarray) -> np.ndarray:
        result=np.asarray(self.checkpoint.predictor.predict(before.reshape(1,-1),action.reshape(1,-1)),dtype=np.float64)
        if result.shape!=(1,len(self.projection.center)) or not np.isfinite(result).all():
            raise ValueError('expert returned incompatible or nonfinite latent prediction')
        return result[0]

    def density(self, predicted: np.ndarray, experiment_id: str) -> PredictiveDistribution:
        return PredictiveDistribution(
            theory_id=self.checkpoint.checkpoint_id,experiment_id=experiment_id,
            probabilities=self.projection.probabilities(predicted,self.sigma),
        )

    def tail_p(self, predicted: np.ndarray, target: np.ndarray) -> float:
        difference=self.projection.project(predicted.reshape(1,-1))[0]-self.projection.project(target.reshape(1,-1))[0]
        t=float(np.sum((difference/self.sigma)**2))
        return float((1+len(self.null_squared)-np.searchsorted(self.null_squared,t,'left'))/(len(self.null_squared)+1))


class MechanismRouter:
    """One-step sequential Bayesian hypothesis router with open-set abstention.

    Existing models remain frozen. Unknown is proposed from repeated calibrated
    outliers, NEVER admitted until fresh corroboration and candidate superiority.
    """

    def __init__(self, checkpoints: tuple[Checkpoint,...],*,config:RouterConfig|None=None,
                 science:ScienceKernel|None=None,change_detector:'BOCPDChangeSignal|None'=None):
        if not isinstance(checkpoints,tuple) or not checkpoints or not all(isinstance(c,Checkpoint) for c in checkpoints):
            raise TypeError('immutable nonempty checkpoint tuple required')
        self.config=config if config is not None else RouterConfig()
        if not isinstance(self.config,RouterConfig):raise TypeError('RouterConfig required')
        ids=[c.checkpoint_id for c in checkpoints]
        if len(set(ids))!=len(ids):raise ValueError('checkpoint IDs must be unique')
        space=checkpoints[0].predictor.space_id
        if any(c.predictor.space_id!=space for c in checkpoints):
            raise ValueError('all checkpoints require the same latent coordinate system')
        width=checkpoints[0].calibration.before.shape[1]
        if any(c.calibration.before.shape[1]!=width for c in checkpoints):
            raise ValueError('all calibration observations require same width')
        all_provenance=tuple(i for c in checkpoints for i in c.calibration_ids+c.training_ids)
        if len(all_provenance)!=len(set(all_provenance)):
            raise ValueError('training and calibration cohorts must be disjoint')
        self._space_id=space
        self._observation_width=width
        self._reference=checkpoints[0].predictor
        self._projection=_Projection.fit(self._reference,checkpoints[0].calibration)
        self._densities={c.checkpoint_id:_Density(c,self._projection) for c in checkpoints}
        self._posterior=TheoryPosterior.uniform(tuple(TheoryRef(id,id) for id in ids))
        self._science=science if science is not None else ScienceKernel(
            population_anomaly_threshold=self.config.population_anomaly_threshold)
        if not isinstance(self._science,ScienceKernel):raise TypeError('ScienceKernel required')
        self._observed_ids:set[str]=set()
        self._novelty=deque(maxlen=self.config.novelty_window)
        self._pending:NoveltyProposal|None=None
        self._change_detector=change_detector

    @property
    def checkpoint_ids(self) -> tuple[str,...]:return tuple(self._densities)

    @property
    def pending_proposal(self) -> NoveltyProposal|None:return self._pending

    @property
    def posterior(self) -> TheoryPosterior:return self._posterior

    def _prior_with_hazard(self) -> TheoryPosterior:
        h=self.config.switch_hazard
        n=len(self._densities)
        return TheoryPosterior(tuple((key, (1-h)*p+h/n) for key,p in self._posterior.probabilities))

    def _predictive(self, before: np.ndarray, action: np.ndarray, experiment_id: str):
        predictions=[]; outputs={}
        for key,density in self._densities.items():
            predicted=density.predict(before,action)
            outputs[key]=predicted
            predictions.append(density.density(predicted,experiment_id))
        return tuple(predictions),outputs

    def _validate_state(self, before: np.ndarray, action: np.ndarray, after: np.ndarray|None=None):
        src=np.asarray(before,dtype=np.float64)
        act=np.asarray(action,dtype=np.float64)
        if src.shape!=(self._observation_width,) or not np.isfinite(src).all():
            raise ValueError('nonfinite or wrong-width observations')
        if act.shape!=(len(_ACTIONS),) or not np.isfinite(act).all() or not np.array_equal(act,act.astype(bool)) or act.sum()!=1:
            raise ValueError('invalid one-hot action')
        if after is None:return src,act,None
        target=np.asarray(after,dtype=np.float64)
        if target.shape!=src.shape or not np.isfinite(target).all():
            raise ValueError('nonfinite or wrong-width observations')
        return src,act,target

    def observe(self,*,transition_id:str,before:np.ndarray,action:np.ndarray,after:np.ndarray)->RouterDecision:
        if not isinstance(transition_id,str) or not transition_id:
            raise ValueError('nonempty transition provenance required')
        if transition_id in self._observed_ids:
            raise ValueError('duplicate observed transition ID')
        src,act,target=self._validate_state(before,action,after)
        assert target is not None
        projected=self._reference.encode(target.reshape(1,-1))
        if projected.shape!=(1,len(self._projection.center)):
            raise ValueError('reference encoder changed latent space')
        observation=Observation(transition_id,self._projection.outcome(projected[0]))
        predictions,outputs=self._predictive(src,act,transition_id)
        prior=self._prior_with_hazard()
        update=self._science.update(prior,predictions,observation)
        if isinstance(update,PosteriorUpdate):
            self._posterior=update.posterior
        elif isinstance(update,PopulationAnomaly):
            self._posterior=prior
        else:raise TypeError('unexpected science kernel update')
        tail={key:density.tail_p(outputs[key],projected[0]) for key,density in self._densities.items()}
        novelty=all(p<=self.config.novelty_p for p in tail.values())
        self._observed_ids.add(transition_id)
        self._novelty.append((transition_id,novelty,min(tail.values())))
        hits=tuple(id for id,detected,_ in self._novelty if detected)
        if len(hits)>=self.config.novelty_streak and self._pending is None:
            material=','.join(hits).encode()
            self._pending=NoveltyProposal('novel-'+sha256(material).hexdigest()[:20],hits,
                                            min(q for _,flag,q in self._novelty if flag))
        change_point_signal=False
        if self._change_detector is not None:
            surprise=-np.log(max(update.evidence_probability,1e-12))
            change_point_signal=bool(self._change_detector.add(float(surprise)))
        if self._pending is not None:
            kind=RouteKind.PROPOSE_NEW
            selected=None
        else:
            selected,p=max(self._posterior.probabilities,key=lambda pair:pair[1])
            if p>=self.config.accept_probability and not change_point_signal:
                kind=RouteKind.REUSE_KNOWN
            else:
                kind=RouteKind.DISAMBIGUATE
                selected=None
        return RouterDecision(transition_id,kind,selected,self._posterior,update.evidence_probability,
                              min(tail.values()),self._pending,change_point_signal)

    def select_experiment(self,before:np.ndarray,*,candidate_actions:tuple[Action,...])->tuple[Action,ExperimentScore]:
        if not isinstance(candidate_actions,tuple) or not candidate_actions or len(set(candidate_actions))!=len(candidate_actions) or any(not isinstance(a,Action) for a in candidate_actions):
            raise ValueError('distinct typed public actions required')
        src,_,_=self._validate_state(before,action_vector(candidate_actions[0]))
        candidates=[]
        for action in candidate_actions:
            proposals,_=self._predictive(src,action_vector(action),'prospective:'+action.value)
            candidates.append((action,self._science.score_experiment(self._posterior,proposals)))
        return max(candidates,key=lambda item:(item[1].information_gain_bits,-candidate_actions.index(item[0])))

    def _mse(self,port:PredictivePort,validation:TransitionDataset)->float:
        predictions=np.asarray(port.predict(validation.before,validation.actions),dtype=np.float64)
        reference=np.asarray(self._reference.encode(validation.after),dtype=np.float64)
        if predictions.shape!=reference.shape or not np.isfinite(predictions).all():
            raise ValueError('candidate latent predictions must be finite in shared frame')
        return float(np.square(predictions-reference).mean())

    def admit_checkpoint(self,checkpoint:Checkpoint,evidence:ConfirmationEvidence,*,prior_mass:float=.12)->AdmissionResult:
        if self._pending is None or evidence.hypothesis_id!=self._pending.proposal_id:
            raise ValueError('admission requires existing matching proposal')
        if checkpoint.checkpoint_id in self._densities:
            raise ValueError('checkpoint already stored')
        if checkpoint.predictor.space_id != self._space_id:
            raise ValueError('new checkpoint must share the latent coordinate system')
        if len(evidence.validation.before)<self.config.min_confirmations:
            raise ValueError('independent confirmation sample budget not satisfied')
        reserved=self._observed_ids|{v for d in self._densities.values() for v in d.checkpoint.calibration_ids+d.checkpoint.training_ids}
        reserved.update(checkpoint.training_ids+checkpoint.calibration_ids)
        if reserved.intersection(evidence.validation_ids) or set(checkpoint.training_ids).intersection(checkpoint.calibration_ids):
            raise ValueError('confirmation IDs must be disjoint from proposal/training/calibration and prior observations')
        candidate_mse=self._mse(checkpoint.predictor,evidence.validation)
        best=min(self._mse(d.checkpoint.predictor,evidence.validation) for d in self._densities.values())
        # Superior prediction on an already stored regime indicates a
        # refinement, not a separately identified causal mechanism.
        known_overlap=any(
            self._mse(checkpoint.predictor,existing.checkpoint.calibration)
            < self.config.promotion_ratio * self._mse(
                existing.checkpoint.predictor,existing.checkpoint.calibration)
            for existing in self._densities.values()
        )
        admitted=(candidate_mse<self.config.promotion_ratio*best and not known_overlap)
        if admitted:
            new_density=_Density(checkpoint,self._projection)
            self._posterior=self._science.admit_theory(self._posterior,
                 TheoryRef(checkpoint.checkpoint_id,checkpoint.checkpoint_id),prior_mass=prior_mass)
            self._densities[checkpoint.checkpoint_id]=new_density
            self._pending=None
            self._novelty.clear()
        else:
            # A rejected independent test must not keep the router stuck in
            # PROPOSE_NEW after the world returns to a known regime.
            self._pending=None
            self._novelty.clear()
        return AdmissionResult(admitted,candidate_mse,best,len(evidence.validation.before),known_overlap)


class BOCPDChangeSignal:
    """Optional bounded BOCPD diagnostic using the repository's existing bocd extra.

    The Bayesian mechanism posterior and open-set safeguard work without it.
    This adapter reports a candidate changepoint, never admits a new theory.
    """
    def __init__(self,*,min_points:int=24,min_reset_drop:int=8,hazard_rate:float=.01):
        if type(min_points) is not int or min_points<4 or type(min_reset_drop) is not int or min_reset_drop<1:
            raise ValueError('invalid BOCPD window')
        if not isfinite(hazard_rate) or not 0<hazard_rate<1:
            raise ValueError('hazard must be in (0,1)')
        self.min_points=min_points
        self.min_reset_drop=min_reset_drop
        self.hazard_rate=hazard_rate
        self._series:list[float]=[]

    def add(self,surprise:float)->bool:
        if not isfinite(surprise):raise ValueError('finite surprise required')
        try:
            from bocd import ConstantHazard, Detector, GaussianModel
        except ImportError as exc:
            raise RuntimeError('bocpd optional extra required: pip install -e ".[bocpd]"') from exc
        self._series.append(float(surprise))
        if len(self._series)<self.min_points:return False
        detector=Detector(model=GaussianModel(),hazard=ConstantHazard(rate=self.hazard_rate),
                          max_run_length=500)
        results=detector.fit(self._series)
        if len(results)<2:return False
        previous=results[-2].most_likely_run_length
        recent=results[-1].most_likely_run_length
        return previous-recent>=self.min_reset_drop
