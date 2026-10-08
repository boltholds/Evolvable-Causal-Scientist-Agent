"""Predictor-only continual adaptation on opaque action/observation transitions.

No simulator mechanism, changepoint, oracle variables, or benchmark state IDs.
Frozen encoder coordinates make pre/post-repair forgetting directly comparable.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch
from torch.nn import functional as F

from .action_jepa import ActionJEPA
from .jepa_world import TransitionDataset


@dataclass(frozen=True)
class AdaptationConfig:
    steps: int = 120
    batch_size: int = 64
    max_new_examples: int = 64
    max_replay_examples: int = 64
    replay_fraction: float = .5
    lr: float = .001
    checkpoint_every: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        for label in ('steps','batch_size','max_new_examples','checkpoint_every'):
            value = getattr(self,label)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{label} must be positive')
        if type(self.max_replay_examples) is not int or self.max_replay_examples < 0:
            raise ValueError('max_replay_examples must be nonnegative')
        if not isfinite(self.replay_fraction) or not 0 <= self.replay_fraction < 1:
            raise ValueError('replay fraction must be in [0,1)')
        if not isfinite(self.lr) or self.lr <= 0:
            raise ValueError('lr must be finite and positive')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be nonnegative')


@dataclass(frozen=True)
class AdaptationTrace:
    updates: int
    new_examples: int
    old_examples: int
    training_loss_initial: float
    training_loss_final: float
    validation_mse_by_update: tuple[tuple[int,float], ...]
    recovery_update_20pct: int | None
    encoder_frozen: bool


@dataclass(frozen=True)
class RouteVote:
    selected_checkpoint: str
    prechange_mse: float
    adapted_mse: float
    count: int


def _tensor_states(model: ActionJEPA, data: TransitionDataset) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    device = next(model.parameters()).device
    with torch.no_grad():
        source = model.encoder(torch.as_tensor(data.before,device=device)).detach()
        target = model.encoder(torch.as_tensor(data.after,device=device)).detach()
    action = torch.as_tensor(data.actions,device=device)
    return source,action,target


def predict_mse(model: ActionJEPA,data: TransitionDataset) -> float:
    if not isinstance(data,TransitionDataset) or len(data.before)<1:
        raise ValueError('nonempty transition dataset required')
    model.eval()
    source,action,target=_tensor_states(model,data)
    with torch.inference_mode():
        score=float(F.mse_loss(model.predict_tensor(source,action),target))
    if not isfinite(score):
        raise ValueError('nonfinite prediction MSE')
    return score


def adapt_predictor(
    pretrained: ActionJEPA, new: TransitionDataset, *,
    config: AdaptationConfig, old_replay: TransitionDataset | None = None,
    validation: TransitionDataset | None = None,
    require_replay: bool = False,
) -> tuple[ActionJEPA,AdaptationTrace]:
    """Same pretrained checkpoint for every arm; no representation drift.

    `validation` only produces an auditable trajectory and does not control
    optimizer, gradients, early stopping, or the returned checkpoint.
    """
    if not isinstance(pretrained,ActionJEPA) or not isinstance(config,AdaptationConfig):
        raise TypeError('typed JEPA and adaptation configuration required')
    if not isinstance(new,TransitionDataset) or len(new.before)<1:
        raise ValueError('new observed transitions required')
    if require_replay and old_replay is None:
        raise ValueError('required replay buffer is missing')
    if old_replay is not None and (not isinstance(old_replay,TransitionDataset) or len(old_replay.before)<1):
        raise ValueError('valid replay transitions required')
    if old_replay is not None and config.max_replay_examples==0:
        raise ValueError('replay memory budget must be positive')
    if validation is not None and not isinstance(validation,TransitionDataset):
        raise TypeError('validation requires observed transitions')
    width=new.before.shape[1]
    if any(data is not None and data.before.shape[1]!=width for data in (old_replay,validation)):
        raise ValueError('all observations must share sensor dimension')

    model=deepcopy(pretrained)
    model.encoder.requires_grad_(False)
    model.target_encoder.requires_grad_(False)
    model.eval()
    rng=np.random.default_rng(config.seed)

    new_indices=rng.permutation(len(new.before))[:config.max_new_examples]
    new_subset=TransitionDataset(new.before[new_indices],new.actions[new_indices],new.after[new_indices])
    nz,na,nt=_tensor_states(model,new_subset)
    replay_size=0
    if old_replay is not None:
        old_indices=rng.permutation(len(old_replay.before))[:config.max_replay_examples]
        old_subset=TransitionDataset(old_replay.before[old_indices],old_replay.actions[old_indices],old_replay.after[old_indices])
        oz,oa,ot=_tensor_states(model,old_subset)
        replay_size=len(old_indices)
    optimizer=torch.optim.AdamW(model.predictor.parameters(),lr=config.lr,weight_decay=1e-4)
    history: list[tuple[int,float]]=[]
    if validation is not None:
        history.append((0,predict_mse(model,validation)))
    start_loss=last_loss=float('nan')
    for step in range(config.steps):
        batch_old=int(round(config.batch_size*config.replay_fraction)) if replay_size else 0
        batch_new=config.batch_size-batch_old
        pick_new=torch.as_tensor(rng.integers(0,len(new_indices),batch_new),device=nz.device)
        zz=[nz[pick_new]]
        aa=[na[pick_new]]
        yy=[nt[pick_new]]
        if batch_old:
            pick_old=torch.as_tensor(rng.integers(0,replay_size,batch_old),device=oz.device)
            zz.append(oz[pick_old]);aa.append(oa[pick_old]);yy.append(ot[pick_old])
        source=torch.cat(zz,dim=0)
        action=torch.cat(aa,dim=0)
        target=torch.cat(yy,dim=0)
        loss=F.mse_loss(model.predict_tensor(source,action),target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step==0:
            start_loss=float(loss.detach())
        last_loss=float(loss.detach())
        completed=step+1
        if validation is not None and (completed%config.checkpoint_every==0 or completed==config.steps):
            history.append((completed,predict_mse(model,validation)))
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    recovery=None
    if history:
        target_mse=history[0][1]*.8
        recovery=next((step for step,loss in history[1:] if loss<=target_mse),None)
    return model,AdaptationTrace(
        updates=config.steps, new_examples=len(new_indices),old_examples=replay_size,
        training_loss_initial=start_loss,training_loss_final=last_loss,
        validation_mse_by_update=tuple(history),
        recovery_update_20pct=recovery,encoder_frozen=True,
    )


def route_by_prediction(
    prechange: ActionJEPA, adapted: ActionJEPA,
    observations: TransitionDataset, *, window: int = 16,
) -> RouteVote:
    """Two stored predictive experts vote without mechanism or time metadata."""
    if type(window) is not int or window < 4 or window>len(observations.before):
        raise ValueError('invalid routing window')
    sample=TransitionDataset(observations.before[-window:],observations.actions[-window:],observations.after[-window:])
    original=predict_mse(prechange,sample)
    modified=predict_mse(adapted,sample)
    return RouteVote(
        selected_checkpoint='prechange' if original<=modified else 'adapted',
        prechange_mse=original,adapted_mse=modified,count=window,
    )
