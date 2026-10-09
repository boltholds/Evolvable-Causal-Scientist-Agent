"""Reproducible offline DiscoveryWorld JEPA experiment from public Arena A logs.

This first stage tests predictive representation quality, NOT policy improvement.
The heldout seed is never seen in neural training, model selection or baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
from math import isfinite

import numpy as np
import torch

from ecsa.experimental.action_jepa import JepaConfig, ActionJEPA, train_jepa
from .jepa_replay import ReplayConfig, ReplaySplits, ReplayTransitions, load_splits


@dataclass(frozen=True)
class ProbeConfig:
    steps: int = 700
    batch_size: int = 128
    latent_dim: int = 8
    hidden_dim: int = 64
    seed: int = 0
    ridge_lambda: float = 1.0

    def __post_init__(self) -> None:
        for name in ('steps', 'batch_size', 'latent_dim', 'hidden_dim'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if type(self.ridge_lambda) not in (float, int) or not isfinite(self.ridge_lambda) or self.ridge_lambda <= 0:
            raise ValueError('ridge_lambda must be a finite positive number')


def _zero_actions(samples: ReplayTransitions) -> ReplayTransitions:
    return ReplayTransitions(samples.before, np.zeros_like(samples.actions),
                             samples.after, samples.action_names, samples.steps)


def _alternatives(train: ReplayTransitions, heldout: ReplayTransitions) -> tuple[np.ndarray, np.ndarray]:
    """Alternative action packets are public train examples of OTHER action types.

    Cannot pretend that evaluating these counterfactual packets demonstrates
    causal effects; this is only a conditional prediction discrimination score.
    """
    catalog: dict[str, np.ndarray] = {}
    for label, values in zip(train.action_names, train.actions):
        catalog.setdefault(label, values)
    ordered = sorted(catalog)
    alternatives = np.zeros_like(heldout.actions)
    eligible = np.zeros(len(heldout.actions), dtype=bool)
    for i, label in enumerate(heldout.action_names):
        other = next((x for x in ordered if x != label), None)
        if other is not None:
            alternatives[i] = catalog[other]
            eligible[i] = True
    return alternatives, eligible


def score_actions(model: ActionJEPA, train: ReplayTransitions,
                  heldout: ReplayTransitions, *, no_actions: bool = False) -> dict[str, float | int]:
    """Evaluate learned latent dynamics and action discrimination on fixed rows."""
    device = next(model.parameters()).device
    with torch.no_grad():
        before = torch.as_tensor(heldout.before, device=device)
        after = torch.as_tensor(heldout.after, device=device)
        act = torch.as_tensor(heldout.actions, device=device)
        z = model.encoder(before)
        truth = model.encoder(after)
        if no_actions:
            act = torch.zeros_like(act)
        prediction = model.predict_tensor(z, act)
        persistence = float((z-truth).square().mean())
        mse = float((prediction-truth).square().mean())
        alternatives, mask = _alternatives(train, heldout)
        if no_actions:
            alternatives = np.zeros_like(alternatives)
        wrong = model.predict_tensor(z, torch.as_tensor(alternatives,device=device))
        true_distance = (prediction-truth).square().sum(dim=1).cpu().numpy()
        wrong_distance = (wrong-truth).square().sum(dim=1).cpu().numpy()
    if np.any(mask):
        accuracy = float(np.mean((true_distance[mask] < wrong_distance[mask]).astype(np.float64)
                                 + .5 * (true_distance[mask] == wrong_distance[mask])))
    else:
        accuracy = .5
    with torch.no_grad():
        latents = model.encoder(torch.as_tensor(heldout.before,device=device)).float().cpu().numpy()
    centered=latents-latents.mean(axis=0,keepdims=True)
    singular=np.linalg.svd(centered,compute_uv=False)
    spectrum=singular.astype(np.float64)**2
    rank=float(spectrum.sum()**2/max(np.square(spectrum).sum(),1e-12))
    return {
        'heldout_mse': mse,
        'persistence_mse': persistence,
        'improvement_over_persistence': float(1.-mse/max(persistence,1e-12)),
        'action_choice_accuracy': accuracy,
        'action_choice_pairs': int(mask.sum()),
        'mean_latent_std': float(latents.std(axis=0).mean()),
        'effective_rank': rank,
    }


class _RawRidge:
    """Multivariate ridge baseline on the same public feature/action input."""
    def __init__(self, source: ReplayTransitions, regularizer: float) -> None:
        features=np.concatenate((source.before,source.actions,
                    np.ones((len(source.before),1),dtype=np.float32)),axis=1).astype(np.float64)
        target=(source.after-source.before).astype(np.float64)
        alpha=regularizer*np.eye(features.shape[1]);alpha[-1,-1]=1e-8
        self._coefficients=np.linalg.solve(features.T@features+alpha,features.T@target)

    def predict(self, before: np.ndarray, actions: np.ndarray) -> np.ndarray:
        x=np.concatenate((before,actions,np.ones((len(before),1))),axis=1).astype(np.float64)
        return before + x@self._coefficients


def _score_raw_ridge(model: _RawRidge, train: ReplayTransitions, heldout: ReplayTransitions) -> dict[str, float | int]:
    predicted=model.predict(heldout.before,heldout.actions)
    true=heldout.after.astype(np.float64)
    mse=float(np.square(predicted-true).mean())
    persistence=float(np.square(heldout.before-true).mean())
    alternatives,mask=_alternatives(train,heldout)
    wrong=model.predict(heldout.before, alternatives)
    true_dist=np.square(predicted-true).sum(axis=1)
    wrong_dist=np.square(wrong-true).sum(axis=1)
    accuracy=(float(np.mean((true_dist[mask]<wrong_dist[mask]).astype(np.float64)
               +.5*(true_dist[mask]==wrong_dist[mask]))) if np.any(mask) else .5)
    return {
        'heldout_mse':mse,
        'persistence_mse':persistence,
        'improvement_over_persistence':float(1.-mse/max(persistence,1e-12)),
        'action_choice_accuracy':accuracy,
        'action_choice_pairs':int(mask.sum()),
    }


def run_probe(splits: ReplaySplits, config: ProbeConfig, *, device: str) -> dict:
    """JEPA vs action-shuffle/no-action and strong raw-vector ridge.

    Independent encoders have different latent frames: compare dimensionless
    within-arm persistence gains and action ranking, NEVER raw neural MSEs.
    """
    if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):
        raise ValueError('device unavailable')
    model_config=JepaConfig(latent_dim=config.latent_dim,hidden_dim=config.hidden_dim,
                            batch_size=config.batch_size,steps=config.steps,seed=config.seed)
    arms = {}
    for name in ('jepa','action_shuffled','no_action'):
        train = _zero_actions(splits.train) if name == 'no_action' else splits.train
        model,report=train_jepa(train,model_config,device=device,
                                shuffle_actions=(name=='action_shuffled'))
        metrics=score_actions(model,splits.train,splits.test,no_actions=(name=='no_action'))
        valid=score_actions(model,splits.train,splits.validation,no_actions=(name=='no_action'))
        arms[name] = dict(**metrics, validation_improvement_over_persistence=valid['improvement_over_persistence'],
                          training_start_loss=report.start_loss,training_end_loss=report.end_loss)
    baseline=_RawRidge(splits.train,config.ridge_lambda)
    arms['raw_ridge']=_score_raw_ridge(baseline,splits.train,splits.test)
    return {
        'protocol':'discoveryworld_action_jepa_replay_v1',
        'split_kind':splits.split_kind,
        'train_seeds':list(splits.train_seeds),
        'validation_seeds':list(splits.validation_seeds),
        'test_seeds':list(splits.test_seeds),
        'train_count':len(splits.train.before),
        'validation_count':len(splits.validation.before),
        'heldout_count':len(splits.test.before),
        'train_action_types':sorted(set(splits.train.action_names)),
        'test_action_types':sorted(set(splits.test.action_names)),
        'device':device,
        'config':{
            'steps':config.steps,'batch_size':config.batch_size,
            'latent_dim':config.latent_dim,'hidden_dim':config.hidden_dim,
            'seed':config.seed,'ridge_lambda':config.ridge_lambda,
        },
        'arms':arms,
        'limitations':[
            'Offline representation probe; all arms see the same logged actions and neither solves DiscoveryWorld interactively.',
            'No oracle scorecard or hidden world values are read for training, validation or test.',
            'Hashing public JSON is fixed manual featurization; learning raw visual/text perception remains future work.',
            'Different JEPA arms learn distinct latent coordinate frames: compare within-arm ratios and action ranking, not raw MSE across arms.',
            'Wrong-action discrimination is observational and does not demonstrate counterfactual causality.',
            'A single-seed smoke split is temporally correlated and proves no cross-seed generalization.',
            'Study mode excludes all heldout seed transitions from training and validation.',
            'Identical cold/reuse episodes from one seed are never counted as independent seeds.',
            'Action-success rate and official scoreNormalized are not estimated in this replay probe.',
        ],
    }


def run_probe_from_logs(log_root: Path, *, output: Path, mode: str='smoke',arm: str='cold',
                        device: str='cpu',observation_dim: int=256,action_dim: int=128,
                        steps: int=700,batch_size: int=128,latent_dim: int=8,
                        hidden_dim: int=64,seed: int=0) -> dict:
    replay_config=ReplayConfig(observation_dim=observation_dim,action_dim=action_dim)
    splits=load_splits(log_root,replay_config,mode=mode,arm=arm)
    report=run_probe(splits,ProbeConfig(steps=steps,batch_size=batch_size,
                                       latent_dim=latent_dim,hidden_dim=hidden_dim,seed=seed),device=device)
    report['feature_widths']={'observation':observation_dim,'action':action_dim}
    report['source_arm']=arm
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return report


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description='JEPA on public DiscoveryWorld action/observation logs')
    parser.add_argument('--runs',type=Path,required=True,help='Arena A output root with seed-N/cold/{actions,observations}.jsonl')
    parser.add_argument('--mode',choices=('smoke','study'),default='smoke')
    parser.add_argument('--arm',choices=('cold','reuse'),default='cold')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=700)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--observation-dim',type=int,default=256)
    parser.add_argument('--action-dim',type=int,default=128)
    parser.add_argument('--latent-dim',type=int,default=8)
    parser.add_argument('--hidden-dim',type=int,default=64)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    report=run_probe_from_logs(args.runs,output=args.output,mode=args.mode,arm=args.arm,
        device=args.device,observation_dim=args.observation_dim,action_dim=args.action_dim,
        steps=args.steps,batch_size=args.batch_size,latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,seed=args.seed)
    print(json.dumps({k:v for k,v in report.items() if k not in ('limitations','train_action_types','test_action_types')},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
