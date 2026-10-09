"""DiscoveryWorld *adapter* for the domain-neutral SharedOutcomeFrame.

Uses only previously logged agent-visible pre/action/outcome/post records.
Benchmark-specific fields stay here; ECSA's world_model core never imports
DiscoveryWorld or exposes evaluator-only scorecards.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import argparse
import json

import numpy as np

from ecsa.world_model.contracts import (
    GroundAction, RawObservation, RawActionOutcome, InteractionTransition,
    freeze_raw_value,
)
from ecsa.world_model.perception.structured import StructuredObservationFrontend
from ecsa.world_model.shared_evaluation import (
    PublicOutcomeTransition, SharedOutcomeFrame, SharedEvaluation, SharedForecast,
    TargetKind,
)
from ecsa.world_model.predictive_adapters import PersistencePort, RawRidgePort

from .interface_induction import load_contract_episode
from .perception import DiscoveryWorldStructuredDecoder


def load_public_outcome_episode(directory: Path) -> tuple[PublicOutcomeTransition,...]:
    """Read public records only; never scorecard or hidden evaluator state."""
    samples=load_contract_episode(Path(directory))
    decoder=StructuredObservationFrontend(DiscoveryWorldStructuredDecoder())
    transitions=[]
    for i,row in enumerate(samples):
        if row.success is None:
            raise ValueError("public action outcome required; no invented success value")
        before=RawObservation(
            f"{row.evidence_id}:pre",i,
            freeze_raw_value({"observation":row.before}),
        )
        after=RawObservation(
            f"{row.evidence_id}:post",i+1,
            freeze_raw_value({"observation":row.after}),
        )
        packet=row.action
        if isinstance(packet.get("action"),str):
            schema_id=packet["action"]
            args=tuple(freeze_raw_value(packet[key])
                       for key in sorted(packet) if key!="action")
        else:
            schema_id="DIALOG_OPTION"
            args=(freeze_raw_value(packet["chosen_dialog_option_int"]),)
        action=GroundAction(schema_id,args)
        transition=InteractionTransition(
            row.evidence_id,before,action,
            RawActionOutcome(row.success,freeze_raw_value({"success":row.success})),
            after,
        )
        transitions.append(PublicOutcomeTransition(
            transition,decoder.perceive(before),decoder.perceive(after),
        ))
    return tuple(transitions)


def _scores_as_dict(scores: dict) -> dict:
    return {k:asdict(v) for k,v in scores.items()}


def _latent_forecasts(
    source: Path,
    frame: SharedOutcomeFrame,
    train: tuple[PublicOutcomeTransition,...],
    heldout: tuple[PublicOutcomeTransition,...],
    *,
    representation: str,
    device: str,
    steps: int,
    batch_size: int,
    seed: int,
):
    """Optional CUDA JEPA arm, projected to the same frame using train-only readout."""
    import torch
    from ecsa.experimental.action_jepa import JepaConfig,train_jepa
    if representation=="public_hash":
        from .jepa_replay import ReplayConfig,load_splits
        splits=load_splits(source,ReplayConfig(),mode="study",arm="cold")
    else:
        from .interface_induction import load_interface_splits
        splits=load_interface_splits(source,mode="study",arm="cold").splits
    if len(splits.train.before)!=len(train) or len(splits.test.before)!=len(heldout):
        raise ValueError("inconsistent public JEPA and target-frame sample cohorts")
    config=JepaConfig(steps=steps,batch_size=batch_size,seed=seed)
    model,report=train_jepa(splits.train,config,device=device)
    model.eval()

    def model_latent(rows):
        with torch.inference_mode():
            x=torch.as_tensor(rows.before,device=device)
            a=torch.as_tensor(rows.actions,device=device)
            predicted=model.predict_tensor(model.encoder(x),a)
        return predicted.float().cpu().numpy().astype(np.float64)

    z_train=model_latent(splits.train)
    z_test=model_latent(splits.test)
    y=np.asarray([frame.project(r).values for r in train],dtype=np.float64)
    masks=np.asarray([frame.project(r).observed_mask for r in train],dtype=bool)
    X=np.c_[z_train,np.ones(len(z_train))]
    testX=np.c_[z_test,np.ones(len(z_test))]
    weight=np.zeros((X.shape[1],y.shape[1]),dtype=np.float64)
    regularizer=np.eye(X.shape[1],dtype=np.float64)
    regularizer[-1,-1]=1e-8
    for j in range(y.shape[1]):
        present=masks[:,j]
        if present.sum()<3:continue
        weight[:,j]=np.linalg.solve(
            X[present].T@X[present]+regularizer,
            X[present].T@y[present,j],
        )
    projected=testX@weight
    forecasts=[]
    for row in projected:
        vals=[]
        for v,coordinate in zip(row,frame.coordinates):
            if coordinate.kind is not TargetKind.NUMERIC_DELTA:
                vals.append(float(np.clip(v,0,1)))
            else:vals.append(float(v))
        forecasts.append(SharedForecast(
            tuple(vals),tuple(True for _ in vals),
            tuple(c.key for c in frame.coordinates),frame.fingerprint,
        ))
    return tuple(forecasts),{
        "steps":steps,"batch_size":batch_size,"seed":seed,"device":device,
        "train_loss_start":report.start_loss,"train_loss_end":report.end_loss,
    }


def run_shared_probe(
    source_root: Path,
    *,
    train_seeds: tuple[int,...],
    validation_seed: int,
    heldout_seed: int,
    output: Path,
    representation: str,
    with_jepa: bool=False,
    device: str="cpu",
    jepa_steps: int=700,
    batch_size: int=128,
    seed: int=0,
) -> dict[str,Any]:
    if representation not in ("public_hash","induced_interface"):
        raise ValueError("supported representations: public_hash or induced_interface")
    if not train_seeds or len(set((*train_seeds,validation_seed,heldout_seed)))!=len(train_seeds)+2:
        raise ValueError("train validation and heldout seeds must be disjoint")
    root=Path(source_root)
    def get(n:int):
        return load_public_outcome_episode(root/f"seed-{n}"/"cold")
    training=tuple(row for s in train_seeds for row in get(s))
    _validation=get(validation_seed)  # Require split existence; never fit from it.
    testing=get(heldout_seed)
    frame=SharedOutcomeFrame.fit(training)
    persistence=PersistencePort(frame)
    ridge=RawRidgePort.fit(frame,training)
    truth=tuple(frame.project(item) for item in testing)
    forecasts={
        "persistence":tuple(persistence.predict_public(row.before,row.transition.action) for row in testing),
        "raw_ridge":tuple(ridge.predict_public(row.before,row.transition.action) for row in testing),
    }
    training_meta=None
    if with_jepa:
        predictions,training_meta=_latent_forecasts(
            root,frame,training,testing,representation=representation,
            device=device,steps=jepa_steps,batch_size=batch_size,seed=seed,
        )
        forecasts["jepa"]=predictions
    scoring=SharedEvaluation.score(
        truth,forecasts,actions=tuple(row.transition.action.schema_id for row in testing),
    )
    report={
        "protocol":"universal_shared_outcome_v1",
        "representation":representation,
        "train_seeds":list(train_seeds),
        "fit_seed_ids":list(train_seeds),
        "validation_seed":validation_seed,
        "heldout_seed":heldout_seed,
        "heldout_rows":len(testing),
        "shared_frame_fingerprint":frame.fingerprint,
        "shared_coordinates":[asdict(c) for c in frame.coordinates],
        "scores":_scores_as_dict(scoring),
        "jepa_training":training_meta,
        "causal_interpretation":"observational prediction only",
    }
    out=Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+"\n")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description="Shared public outcome evaluation of ECSA backends")
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--representation",choices=("public_hash","induced_interface"),required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--with-jepa",action="store_true")
    p.add_argument("--steps",type=int,default=700)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    result=run_shared_probe(a.runs,train_seeds=(0,1,2),validation_seed=3,heldout_seed=4,
                            output=a.output,representation=a.representation,
                            with_jepa=a.with_jepa,device=a.device,jepa_steps=a.steps,
                            batch_size=a.batch_size,seed=a.seed)
    print(json.dumps({"output":str(a.output),"fingerprint":result["shared_frame_fingerprint"],
                     "scores":result["scores"]},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
