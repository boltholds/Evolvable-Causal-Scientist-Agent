"""Run a frozen pretrained world-model -> KAN representation comparison.

This is a two-part study:
  1) identical KAN heads on latent states extracted from a frozen causal LM;
  2) the same LM's teacher-forced next-observation NLL (independent of KAN).

Real BehR weights are optional and require hardware. A tiny Qwen2.5 checkpoint
is provided only to smoke-test the *real pretrained inference interface*.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .token_state_relations import (
    DEFAULT_ARMS,
    StateArm,
    TokenRelationConfig,
    benchmark_one,
)
from .verbalization_study import StudyLaw, _simulation, _negative_law_outcomes
from .world_model_encoders import (
    Candidate, CausalLMConfig, FrozenCausalLMTokenPort,
    SPECS, evaluate_next_state_likelihood,
)


def _same_heldout(law: StudyLaw, seed: int, heldout: int):
    # Same fixture split as token_state_relations.benchmark_one.
    rows=_simulation(law,seed*101+109,heldout,heldout=True)
    negative=_negative_law_outcomes(rows,law)
    return rows,negative


def run_study(
    port: FrozenCausalLMTokenPort, *,
    config: TokenRelationConfig,
    laws: tuple[StudyLaw, ...],
    seeds: tuple[int, ...],
    arms: tuple[StateArm, ...],
    include_latents: bool = True,
    include_likelihood: bool = False,
    max_likelihood_pairs: int = 8,
) -> dict[str, object]:
    if not laws or not seeds or not arms or len(set(arms))!=len(arms):
        raise ValueError("nonempty laws/seeds and unique arms required")
    if not all(isinstance(x,StudyLaw) for x in laws):
        raise TypeError("StudyLaw required")
    if not all(isinstance(x,StateArm) for x in arms):
        raise TypeError("StateArm required")
    if not include_latents and not include_likelihood:
        raise ValueError("enable latent KAN or generative likelihood")
    if type(max_likelihood_pairs) is not int or max_likelihood_pairs < 1:
        raise ValueError("positive likelihood sample cap required")
    results=[]
    generative=[]
    for law in laws:
        for seed in seeds:
            if include_latents:
                results.extend(asdict(row) for row in benchmark_one(
                    law,seed,config,port,
                    model_id=port.model_id,revision=port.revision,arms=arms,
                ))
            if include_likelihood:
                observed, wrong=_same_heldout(law,seed,config.heldout)
                generative.append(asdict(evaluate_next_state_likelihood(
                    port,observed,wrong,law=law.value,seed=seed,
                    maximum=max_likelihood_pairs,
                )))
    summary={}
    for law in laws:
        summary[law.value]={}
        for arm in arms:
            scores=[float(row["auroc"]) for row in results
                    if row["law"]==law.value and row["arm"]==arm.value
                    and row["domain"]=="counterfactual"]
            if scores:
                summary[law.value][arm.value]=float(np.mean(scores))
    return {
        "candidate":port.config.candidate.value,
        "model_id":port.model_id,
        "revision":port.revision,
        "model_source":port.spec.source if port.model_id==port.spec.model_id else "custom override",
        "model_architecture":port.spec.kind,
        "weights_loaded_by_port":port.loaded_checkpoint_by_port,
        "is_actual_behR":(port.loaded_checkpoint_by_port and
            port.model_id==SPECS[Candidate.BEHR_TEXTWORLD].model_id),
        "config":asdict(config),
        "arms":[x.value for x in arms],
        "laws":[x.value for x in laws],
        "seeds":list(seeds),
        "scores":results,
        "generative_likelihood":generative,
        "mean_counterfactual_auroc":summary,
        "max_observed_tokens":port.max_observed_length,
        "audited_sequences":port.verified_length_count,
        "backbone_forward_batches":port.backbone_forward_batches,
        "limitations":[
            "BehR is a generative causal LM, not a pretrained symmetric embedding model.",
            "Output-layer features and next-state NLL measure distinct capabilities.",
            "Synthetic heldout counterfactuals are generated only for evaluation.",
            "A counterfactual Y is not assumed impossible when there are multiple valid outcomes.",
            "Plain canonical JSON is OOD relative to BehR's TextWorld interaction-history training.",
            "The numeric baseline is a tanh-compressed hash, not an exact numeric oracle.",
            "The KAN receives no hidden-law supervision; all X/Y encoding weights are frozen.",
            "All train/test IDs are disjoint; calibration is train-only.",
            "AUROC on pair discrimination is NOT calibrated p(Y|do(X)).",
            "Cross-backbone parameter counts and compute are NOT equal; report costs separately.",
        ],
    }


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",choices=[x.value for x in Candidate],
                   default=Candidate.BEHR_TEXTWORLD.value)
    p.add_argument("--model-id",default=None,help="override only for explicit control or smoke; reported")
    p.add_argument("--revision",default=None)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--4bit",action="store_true",dest="quantized")
    p.add_argument("--allow-large-cpu",action="store_true")
    p.add_argument("--max-length",type=int,default=512)
    p.add_argument("--laws",nargs="+",choices=[x.value for x in StudyLaw],
                   default=["precision","operator"])
    p.add_argument("--seeds",nargs="+",type=int,default=[0])
    p.add_argument("--arms",nargs="+",choices=[x.value for x in StateArm],
                   default=["sentence_64","sentence_full","token_mean",
                            "numeric_control","gated_attention","gated_kan_update"])
    p.add_argument("--bootstrap",type=int,default=16)
    p.add_argument("--adaptation",type=int,default=8)
    p.add_argument("--calibration",type=int,default=8)
    p.add_argument("--heldout",type=int,default=12)
    p.add_argument("--warmup-steps",type=int,default=30)
    p.add_argument("--extra-kan-steps",type=int,default=12)
    p.add_argument("--checkpoint-every",type=int,default=6)
    p.add_argument("--kan-only",action="store_true",help="no generative likelihood")
    p.add_argument("--likelihood-only",action="store_true",help="skip latent KAN fitting")
    p.add_argument("--likelihood-pairs",type=int,default=8)
    p.add_argument("--list-candidates",action="store_true",help="no model download")
    p.add_argument("--output",default="")
    args=p.parse_args()
    if args.list_candidates:
        print(json.dumps([asdict(x) for x in SPECS.values()],indent=2))
        return
    if args.kan_only and args.likelihood_only:
        p.error("--kan-only and --likelihood-only are mutually exclusive")
    candidate=Candidate(args.candidate)
    if candidate is Candidate.AGENTWORLD:
        p.error("AgentWorld is a generative simulator, not a Qwen2 hidden-state encoder; use its official vLLM endpoint for a separate simulation evaluation")
    c=CausalLMConfig(
        candidate=candidate,model_id=args.model_id,revision=args.revision,
        device=args.device,max_length=args.max_length,quantized_4bit=args.quantized,
        allow_large_cpu=args.allow_large_cpu,
    )
    port=FrozenCausalLMTokenPort(c)
    config=TokenRelationConfig(
        bootstrap=args.bootstrap,adaptation=args.adaptation,
        calibration=args.calibration,heldout=args.heldout,
        warmup_steps=args.warmup_steps,extra_kan_steps=args.extra_kan_steps,
        checkpoint_every=args.checkpoint_every,
    )
    results=run_study(
        port,config=config,
        laws=tuple(StudyLaw(x) for x in args.laws),
        seeds=tuple(args.seeds),
        arms=tuple(StateArm(x) for x in args.arms),
        include_latents=not args.likelihood_only,
        include_likelihood=not args.kan_only,
        max_likelihood_pairs=args.likelihood_pairs,
    )
    body=json.dumps(results,ensure_ascii=False,indent=2)
    if args.output:
        path=Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(body+"\n",encoding="utf8")
    print(json.dumps({
        "model_id":results["model_id"],
        "revision":results["revision"],
        "is_actual_behR":results["is_actual_behR"],
        "mean_counterfactual_auroc":results["mean_counterfactual_auroc"],
        "generative_likelihood":results["generative_likelihood"],
        "backbone_forward_batches":results["backbone_forward_batches"],
    },indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
