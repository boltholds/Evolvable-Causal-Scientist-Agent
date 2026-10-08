"""Posthoc, seed-clustered analysis for the opaque-mechanism ablation.

This module cannot influence the scientist or its interventions. By design it
only receives completed JSON files, including held-out outcomes.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .mechanism_shift import NewMechanism, ShiftArm


NOVEL = tuple(m.value for m in NewMechanism if m is not NewMechanism.UNCHANGED_CONTROL)
RANDOM_ARMS = (
    ShiftArm.RETRAIN.value,
    ShiftArm.PARAMETERS.value,
    ShiftArm.STRUCTURE.value,
    ShiftArm.MIXED.value,
    ShiftArm.GRID_GEOMETRY.value,
)


def _average(rows: list[dict], field: str) -> float:
    return float(np.mean([r[field] for r in rows]))


def analyze(
    reports: list[dict], *, draws: int = 30_000,
    random_seed: int = 2026,
) -> dict:
    if not reports:
        raise ValueError("completed report files required")
    if type(draws) is not int or draws < 1000:
        raise ValueError("at least 1000 bootstrap resamples required")
    rows = [dict(row) for file in reports for row in file["results"]]
    for row in rows:
        row.setdefault("basis_changes_accepted", 0)  # v1 pre-grid artifacts
    index = {}
    for row in rows:
        key=(row["mechanism"],row["seed"],row["arm"])
        if key in index:
            raise ValueError(f"duplicate result: {key}")
        if row["world_interventions"] <= 0 or not np.isfinite(row["final_nll"]):
            raise ValueError(f"invalid run: {key}")
        index[key]=row
    novel = [row for row in rows if row["mechanism"] in NOVEL]
    stationary = [row for row in rows if row["mechanism"] == NewMechanism.UNCHANGED_CONTROL.value]
    arms = sorted({row["arm"] for row in novel})
    seeds = sorted({row["seed"] for row in novel})
    for law in NOVEL:
        for seed in seeds:
            for arm in arms:
                if (law,seed,arm) not in index:
                    raise ValueError(f"unpaired sample: {(law,seed,arm)}")
    summary={}
    for arm in arms:
        arm_rows=[r for r in novel if r["arm"]==arm]
        summary[arm]={
            "runs":len(arm_rows),
            "mean_final_nll":_average(arm_rows,"final_nll"),
            "mean_nll_curve_auc":_average(arm_rows,"normalized_nll_auc"),
            "mean_brier":_average(arm_rows,"heldout_brier"),
            "mean_ece_top1":_average(arm_rows,"heldout_top_class_ece"),
            "mean_prediction_80_coverage":_average(arm_rows,"observed_80_coverage"),
            "mean_training_steps":_average(arm_rows,"gradient_steps"),
            "mean_parameter_steps":_average(arm_rows,"parameter_steps"),
            "mean_accepted_structural_mutations":_average(arm_rows,"topology_changes_accepted"),
            "mean_accepted_grid_mutations":_average(arm_rows,"basis_changes_accepted"),
            "mean_eig_choices":_average(arm_rows,"eig_choices"),
        }
    by_mechanism={}
    for law in NOVEL:
        by_mechanism[law]={}
        for arm in arms:
            subset=[r for r in novel if r["arm"]==arm and r["mechanism"]==law]
            by_mechanism[law][arm]={
                "n":len(subset),"mean_final_nll":_average(subset,"final_nll"),
                "mean_nll_auc":_average(subset,"normalized_nll_auc"),
                "mean_prequential_nll":_average(subset,"prequential_nll"),
                "mean_heldout_outside_grid":_average(subset,"heldout_outside_grid"),
            }
    rng=np.random.default_rng(random_seed)
    def paired(left: str,right: str) -> dict:
        deltas=np.asarray([
            np.mean([index[(law,s,left)]["final_nll"]-index[(law,s,right)]["final_nll"]
                     for law in NOVEL])
            for s in seeds
        ])
        resampled=np.mean(rng.choice(deltas,size=(draws,len(seeds)),replace=True),axis=1)
        return {
            "contrast":f"{left} minus {right}","n_seed_clusters":len(seeds),
            "mean_delta_nll":float(deltas.mean()),
            "ci95_percentile":list(map(float,np.quantile(resampled,[.025,.975]))),
            "seed_clusters_better":int((deltas<0).sum()),
            "interpretation":"exploratory: synthetic mechanisms, non-pre-registered multiples",
        }
    comparisons=[paired(arm,ShiftArm.RETRAIN.value) for arm in arms if arm!=ShiftArm.RETRAIN.value]
    if ShiftArm.STRUCTURE_ADAPTIVE_EIG.value in arms:
        comparisons.append(paired(ShiftArm.STRUCTURE_ADAPTIVE_EIG.value,ShiftArm.STRUCTURE.value))
    stationary_summary={}
    for arm in sorted({r["arm"] for r in stationary}):
        subset=[r for r in stationary if r["arm"]==arm]
        stationary_summary[arm]={"n":len(subset),"mean_final_nll":_average(subset,"final_nll"),
                                 "mean_auc":_average(subset,"normalized_nll_auc")}
    return {
        "novel_mechanisms":list(NOVEL),
        "seed_clusters":seeds,
        "novel_runs":len(novel),"stationary_control_runs":len(stationary),
        "summary":summary,"by_mechanism":by_mechanism,
        "paired_bootstrap":comparisons,"stationary_control":stationary_summary,
        "notes":[
            "Arms have identical warmup, random-pool X and do(X) count; random arms have exactly identical intervention outcomes.",
            "All arms spend equal optimizer steps on candidate offspring; changes in width or knot count change number of parameter updates and FLOPs.",
            "Grid-geometry mutants retain architecture shape and parameter count per parent but change spline knot locations.",
            "Hidden laws are absent from warmup DATA; a sufficiently expressive initial KAN may already represent them.",
            "Heldout Y and exact mechanism are never supplied to run_shift; recovery times are evaluation-only.",
            "Bayesian posteriors are valid only within each frozen generation; boundary pseudo-priors reuse scarce calibration points.",
            "Correlated tasks are clustered by seed for exploratory percentile bootstrap; no causality or general-domain benefit established.",
        ],
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("reports",nargs="+",help="JSON reports with results arrays")
    parser.add_argument("--output",default="")
    args=parser.parse_args()
    payload=analyze([json.loads(Path(p).read_text()) for p in args.reports])
    if args.output:
        path=Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "novel_runs":payload["novel_runs"],
        "stationary_runs":payload["stationary_control_runs"],
        "summary":payload["summary"],
        "paired_bootstrap":payload["paired_bootstrap"],
    },indent=2,sort_keys=True))

if __name__=="__main__":
    main()
