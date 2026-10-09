"""Benchmark adapter can load public logs but never inspect oracles."""
import inspect
from pathlib import Path
import json

import pytest

from ecsa.benchmarks.discoveryworld.shared_probe import (
    load_public_outcome_episode, run_shared_probe,
)


def _episode(root,seed,n=8):
    p=root/f"seed-{seed}"/"cold"
    p.mkdir(parents=True)
    (p/"run.json").write_text(json.dumps({
        "scenario":"Reactor Lab","difficulty":"Normal","seed":seed,
    }))
    with (p/"actions.jsonl").open("w") as actions, \
         (p/"action_outcomes.jsonl").open("w") as outcomes, \
         (p/"observations.jsonl").open("w") as obs:
        for i in range(n):
            before={"ui":{"accessibleEnvironmentObjects":[
                {"uuid":i+seed*100,"name":"item","charge":i}],
                "lastActionMessage": "checked"}}
            after={"ui":{"accessibleEnvironmentObjects":[
                {"uuid":i+seed*100,"name":"item","charge":i+1}],
                "lastActionMessage":"checked"}}
            obs.write(json.dumps({"step":i,"phase":"pre","observation":before})+"\n")
            obs.write(json.dumps({"step":i+1,"phase":"post","observation":after})+"\n")
            actions.write(json.dumps({"step":i,"action":{"action":"UNKNOWN","arg1":i+seed*100}})+"\n")
            outcomes.write(json.dumps({"step":i+1,"success":True,"errors":[]})+"\n")
    return p


def test_discoveryworld_adapter_has_no_oracle_input(tmp_path):
    p=_episode(tmp_path,0)
    (p/"final_scorecard.json").write_text("SECRET INVALID JSON")
    rows=load_public_outcome_episode(p)
    assert len(rows)==8
    assert all(row.transition.outcome.success for row in rows)
    source=inspect.getsource(load_public_outcome_episode)
    assert "getTaskScorecard" not in source
    assert "final_scorecard.json" not in source


def test_hash_and_learned_interface_use_same_frame_fingerprint(tmp_path):
    for seed in range(5):_episode(tmp_path,seed)
    a=run_shared_probe(tmp_path,train_seeds=(0,1,2),validation_seed=3,
                       heldout_seed=4,output=tmp_path/"hash.json",representation="public_hash")
    b=run_shared_probe(tmp_path,train_seeds=(0,1,2),validation_seed=3,
                       heldout_seed=4,output=tmp_path/"learned.json",representation="induced_interface")
    assert a["shared_frame_fingerprint"]==b["shared_frame_fingerprint"]
    assert a["heldout_rows"]==b["heldout_rows"]==8


def test_seed4_not_used_for_vocabulary_or_readout_fit(tmp_path):
    for seed in range(5):_episode(tmp_path,seed)
    result=run_shared_probe(tmp_path,train_seeds=(0,1,2),validation_seed=3,
                            heldout_seed=4,output=tmp_path/"output.json",representation="public_hash")
    assert result["train_seeds"]==[0,1,2]
    assert result["validation_seed"]==3
    assert result["heldout_seed"]==4
    assert "4" not in result.get("fit_seed_ids",[])
    assert Path(tmp_path/"output.json").exists()


def test_baselines_receive_identical_action_and_sample_budget(tmp_path):
    for seed in range(5):_episode(tmp_path,seed)
    result=run_shared_probe(tmp_path,train_seeds=(0,1,2),validation_seed=3,
                       heldout_seed=4,output=tmp_path/"common.json",representation="public_hash")
    assert result["scores"]["persistence"]["row_count"]==8
    assert result["scores"]["raw_ridge"]["row_count"]==8
    assert result["scores"]["persistence"]["coverage"]==1
    assert result["scores"]["raw_ridge"]["coverage"]==1
