"""Paired ECSA learning-progress experiment: pre-registered metrics, no oracle feedback."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import pytest

from ecsa.benchmarks.discoveryworld.arena_b import ArenaBResult, run_autonomous_episode
from ecsa.benchmarks.discoveryworld.paired_progress_arena_b import (
    run_paired_progress_arena,
    PairedProgressSeed,
)
from ecsa.benchmarks.discoveryworld.progress_metrics import (
    ProgressActionEvent, ProgressEpisodeMetrics, summarize_progress_actions,
)


def event(n, schema, success, reduction, args=()):
    return ProgressActionEvent(
        transition_id=f"transition:{n}",
        schema_id=schema,
        arguments=tuple(args),
        success=success,
        prior_schema_variance_reduction=reduction,
    )


def test_repeated_low_gain_failed_actions_count_without_action_blacklist():
    events=(
        event(0,"Z",False,.012),event(1,"Z",False,.008),
        event(2,"Z",False,.001),event(3,"Z",False,.0003),
        event(4,"W",True,.0001),
    )
    report=summarize_progress_actions(events,min_gain=0.002,min_repeats=2)
    assert report.total_actions==5
    assert report.action_counts==(("W",1),("Z",4))
    assert report.longest_exact_action_streak==4
    assert report.low_information_failed_repeats==2
    assert report.distinct_schemas==2
    assert report.max_schema_fraction==.8
    assert report.schema_entropy_bits>0


def test_retrying_successful_scientific_action_is_not_classified_sterile():
    events=tuple(event(i,"arbitrary",True,0.0) for i in range(5))
    report=summarize_progress_actions(events)
    assert report.low_information_failed_repeats==0
    assert report.longest_exact_action_streak==5


def test_action_identity_includes_arguments_and_not_just_schema():
    events=(
        event(0,"operation",False,0.0,("first",)),
        event(1,"operation",False,0.0,("second",)),
        event(2,"operation",False,0.0,("first",)),
        event(3,"operation",False,0.0,("second",)),
    )
    report=summarize_progress_actions(events,min_gain=.002,min_repeats=2)
    assert report.longest_exact_action_streak==1
    assert report.low_information_failed_repeats==0


def test_duplicate_provenance_and_invalid_gain_rejected():
    with pytest.raises(ValueError,match="duplicate"):
        summarize_progress_actions((event(0,"A",True,.0),event(0,"B",False,.0)))
    with pytest.raises(ValueError,match="finite"):
        ProgressActionEvent("id","A",(),False,float("nan"))


def test_paired_runner_uses_same_seed_budget_and_baseline_flags(tmp_path):
    received=[]
    def fake_episode(**kwargs):
        received.append(kwargs)
        progress=kwargs["use_progress_scoring"]
        return ArenaBResult(
            steps=5,transitions=5,grounding_evidence_count=10,contract_count=0,
            successes=1 if not progress else 2,
            official_score_normalized=.2 if not progress else .3,
            official_completed_successfully=False,
            progress_metrics=ProgressEpisodeMetrics(
                total_actions=5,distinct_schemas=1 if not progress else 2,
                action_counts=(("Z",5),) if not progress else (("X",2),("Z",3)),
                longest_exact_action_streak=5 if not progress else 3,
                low_information_failed_repeats=3 if not progress else 1,
                max_schema_fraction=1.0 if not progress else .6,
                schema_entropy_bits=0.0 if not progress else .97,
            ),
        )
    result=run_paired_progress_arena(
        seeds=(0,1),max_steps=5,max_ground_actions=16,
        output_dir=tmp_path/"study",episode_runner=fake_episode,
    )
    assert len(result.seeds)==2
    assert all(isinstance(row,PairedProgressSeed) for row in result.seeds)
    assert len(received)==4
    assert [item["seed"] for item in received]==[0,0,1,1]
    for a,b in ((received[0],received[1]),(received[2],received[3])):
        assert a["max_steps"]==b["max_steps"]==5
        assert a["max_ground_actions"]==b["max_ground_actions"]==16
        assert a["use_progress_scoring"] is False
        assert b["use_progress_scoring"] is True
        assert a["use_applicability_selection"]==b["use_applicability_selection"]==True
        assert a["use_lifted_selection"]==b["use_lifted_selection"]==True
        assert a["output_dir"]!=b["output_dir"]
        assert a["include_official_evaluation"] and b["include_official_evaluation"]
    assert result.mean_score_delta==pytest.approx(.1)
    assert result.mean_low_information_failed_repeat_delta==-2
    assert result.confirmed_causal_hypothesis_delta is None
    assert json.loads((tmp_path/"study"/"paired_summary.json").read_text())["protocol"]=="ecsa_progress_paired_v1"


def test_existing_output_directory_does_not_silently_overwrite(tmp_path):
    root=tmp_path/"already"
    root.mkdir()
    (root/"existing.txt").write_text("must keep")
    with pytest.raises(FileExistsError):
        run_paired_progress_arena(seeds=(0,),max_steps=5,max_ground_actions=16,
                                  output_dir=root,episode_runner=lambda **kw: None)
    assert (root/"existing.txt").read_text()=="must keep"


def test_unobserved_independent_causal_confirmation_is_null(tmp_path):
    def fake_episode(**kwargs):
        return ArenaBResult(steps=1,transitions=1,grounding_evidence_count=1,
                            contract_count=0,official_score_normalized=0.0,
                            official_completed_successfully=False)
    result=run_paired_progress_arena(
        seeds=(0,),max_steps=1,max_ground_actions=2,
        output_dir=tmp_path/"pair",episode_runner=fake_episode)
    assert result.seeds[0].baseline.independently_confirmed_causal_hypotheses is None
    assert result.seeds[0].progress.independently_confirmed_causal_hypotheses is None
    assert result.confirmed_causal_hypothesis_delta is None


def test_live_progress_episode_has_post_run_official_score_and_telemetry(tmp_path):
    result=run_autonomous_episode(
        scenario="Reactor Lab",difficulty="Normal",seed=0,
        max_steps=3,max_ground_actions=16,use_progress_scoring=True,
        include_official_evaluation=True,output_dir=tmp_path/"live",
    )
    assert result.steps==result.transitions==3
    assert result.progress_metrics is not None
    assert result.progress_metrics.total_actions==3
    assert result.official_score_normalized is not None
    assert 0<=result.official_score_normalized<=1
    assert result.independently_confirmed_causal_hypotheses is None
    assert (tmp_path/"live"/"final_evaluation.json").exists()
    records=[json.loads(line) for line in (tmp_path/"live"/"raw_transitions.jsonl").read_text().splitlines()]
    assert len(records)==3
    assert all("prequential_variance_reduction" in row for row in records)
    assert all("official_score" not in row for row in records)
