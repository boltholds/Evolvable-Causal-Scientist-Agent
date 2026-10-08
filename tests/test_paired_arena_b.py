from pathlib import Path

from ecsa.benchmarks.discoveryworld.arena_b import ArenaBResult
from ecsa.benchmarks.discoveryworld import paired_arena_b


def test_paired_arena_holds_seed_and_budget_fixed(monkeypatch, tmp_path: Path):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return ArenaBResult(
            steps=kwargs["max_steps"],
            transitions=kwargs["max_steps"],
            grounding_evidence_count=1,
            contract_count=1,
            successes=2 if kwargs["use_applicability_selection"] else 1,
        )

    monkeypatch.setattr(paired_arena_b, "run_autonomous_episode", fake_run)
    results = paired_arena_b.run_paired_arena_b(
        seeds=(0, 1), max_steps=12, max_ground_actions=64, output_dir=tmp_path,
    )
    assert len(calls) == 4
    assert all(call["max_steps"] == 12 for call in calls)
    assert all(call["max_ground_actions"] == 64 for call in calls)
    assert [call["seed"] for call in calls] == [0, 0, 1, 1]
    assert all(call["use_affordance_scoring"] for call in calls)
    assert [call["use_applicability_selection"] for call in calls] == [
        False, True, False, True,
    ]
    assert [result.success_delta for result in results] == [1, 1]
