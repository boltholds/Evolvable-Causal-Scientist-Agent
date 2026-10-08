from pathlib import Path

from ecsa.benchmarks.discoveryworld.arena_b import ArenaBResult
from ecsa.benchmarks.discoveryworld import lifted_ablation


def test_lifted_ablation_uses_identical_seeds_steps_budget(monkeypatch, tmp_path: Path):
    called = []

    def fake_episode(**kwargs):
        called.append(kwargs)
        return ArenaBResult(
            steps=kwargs["max_steps"],
            transitions=kwargs["max_steps"],
            grounding_evidence_count=1,
            contract_count=1,
            successes=int(kwargs["use_applicability_selection"])
                + int(kwargs["use_lifted_selection"] and kwargs["use_applicability_selection"]),
        )

    monkeypatch.setattr(lifted_ablation, "run_autonomous_episode", fake_episode)
    results = lifted_ablation.run_lifted_arena_ablation(
        seeds=(0, 11), max_steps=12, max_ground_actions=32, output_dir=tmp_path,
    )
    assert len(results) == 2
    assert [(r.grounded_success_delta, r.lifted_success_delta) for r in results] == [
        (1, 1), (1, 1),
    ]
    assert [entry["seed"] for entry in called] == [0, 0, 0, 11, 11, 11]
    assert all(entry["max_steps"] == 12 for entry in called)
    assert all(entry["max_ground_actions"] == 32 for entry in called)
    assert [
        (c["use_applicability_selection"], c["use_lifted_selection"])
        for c in called
    ] == [
        (False, False), (True, False), (True, True),
        (False, False), (True, False), (True, True),
    ]
