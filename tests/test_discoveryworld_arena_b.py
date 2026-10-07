import inspect
import json
from pathlib import Path

from ecsa.benchmarks.discoveryworld.arena_b import (
    run_autonomous_episode,
)


def test_arena_b_real_seed_runs_raw_transition_loop(
    tmp_path: Path,
) -> None:
    result = run_autonomous_episode(
        scenario="Reactor Lab",
        difficulty="Normal",
        seed=0,
        max_steps=3,
        output_dir=tmp_path / "arena-b",
        max_ground_actions=16,
    )

    assert 1 <= result.steps <= 3
    assert result.transitions == result.steps
    assert result.grounding_evidence_count >= result.steps
    assert (
        tmp_path / "arena-b" / "raw_transitions.jsonl"
    ).exists()


def test_arena_b_production_path_has_no_structured_transfer_semantics() -> None:
    import ecsa.benchmarks.discoveryworld.arena_b as arena_b
    import ecsa.benchmarks.discoveryworld.perception as perception
    import ecsa.benchmarks.discoveryworld.raw_environment as raw_environment

    source = "\n".join(
        (
            inspect.getsource(arena_b),
            inspect.getsource(perception),
            inspect.getsource(raw_environment),
        )
    )
    forbidden = (
        "_ROLE_MAP",
        "PROBE_BINARY",
        "OBSERVE_UNARY",
        "ACQUIRE",
        "PLACE",
        "MeasurementKind",
        "ReactorMeasurement",
        "ReactorMechanismHypothesis",
        "ReactorLabScientificSidecar",
    )

    assert not any(token in source for token in forbidden)



def test_arena_b_runs_unfamiliar_scenario_and_writes_contract_snapshot(
    tmp_path: Path,
) -> None:
    output = tmp_path / "archaeology"
    result = run_autonomous_episode(
        scenario="Archaeology Dating",
        difficulty="Normal",
        seed=0,
        max_steps=4,
        output_dir=output,
        max_ground_actions=24,
    )

    assert result.transitions == result.steps
    assert result.steps == 4
    assert result.contract_count > 0
    snapshot = json.loads(
        (output / "world_contracts.json").read_text()
    )
    assert snapshot["scenario"] == "Archaeology Dating"
    assert snapshot["difficulty"] == "Normal"
    assert snapshot["seed"] == 0
    assert snapshot["contracts"]
    assert all(
        contract["canonical_fingerprint"]
        for contract in snapshot["contracts"]
    )
