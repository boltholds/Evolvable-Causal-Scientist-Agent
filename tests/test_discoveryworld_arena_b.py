import inspect
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
