"""Three-arm, matched-seed Arena B ablation.

All arms share the same environment seed and action-generation budget. The
only difference is which applicability selector is enabled; science contracts,
grounding and optional dependencies are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .arena_b import ArenaBResult, run_autonomous_episode


@dataclass(frozen=True)
class LiftedArenaAblationResult:
    seed: int
    structural: ArenaBResult
    grounded: ArenaBResult
    lifted: ArenaBResult
    grounded_success_delta: int
    lifted_success_delta: int


def run_lifted_arena_ablation(
    *,
    seeds: tuple[int, ...],
    max_steps: int,
    max_ground_actions: int,
    output_dir: Path,
) -> tuple[LiftedArenaAblationResult, ...]:
    if (
        not isinstance(seeds, tuple)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
    ):
        raise ValueError("nonempty tuple of seed integers required")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("positive step budget required")
    if type(max_ground_actions) is not int or max_ground_actions < 1:
        raise ValueError("positive candidate budget required")

    results: list[LiftedArenaAblationResult] = []
    for seed in seeds:
        def episode(label: str, *, grounded: bool, lifted: bool) -> ArenaBResult:
            return run_autonomous_episode(
                scenario="Reactor Lab",
                difficulty="Normal",
                seed=seed,
                max_steps=max_steps,
                max_ground_actions=max_ground_actions,
                output_dir=Path(output_dir) / str(seed) / label,
                use_affordance_scoring=True,
                use_applicability_selection=grounded,
                use_lifted_selection=lifted,
            )

        structural = episode("structural", grounded=False, lifted=False)
        grounded = episode("grounded", grounded=True, lifted=False)
        lifted = episode("lifted", grounded=True, lifted=True)
        results.append(LiftedArenaAblationResult(
            seed=seed,
            structural=structural,
            grounded=grounded,
            lifted=lifted,
            grounded_success_delta=grounded.successes - structural.successes,
            lifted_success_delta=lifted.successes - grounded.successes,
        ))
    return tuple(results)
