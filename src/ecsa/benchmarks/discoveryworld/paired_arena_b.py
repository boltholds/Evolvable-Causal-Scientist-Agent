from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .arena_b import ArenaBResult, run_autonomous_episode


@dataclass(frozen=True)
class PairedArenaBResult:
    seed: int
    baseline: ArenaBResult
    affordance: ArenaBResult
    success_delta: int


def run_paired_arena_b(
    *,
    seeds: tuple[int, ...],
    max_steps: int,
    max_ground_actions: int,
    output_dir: Path,
) -> tuple[PairedArenaBResult, ...]:
    """Run matched Arena B episodes with applicability EIG switched off/on.

    Fixed seeds and budgets permit within-seed comparisons, but not claims
    of improved generalization without held-out episodes.
    """
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("seeds must be a nonempty tuple of integers")
    root = Path(output_dir)
    results = []
    for seed in seeds:
        baseline = run_autonomous_episode(
            scenario="Reactor Lab",
            difficulty="Normal",
            seed=seed,
            max_steps=max_steps,
            max_ground_actions=max_ground_actions,
            output_dir=root / str(seed) / "baseline",
            use_affordance_scoring=True,
            use_applicability_selection=False,
        )
        affordance = run_autonomous_episode(
            scenario="Reactor Lab",
            difficulty="Normal",
            seed=seed,
            max_steps=max_steps,
            max_ground_actions=max_ground_actions,
            output_dir=root / str(seed) / "affordance",
            use_affordance_scoring=True,
            use_applicability_selection=True,
        )
        results.append(PairedArenaBResult(
            seed=seed, baseline=baseline, affordance=affordance,
            success_delta=affordance.successes - baseline.successes,
        ))
    return tuple(results)
