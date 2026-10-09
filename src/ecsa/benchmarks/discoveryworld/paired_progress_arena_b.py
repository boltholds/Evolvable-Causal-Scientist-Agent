"""Same-scientist paired test of learning-progress selection in DiscoveryWorld.

No modified reward, no scenario-specific action blacklist, no transfer memory,
and no evaluator feedback before episode termination. This experiment compares
only one explicit policy flag using fresh worlds with paired seeds.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Callable

from .arena_b import ArenaBResult, run_autonomous_episode


@dataclass(frozen=True)
class PairedProgressSeed:
    seed: int
    baseline: ArenaBResult
    progress: ArenaBResult
    score_delta: float | None
    low_information_failed_repeat_delta: int | None
    successful_action_delta: int


@dataclass(frozen=True)
class PairedProgressStudy:
    protocol: str
    scenario: str
    difficulty: str
    seeds: tuple[PairedProgressSeed, ...]
    max_steps: int
    max_ground_actions: int
    mean_score_delta: float | None
    mean_low_information_failed_repeat_delta: float | None
    confirmed_causal_hypothesis_delta: int | None
    independent_confirmation_status: str


def _mean_available(values: tuple[float | int | None, ...]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return fmean(float(value) for value in values if value is not None)


def run_paired_progress_arena(
    *,
    seeds: tuple[int, ...],
    max_steps: int,
    max_ground_actions: int,
    output_dir: Path,
    episode_runner: Callable[..., ArenaBResult] = run_autonomous_episode,
) -> PairedProgressStudy:
    if not isinstance(seeds, tuple) or not seeds or any(
        type(seed) is not int or seed < 0 or seed > 4 for seed in seeds
    ) or len(set(seeds)) != len(seeds):
        raise ValueError("distinct official Reactor Lab seed indices 0..4 required")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("positive action budget required")
    if type(max_ground_actions) is not int or max_ground_actions < 1:
        raise ValueError("positive candidate budget required")

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"results directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    rows: list[PairedProgressSeed] = []
    for seed in seeds:
        outputs = {}
        for name, progress_enabled in (("baseline", False), ("progress", True)):
            arm_dir = root / f"seed-{seed}" / name
            # Both arms have identical exploration and applicability
            # configuration. Per-episode world model and learning history
            # are instantiated within the episode runner, never shared.
            outputs[name] = episode_runner(
                scenario="Reactor Lab",
                difficulty="Normal",
                seed=seed,
                max_steps=max_steps,
                max_ground_actions=max_ground_actions,
                output_dir=arm_dir,
                use_affordance_scoring=True,
                use_applicability_selection=True,
                use_lifted_selection=True,
                use_progress_scoring=progress_enabled,
                include_official_evaluation=True,
            )

        baseline, progress = outputs["baseline"], outputs["progress"]
        if not isinstance(baseline, ArenaBResult) or not isinstance(progress, ArenaBResult):
            raise TypeError("paired episode runner must return ArenaBResult")

        b_metrics = baseline.progress_metrics
        p_metrics = progress.progress_metrics
        score_delta = (
            progress.official_score_normalized - baseline.official_score_normalized
            if progress.official_score_normalized is not None
            and baseline.official_score_normalized is not None else None
        )
        retry_delta = (
            p_metrics.low_information_failed_repeats - b_metrics.low_information_failed_repeats
            if p_metrics is not None and b_metrics is not None else None
        )
        rows.append(PairedProgressSeed(
            seed=seed, baseline=baseline, progress=progress,
            score_delta=score_delta,
            low_information_failed_repeat_delta=retry_delta,
            successful_action_delta=progress.successes - baseline.successes,
        ))

    confirmations = tuple(
        (row.progress.independently_confirmed_causal_hypotheses,
         row.baseline.independently_confirmed_causal_hypotheses)
        for row in rows
    )
    if any(p is None or b is None for p, b in confirmations):
        confirmed_delta = None
        confirmation_status = (
            "not measured: no prospective independent intervention "
            "confirmation protocol is active in Arena B"
        )
    else:
        confirmed_delta = sum(
            int(p) - int(b) for p, b in confirmations if p is not None and b is not None
        )
        confirmation_status = "measured prospective independent interventions"

    report = PairedProgressStudy(
        protocol="ecsa_progress_paired_v1",
        scenario="Reactor Lab",
        difficulty="Normal",
        seeds=tuple(rows),
        max_steps=max_steps,
        max_ground_actions=max_ground_actions,
        mean_score_delta=_mean_available(tuple(row.score_delta for row in rows)),
        mean_low_information_failed_repeat_delta=_mean_available(
            tuple(row.low_information_failed_repeat_delta for row in rows)
        ),
        confirmed_causal_hypothesis_delta=confirmed_delta,
        independent_confirmation_status=confirmation_status,
    )
    (root / "paired_summary.json").write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Paired old vs progress ECSA scientist on public DiscoveryWorld"
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-ground-actions", type=int, default=64)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    seeds = tuple(int(item) for item in args.seeds.split(","))
    result = run_paired_progress_arena(
        seeds=seeds, max_steps=args.max_steps,
        max_ground_actions=args.max_ground_actions,
        output_dir=args.output,
    )
    print(json.dumps({
        "protocol": result.protocol,
        "seeds": [row.seed for row in result.seeds],
        "mean_score_delta": result.mean_score_delta,
        "mean_low_information_failed_repeat_delta":
            result.mean_low_information_failed_repeat_delta,
        "independent_confirmation_status":
            result.independent_confirmation_status,
        "summary_file": str(args.output / "paired_summary.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
