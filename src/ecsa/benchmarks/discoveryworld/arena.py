from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from collections.abc import Callable

from ecsa.adapters.mlmd import MLMDMechanismRepository
from ecsa.mechanisms import MechanismRecord, MechanismRepository

from .contracts import (
    ArenaArm,
    ArenaEpisodeResult,
    ArmEpisodeResult,
    DiscoveryWorldActionPolicy,
    DiscoveryWorldEpisodeConfig,
    PairedSeedResult,
    PolicyDecision,
    ReactorMechanismHypothesis,
    ScientificContext,
    TransferArenaResult,
)
from .environment import (
    DISCOVERYWORLD_REVISION,
    DiscoveryWorldEnvironmentAdapter,
)
from .reactor_lab import (
    ReactorLabScientificSidecar,
    ReactorRuleArtifactStore,
    build_admitted_reactor_mechanism,
    measurement_action_key,
    reactor_context,
)
from .runlog import ArenaRunWriter
from .metrics import ReactorRunMetricsAccumulator, summarize_transfer


def _ecsa_revision() -> str:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    revision = process.stdout.strip()
    return revision if len(revision) == 40 else "unknown"


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_episode(
    *,
    config: DiscoveryWorldEpisodeConfig,
    policy: DiscoveryWorldActionPolicy,
    repository: MechanismRepository,
    output_dir: Path,
) -> ArenaEpisodeResult:
    if config.scenario != "Reactor Lab" or config.difficulty != "Normal":
        raise ValueError("first DiscoveryWorld arena supports Reactor Lab / Normal only")
    if not isinstance(policy, DiscoveryWorldActionPolicy):
        raise TypeError("policy must implement DiscoveryWorldActionPolicy")

    environment = DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(
        config.seed,
        max_steps=config.max_steps,
    )
    context = reactor_context(config.seed)
    sidecar = ReactorLabScientificSidecar()
    artifact_store = ReactorRuleArtifactStore(Path(output_dir) / "artifacts")
    writer = ArenaRunWriter(output_dir)
    writer.write_json(
        "run.json",
        {
            "scenario": config.scenario,
            "difficulty": config.difficulty,
            "seed": config.seed,
            "max_steps": config.max_steps,
            "context_id": context.context_id,
            "ecsa_revision": _ecsa_revision(),
            "discoveryworld_revision": DISCOVERYWORLD_REVISION,
            "mechanism_repository": type(repository).__name__,
            "timestamp_utc": _timestamp_utc(),
            "arm": None,
            "policy_config_hash": None,
        },
    )

    admitted_this_run: list[MechanismRecord] = []
    frozen: dict[str, ReactorMechanismHypothesis] = {}
    run_metrics = ReactorRunMetricsAccumulator()
    tested_transfer_refs: set[str] = set()
    accepted_transfer_refs: set[str] = set()

    while not environment.done and environment.steps < config.max_steps:
        pre = environment.observe()
        writer.append_jsonl(
            "observations.jsonl",
            {"step": environment.steps, "phase": "pre", "observation": pre},
        )

        transfer_candidates = repository.find_transfer_candidates(context)
        transfer_candidate_refs = {
            candidate.ref for candidate in transfer_candidates
        }
        for candidate in transfer_candidates:
            run_metrics.record_candidate_retrieved(
                f"{candidate.mechanism_id}@{candidate.version}"
            )
        scientific_context = ScientificContext(
            context_id=context.context_id or "",
            measurements=sidecar.measurements,
            transfer_candidates=transfer_candidates,
            admitted_mechanisms=repository.find_applicable(context),
        )
        decision = policy.decide(
            pre,
            environment.available_actions(),
            environment.teleport_locations(),
            scientific_context,
        )
        if not isinstance(decision, PolicyDecision):
            raise TypeError("policy must return PolicyDecision")

        for hypothesis in decision.hypotheses:
            if (
                hypothesis.source_mechanism is not None
                and hypothesis.source_mechanism not in transfer_candidate_refs
            ):
                raise ValueError(
                    "hypothesis source_mechanism must be a current transfer candidate"
                )
            previous = frozen.get(hypothesis.hypothesis_id)
            if previous is None:
                sidecar.freeze_hypothesis(hypothesis)
                frozen[hypothesis.hypothesis_id] = hypothesis
                writer.append_jsonl(
                    "scientific_events.jsonl",
                    {"kind": "hypothesis_frozen", "hypothesis": hypothesis},
                )
            elif previous != hypothesis:
                raise ValueError(
                    f"hypothesis_id reused with different content: {hypothesis.hypothesis_id}"
                )

        for hypothesis_id in decision.validation_hypothesis_ids:
            if hypothesis_id not in frozen:
                raise ValueError(
                    f"validation requested for unfrozen hypothesis: {hypothesis_id}"
                )
            hypothesis = frozen[hypothesis_id]
            if hypothesis.source_mechanism is not None:
                ref = (
                    f"{hypothesis.source_mechanism.mechanism_id}@"
                    f"{hypothesis.source_mechanism.version}"
                )
                run_metrics.record_candidate_tested(ref)
                tested_transfer_refs.add(ref)

        measurement_key = measurement_action_key(
            pre,
            decision.action,
        )
        if measurement_key is not None:
            run_metrics.record_measurement(
                instrument_uuid=measurement_key[0],
                crystal_uuid=measurement_key[1],
            )

        writer.append_jsonl(
            "actions.jsonl",
            {
                "step": environment.steps,
                "action": decision.action,
                "reasoning": decision.reasoning,
                "memory": decision.memory,
            },
        )
        action_result = environment.act(decision.action)
        post = environment.observe()
        writer.append_jsonl(
            "observations.jsonl",
            {"step": environment.steps, "phase": "post", "observation": post},
        )

        measurements = sidecar.record_transition(
            step=environment.steps,
            context_id=context.context_id or "",
            pre_observation=pre,
            action=decision.action,
            action_result=action_result,
            post_observation=post,
        )
        for measurement in measurements:
            writer.append_jsonl(
                "scientific_events.jsonl",
                {"kind": "measurement", "measurement": measurement},
            )

        validations = sidecar.observe_validation(
            step=environment.steps,
            pre_observation=pre,
            post_observation=post,
            hypothesis_ids=decision.validation_hypothesis_ids,
        )
        for validation in validations:
            hypothesis = frozen[validation.hypothesis_id]
            writer.append_jsonl(
                "scientific_events.jsonl",
                {"kind": "validation", "validation": validation},
            )
            if not validation.success:
                if hypothesis.source_mechanism is not None:
                    ref = (
                        f"{hypothesis.source_mechanism.mechanism_id}@"
                        f"{hypothesis.source_mechanism.version}"
                    )
                    run_metrics.record_candidate_rejected(
                        ref,
                        false_transfer=True,
                    )
                continue

            mechanism = build_admitted_reactor_mechanism(
                hypothesis=hypothesis,
                validation=validation,
                context_id=context.context_id or "",
                artifact_store=artifact_store,
            )
            repository.admit(mechanism)
            admitted_this_run.append(mechanism)
            if hypothesis.source_mechanism is not None:
                ref = (
                    f"{hypothesis.source_mechanism.mechanism_id}@"
                    f"{hypothesis.source_mechanism.version}"
                )
                run_metrics.record_candidate_accepted(ref)
                accepted_transfer_refs.add(ref)
            writer.append_jsonl(
                "mechanism_events.jsonl",
                {"kind": "admitted", "mechanism": mechanism},
            )

    for ref in sorted(tested_transfer_refs - accepted_transfer_refs):
        run_metrics.record_candidate_rejected(ref)

    evaluation = environment.evaluate_after_run()
    writer.write_json("final_scorecard.json", evaluation.scorecard)
    snapshot = run_metrics.snapshot()
    writer.write_json(
        "metrics.json",
        {
            "completed_successfully": evaluation.completed_successfully,
            "score_normalized": evaluation.score_normalized,
            "steps": evaluation.steps,
            "measurement_count": snapshot.measurement_actions,
            "measurement_actions": snapshot.measurement_actions,
            "parsed_measurement_count": len(sidecar.measurements),
            "distinct_measurements": snapshot.distinct_measurements,
            "transfer_candidates_retrieved": snapshot.transfer_candidates_retrieved,
            "transfer_candidates_tested": snapshot.transfer_candidates_tested,
            "transfer_candidates_accepted": snapshot.transfer_candidates_accepted,
            "transfer_candidates_rejected": snapshot.transfer_candidates_rejected,
            "false_transfer_events": snapshot.false_transfer_events,
            "admitted_mechanism_count": len(admitted_this_run),
        },
    )

    return ArenaEpisodeResult(
        config=config,
        evaluation=evaluation,
        admitted_mechanisms=tuple(admitted_this_run),
        measurement_count=snapshot.measurement_actions,
    )



def policy_config_hash(config: dict) -> str:
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


def _augment_run_metadata(
    output_dir: Path,
    *,
    arm: ArenaArm,
    config_hash: str,
) -> None:
    path = Path(output_dir) / "run.json"
    if not path.exists():
        return
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("run.json must contain an object")
    value["arm"] = arm.value
    value["policy_config_hash"] = config_hash
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n"
    )


def run_progressive_transfer(
    *,
    seeds: tuple[int, ...],
    policy_factory: Callable[[dict], DiscoveryWorldActionPolicy],
    policy_config: dict,
    output_dir: Path,
    max_steps: int = 1000,
    episode_runner=run_episode,
) -> TransferArenaResult:
    if seeds != tuple(sorted(seeds)) or not seeds:
        raise ValueError("seeds must be a nonempty ascending tuple")
    if any(type(seed) is not int or seed not in range(5) for seed in seeds):
        raise ValueError("first DiscoveryWorld arena supports official seeds 0..4")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_hash = policy_config_hash(policy_config)

    reuse_db = root / "reuse-mechanisms.sqlite"
    if reuse_db.exists():
        raise FileExistsError(
            f"reuse repository already exists: {reuse_db}"
        )
    reuse_repo = MLMDMechanismRepository.sqlite(reuse_db)
    reuse_count = 0
    pairs: list[PairedSeedResult] = []

    for seed in seeds:
        seed_dir = root / f"seed-{seed}"
        cold_dir = seed_dir / "cold"
        reuse_dir = seed_dir / "reuse"
        cold_db = seed_dir / "cold-mechanisms.sqlite"
        if cold_db.exists():
            raise FileExistsError(
                f"cold repository already exists: {cold_db}"
            )
        cold_repo = MLMDMechanismRepository.sqlite(cold_db)
        episode_config = DiscoveryWorldEpisodeConfig(
            scenario="Reactor Lab",
            difficulty="Normal",
            seed=seed,
            max_steps=max_steps,
        )

        cold_policy = _create_policy(policy_factory, policy_config)
        reuse_policy = _create_policy(policy_factory, policy_config)

        cold_episode = episode_runner(
            config=episode_config,
            policy=cold_policy,
            repository=cold_repo,
            output_dir=cold_dir,
        )
        _augment_run_metadata(
            cold_dir,
            arm=ArenaArm.COLD,
            config_hash=config_hash,
        )
        cold_admitted = len(cold_episode.admitted_mechanisms)
        cold_result = ArmEpisodeResult(
            arm=ArenaArm.COLD,
            seed=seed,
            episode=cold_episode,
            starting_mechanism_count=0,
            ending_mechanism_count=cold_admitted,
            policy_config_hash=config_hash,
        )

        reuse_start = reuse_count
        reuse_episode = episode_runner(
            config=episode_config,
            policy=reuse_policy,
            repository=reuse_repo,
            output_dir=reuse_dir,
        )
        _augment_run_metadata(
            reuse_dir,
            arm=ArenaArm.REUSE,
            config_hash=config_hash,
        )
        reuse_count += len(reuse_episode.admitted_mechanisms)
        reuse_result = ArmEpisodeResult(
            arm=ArenaArm.REUSE,
            seed=seed,
            episode=reuse_episode,
            starting_mechanism_count=reuse_start,
            ending_mechanism_count=reuse_count,
            policy_config_hash=config_hash,
        )
        pairs.append(
            PairedSeedResult(
                seed=seed,
                cold=cold_result,
                reuse=reuse_result,
            )
        )

    return TransferArenaResult(
        pairs=tuple(pairs),
        policy_config_hash=config_hash,
    )



def load_policy_factory(
    path: str,
) -> Callable[[dict], DiscoveryWorldActionPolicy]:
    if not isinstance(path, str) or path.count(":") != 1:
        raise ValueError("policy factory must use module:factory syntax")
    module_name, attribute = path.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("policy factory must use module:factory syntax")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"cannot import policy module: {module_name}") from exc
    if not hasattr(module, attribute):
        raise ValueError(f"policy factory not found: {path}")
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"policy factory is not callable: {path}")
    return factory


def _create_policy(
    factory: Callable[[dict], DiscoveryWorldActionPolicy],
    config: dict,
) -> DiscoveryWorldActionPolicy:
    policy = factory(deepcopy(config))
    if not isinstance(policy, DiscoveryWorldActionPolicy):
        raise TypeError("policy factory must return DiscoveryWorldActionPolicy")
    return policy


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _summary_wire(result: TransferArenaResult) -> dict:
    transfer = summarize_transfer(result)
    by_seed = {item.seed: item for item in transfer.per_seed}
    return {
        "policy_config_hash": result.policy_config_hash,
        "mean_measurement_transfer_gain": (
            transfer.mean_measurement_transfer_gain
        ),
        "mean_step_transfer_gain": transfer.mean_step_transfer_gain,
        "pairs": [
            {
                "seed": pair.seed,
                "cold": {
                    "completed_successfully": pair.cold.episode.evaluation.completed_successfully,
                    "score_normalized": pair.cold.episode.evaluation.score_normalized,
                    "steps": pair.cold.episode.evaluation.steps,
                    "starting_mechanism_count": pair.cold.starting_mechanism_count,
                    "ending_mechanism_count": pair.cold.ending_mechanism_count,
                },
                "reuse": {
                    "completed_successfully": pair.reuse.episode.evaluation.completed_successfully,
                    "score_normalized": pair.reuse.episode.evaluation.score_normalized,
                    "steps": pair.reuse.episode.evaluation.steps,
                    "starting_mechanism_count": pair.reuse.starting_mechanism_count,
                    "ending_mechanism_count": pair.reuse.ending_mechanism_count,
                },
                "measurement_transfer_gain": (
                    by_seed[pair.seed].measurement_transfer_gain
                ),
                "step_transfer_gain": by_seed[pair.seed].step_transfer_gain,
            }
            for pair in result.pairs
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the ECSA DiscoveryWorld Reactor Lab transfer arena."
    )
    parser.add_argument("--scenario", default="Reactor Lab")
    parser.add_argument("--difficulty", default="Normal")
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2, 3, 4))
    parser.add_argument("--arms", default="cold,reuse")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-factory", required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.scenario != "Reactor Lab" or args.difficulty != "Normal":
        parser.error("first arena supports only Reactor Lab / Normal")
    if args.arms != "cold,reuse":
        parser.error("first arena requires --arms cold,reuse")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")

    config_value = json.loads(args.policy_config.read_text())
    if not isinstance(config_value, dict):
        parser.error("--policy-config must contain a JSON object")
    factory = load_policy_factory(args.policy_factory)
    result = run_progressive_transfer(
        seeds=args.seeds,
        policy_factory=factory,
        policy_config=config_value,
        output_dir=args.output,
        max_steps=args.max_steps,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(_summary_wire(result), sort_keys=True, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
