from __future__ import annotations

import json
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
from .environment import DiscoveryWorldEnvironmentAdapter
from .reactor_lab import (
    ReactorLabScientificSidecar,
    ReactorRuleArtifactStore,
    build_admitted_reactor_mechanism,
    reactor_context,
)
from .runlog import ArenaRunWriter
from .metrics import ReactorRunMetricsAccumulator


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
            previous = frozen.get(hypothesis.hypothesis_id)
            if hypothesis.source_mechanism is not None:
                ref = (
                    f"{hypothesis.source_mechanism.mechanism_id}@"
                    f"{hypothesis.source_mechanism.version}"
                )
                run_metrics.record_candidate_tested(ref)
                tested_transfer_refs.add(ref)
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
            run_metrics.record_measurement(
                instrument_uuid=measurement.instrument_uuid,
                crystal_uuid=measurement.crystal_uuid,
            )
            writer.append_jsonl(
                "scientific_events.jsonl",
                {"kind": "measurement", "measurement": measurement},
            )

        validations = sidecar.observe_validation(
            step=environment.steps,
            pre_observation=pre,
            post_observation=post,
        )
        for validation in validations:
            hypothesis = frozen[validation.hypothesis_id]
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
                "scientific_events.jsonl",
                {"kind": "validation", "validation": validation},
            )
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
            "measurement_count": len(sidecar.measurements),
            "measurement_actions": snapshot.measurement_actions,
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
        measurement_count=len(sidecar.measurements),
    )



def _policy_config_hash(config: dict) -> str:
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


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
    config_hash = _policy_config_hash(policy_config)

    reuse_repo = MLMDMechanismRepository.sqlite(
        root / "reuse-mechanisms.sqlite"
    )
    reuse_count = 0
    pairs: list[PairedSeedResult] = []

    for seed in seeds:
        seed_dir = root / f"seed-{seed}"
        cold_dir = seed_dir / "cold"
        reuse_dir = seed_dir / "reuse"
        cold_repo = MLMDMechanismRepository.sqlite(
            seed_dir / "cold-mechanisms.sqlite"
        )
        episode_config = DiscoveryWorldEpisodeConfig(
            scenario="Reactor Lab",
            difficulty="Normal",
            seed=seed,
            max_steps=max_steps,
        )

        cold_policy = policy_factory(dict(policy_config))
        reuse_policy = policy_factory(dict(policy_config))
        if not isinstance(cold_policy, DiscoveryWorldActionPolicy) or not isinstance(
            reuse_policy, DiscoveryWorldActionPolicy
        ):
            raise TypeError("policy factory must return DiscoveryWorldActionPolicy")

        cold_episode = episode_runner(
            config=episode_config,
            policy=cold_policy,
            repository=cold_repo,
            output_dir=cold_dir,
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
