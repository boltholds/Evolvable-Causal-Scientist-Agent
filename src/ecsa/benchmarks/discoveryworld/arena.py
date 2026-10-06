from __future__ import annotations

from pathlib import Path

from ecsa.mechanisms import MechanismRecord, MechanismRepository

from .contracts import (
    ArenaEpisodeResult,
    DiscoveryWorldActionPolicy,
    DiscoveryWorldEpisodeConfig,
    PolicyDecision,
    ReactorMechanismHypothesis,
    ScientificContext,
)
from .environment import DiscoveryWorldEnvironmentAdapter
from .reactor_lab import (
    ReactorLabScientificSidecar,
    ReactorRuleArtifactStore,
    build_admitted_reactor_mechanism,
    reactor_context,
)
from .runlog import ArenaRunWriter


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

    while not environment.done and environment.steps < config.max_steps:
        pre = environment.observe()
        writer.append_jsonl(
            "observations.jsonl",
            {"step": environment.steps, "phase": "pre", "observation": pre},
        )

        scientific_context = ScientificContext(
            context_id=context.context_id or "",
            measurements=sidecar.measurements,
            transfer_candidates=repository.find_transfer_candidates(context),
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
            writer.append_jsonl(
                "scientific_events.jsonl",
                {"kind": "validation", "validation": validation},
            )
            writer.append_jsonl(
                "mechanism_events.jsonl",
                {"kind": "admitted", "mechanism": mechanism},
            )

    evaluation = environment.evaluate_after_run()
    writer.write_json("final_scorecard.json", evaluation.scorecard)
    writer.write_json(
        "metrics.json",
        {
            "completed_successfully": evaluation.completed_successfully,
            "score_normalized": evaluation.score_normalized,
            "steps": evaluation.steps,
            "measurement_count": len(sidecar.measurements),
            "admitted_mechanism_count": len(admitted_this_run),
        },
    )

    return ArenaEpisodeResult(
        config=config,
        evaluation=evaluation,
        admitted_mechanisms=tuple(admitted_this_run),
        measurement_count=len(sidecar.measurements),
    )
