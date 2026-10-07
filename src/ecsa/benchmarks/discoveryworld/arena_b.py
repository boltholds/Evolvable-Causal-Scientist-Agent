from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path

from ecsa.autonomy.scientist import AutonomousScientist
from ecsa.world_model.contracts import (
    InteractionTransition,
    freeze_raw_value,
)
from ecsa.world_model.experiments import (
    ContractExperimentCoordinator,
    ExperimentBudget,
)
from ecsa.world_model.kernel import (
    WorldModelAcquisitionKernel,
)
from ecsa.world_model.perception.structured import (
    StructuredObservationFrontend,
)

from .perception import DiscoveryWorldStructuredDecoder
from .raw_environment import DiscoveryWorldRawEnvironment


@dataclass(frozen=True)
class ArenaBResult:
    steps: int
    transitions: int
    grounding_evidence_count: int
    contract_count: int


def _wire(value):
    if is_dataclass(value):
        return _wire(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_wire(item) for item in value]
    if isinstance(value, list):
        return [_wire(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _wire(item)
            for key, item in value.items()
        }
    return value


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a") as handle:
        handle.write(
            json.dumps(
                _wire(value),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )


def run_autonomous_episode(
    *,
    scenario: str,
    difficulty: str,
    seed: int,
    max_steps: int,
    output_dir: Path,
    max_ground_actions: int = 64,
) -> ArenaBResult:
    if scenario != "Reactor Lab" or difficulty != "Normal":
        raise ValueError(
            "current DiscoveryWorld Arena B runner "
            "supports Reactor Lab / Normal loading"
        )
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be positive")

    environment = DiscoveryWorldRawEnvironment.reactor_lab_normal(
        seed,
        max_steps=max_steps,
    )
    world_model = WorldModelAcquisitionKernel()
    perception = StructuredObservationFrontend(
        DiscoveryWorldStructuredDecoder()
    )
    scientist = AutonomousScientist(
        world_model=world_model,
        experiments=ContractExperimentCoordinator(),
        perception=perception,
    )
    budget = ExperimentBudget(
        max_ground_actions=max_ground_actions,
    )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    transition_path = root / "raw_transitions.jsonl"
    outcome_path = root / "world_model_updates.jsonl"
    transition_path.write_text("")
    outcome_path.write_text("")

    transition_count = 0
    evidence_ids: set[str] = set()

    while (
        not environment.done
        and environment.steps < max_steps
    ):
        before = environment.observe_raw()
        actions = environment.list_raw_actions()
        experiment = scientist.choose_experiment(
            observation=before,
            actions=actions,
            goal=freeze_raw_value(
                {"public_observation": before.payload.thaw()}
            ),
            budget=budget,
        )
        outcome = environment.execute_raw_action(
            experiment.action
        )
        after = environment.observe_raw()
        transition = InteractionTransition(
            transition_id=(
                f"dw-transition:{seed}:"
                f"{transition_count}"
            ),
            before=before,
            action=experiment.action,
            outcome=outcome,
            after=after,
        )
        update = scientist.observe_transition(transition)
        evidence_ids.update(update.new_evidence_ids)
        transition_count += 1

        _append_jsonl(
            transition_path,
            {
                "transition_id": transition.transition_id,
                "before_id": before.observation_id,
                "action": {
                    "schema_id": experiment.action.schema_id,
                    "arguments": [
                        value.thaw()
                        for value in experiment.action.arguments
                    ],
                },
                "outcome": {
                    "success": outcome.success,
                    "payload": outcome.payload.thaw(),
                },
                "after_id": after.observation_id,
            },
        )
        _append_jsonl(
            outcome_path,
            {
                "transition_id": transition.transition_id,
                "new_evidence_ids": update.new_evidence_ids,
                "contract_ids": tuple(
                    value.contract_id
                    for value in update.contracts
                ),
            },
        )

    return ArenaBResult(
        steps=environment.steps,
        transitions=transition_count,
        grounding_evidence_count=len(evidence_ids),
        contract_count=len(
            world_model.contract_hypotheses()
        ),
    )
