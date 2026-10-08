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
from ecsa.world_model.learners.locm2 import Locm2Learner
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
    successes: int = 0
    applicability_eig_count: int = 0
    contract_eig_count: int = 0
    applicability_hypothesis_count: int = 0
    model_prediction_count: int = 0
    schema_brier: float = 0.0
    applicability_brier: float = 0.0
    model_ready_schema_brier: float = 0.0
    model_ready_applicability_brier: float = 0.0
    lifted_eig_count: int = 0
    lifted_hypothesis_count: int = 0
    lifted_prediction_count: int = 0
    lifted_brier: float = 0.0
    lifted_ready_brier: float = 0.0
    lifted_ready_schema_brier: float = 0.0


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
    use_affordance_scoring: bool = True,
    use_applicability_selection: bool = True,
    use_lifted_selection: bool = True,
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
    world_model = WorldModelAcquisitionKernel(
        learners=(Locm2Learner(),),
    )
    perception = StructuredObservationFrontend(
        DiscoveryWorldStructuredDecoder()
    )
    scientist = AutonomousScientist(
        world_model=world_model,
        experiments=ContractExperimentCoordinator(
            use_affordance_scoring=use_affordance_scoring,
            use_applicability_selection=use_applicability_selection,
            use_lifted_selection=use_lifted_selection,
        ),
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
    successes = 0
    applicability_eig_count = 0
    contract_eig_count = 0
    lifted_eig_count = 0
    lifted_prediction_count = 0
    lifted_brier_total = 0.0
    ready_lifted_brier_total = 0.0
    ready_lifted_schema_brier_total = 0.0
    action_signatures: set[tuple[str, int]] = set()
    evidence_ids: set[str] = set()
    model_prediction_count = 0
    schema_brier_total = 0.0
    applicability_brier_total = 0.0
    ready_schema_brier_total = 0.0
    ready_applicability_brier_total = 0.0

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
        # Prequential evaluation: each prediction is calculated from the
        # history available *before* executing the selected ground action.
        history = scientist.experiments.history
        schema_probability = history.schema_success_probability(
            experiment.action.schema_id, len(experiment.action.arguments),
        )
        likelihood = schema_probability
        lifted_likelihood = schema_probability
        before_perception = perception.perceive(before)
        lifted_belief = scientist.experiments.lifted_applicability.belief(
            experiment.action.schema_id, len(experiment.action.arguments),
        )
        if lifted_belief is not None:
            prospective_lifted = scientist.experiments.lifted_applicability.predictions(
                lifted_belief,
                experiment_id=experiment.experiment_id,
                action=experiment.action,
                before=before_perception,
            )
            lifted_probabilities = {
                prediction.theory_id: prediction.probability(("success",))
                for prediction in prospective_lifted
            }
            lifted_likelihood = sum(
                mass * lifted_probabilities[theory_id]
                for theory_id, mass in lifted_belief.posterior.probabilities
            )
            lifted_prediction_count += 1
        active_belief = scientist.experiments.applicability.belief(
            experiment.action.schema_id, len(experiment.action.arguments),
        )
        if active_belief is not None:
            prospective = scientist.experiments.applicability.predictions(
                active_belief,
                experiment_id=experiment.experiment_id,
                action=experiment.action,
                before=before_perception,
            )
            by_theory = {
                prediction.theory_id: prediction.probability(("success",))
                for prediction in prospective
            }
            likelihood = sum(
                mass * by_theory[theory_id]
                for theory_id, mass in active_belief.posterior.probabilities
            )
            model_prediction_count += 1

        selection_mode = scientist.experiments.last_selection_mode.value
        applicability_eig_count += int(selection_mode == "applicability_eig")
        contract_eig_count += int(selection_mode == "world_contract_eig")
        lifted_eig_count += int(selection_mode == "lifted_applicability_eig")
        action_signatures.add((experiment.action.schema_id, len(experiment.action.arguments)))
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
        successes += int(outcome.success)
        observation_outcome = float(outcome.success)
        baseline_loss = (schema_probability - observation_outcome) ** 2
        model_loss = (likelihood - observation_outcome) ** 2
        schema_brier_total += baseline_loss
        applicability_brier_total += model_loss
        lifted_loss = (lifted_likelihood - observation_outcome) ** 2
        lifted_brier_total += lifted_loss
        if lifted_belief is not None:
            ready_lifted_brier_total += lifted_loss
            ready_lifted_schema_brier_total += baseline_loss
        if active_belief is not None:
            ready_schema_brier_total += baseline_loss
            ready_applicability_brier_total += model_loss

        _append_jsonl(
            transition_path,
            {
                "transition_id": transition.transition_id,
                "selection_mode": selection_mode,
                "schema_success_probability": schema_probability,
                "applicability_success_probability": likelihood,
                "lifted_success_probability": lifted_likelihood,
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
        successes=successes,
        applicability_eig_count=applicability_eig_count,
        contract_eig_count=contract_eig_count,
        model_prediction_count=model_prediction_count,
        schema_brier=schema_brier_total / max(1, transition_count),
        applicability_brier=applicability_brier_total / max(1, transition_count),
        lifted_brier=lifted_brier_total / max(1, transition_count),
        lifted_prediction_count=lifted_prediction_count,
        lifted_ready_brier=ready_lifted_brier_total / max(1, lifted_prediction_count),
        lifted_ready_schema_brier=ready_lifted_schema_brier_total / max(1, lifted_prediction_count),
        lifted_eig_count=lifted_eig_count,
        lifted_hypothesis_count=sum(
            len(belief.hypotheses)
            for schema_id, arity in sorted(action_signatures)
            if (belief := scientist.experiments.lifted_applicability.belief(schema_id, arity)) is not None
        ),
        model_ready_schema_brier=ready_schema_brier_total / max(1, model_prediction_count),
        model_ready_applicability_brier=ready_applicability_brier_total / max(1, model_prediction_count),
        applicability_hypothesis_count=sum(
            len(belief.hypotheses)
            for schema_id, arity in sorted(action_signatures)
            if (belief := scientist.experiments.applicability.belief(schema_id, arity)) is not None
        ),
    )
