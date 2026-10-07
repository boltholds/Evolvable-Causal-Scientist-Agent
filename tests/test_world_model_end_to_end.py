from pathlib import Path

from ecsa.benchmarks.discoveryworld.arena_b import (
    run_autonomous_episode,
)
from ecsa.world_model.canonical import canonicalize_world_contract
from ecsa.world_model.contracts import (
    GroundAction,
    InteractionTransition,
    RawActionOutcome,
    RawObservation,
    freeze_raw_value,
)
from ecsa.world_model.grounding import InteractionGrounder
from ecsa.world_model.kernel import WorldModelAcquisitionKernel
from ecsa.world_model.learners.locm2 import Locm2Learner
from ecsa.world_model.perception.base import PerceptualObservation


def _raw(name: str, step: int) -> RawObservation:
    return RawObservation(
        observation_id=name,
        step=step,
        payload=freeze_raw_value({"step": step}),
    )


def _empty(name: str) -> PerceptualObservation:
    return PerceptualObservation(
        observation_id=name,
        entities=(),
        global_features=(),
    )


def test_online_kernel_turns_action_trace_into_world_contract() -> None:
    kernel = WorldModelAcquisitionKernel(
        learners=(Locm2Learner(),),
    )
    grounder = InteractionGrounder()

    transitions = (
        InteractionTransition(
            transition_id="t1",
            before=_raw("o0", 0),
            action=GroundAction(
                "A17",
                (freeze_raw_value("x"),),
            ),
            outcome=RawActionOutcome(
                True,
                freeze_raw_value({}),
            ),
            after=_raw("o1", 1),
        ),
        InteractionTransition(
            transition_id="t2",
            before=_raw("o1", 1),
            action=GroundAction(
                "B9",
                (freeze_raw_value("x"),),
            ),
            outcome=RawActionOutcome(
                True,
                freeze_raw_value({}),
            ),
            after=_raw("o2", 2),
        ),
    )

    for transition in transitions:
        grounding = grounder.observe(
            transition,
            _empty(transition.before.observation_id),
            _empty(transition.after.observation_id),
        )
        kernel.observe_grounding(grounding)

    contracts = kernel.contract_hypotheses()

    assert contracts
    assert contracts[0].actions
    assert canonicalize_world_contract(contracts[0]).fingerprint


def test_real_discoveryworld_arena_b_induces_contract_without_llm(
    tmp_path: Path,
) -> None:
    result = run_autonomous_episode(
        scenario="Reactor Lab",
        difficulty="Normal",
        seed=0,
        max_steps=4,
        output_dir=tmp_path / "arena-b",
        max_ground_actions=24,
    )

    assert result.transitions == result.steps
    assert result.contract_count > 0
