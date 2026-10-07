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
from ecsa.world_model.learners.base import (
    AcquisitionActionStep,
    AcquisitionTrace,
    ActionModelProposal,
    proposal_to_world_contract,
)
from ecsa.world_model.learners.locm2 import Locm2Learner
from ecsa.world_model.memory import SQLiteWorldContractStore
from ecsa.world_model.perception.base import (
    ObservedFeature,
    PerceptualObservation,
)
from ecsa.world_model.perception.slot_attention import (
    PerceptualSlot,
    SlotAttentionFrontend,
)
from ecsa.world_model.perception.structured import (
    StructuredEntityRecord,
    StructuredObservationFrontend,
)


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



class _StructuredDecoder:
    def decode_entities(
        self,
        raw_observation: RawObservation,
    ) -> tuple[StructuredEntityRecord, ...]:
        return (
            StructuredEntityRecord(
                local_ref="visible-1",
                source_identity="source-entity",
                features=(
                    ObservedFeature(
                        feature_id="appearance.embedding",
                        value=freeze_raw_value([0.2, 0.8]),
                        provenance_id=raw_observation.observation_id,
                    ),
                ),
            ),
        )


class _SlotEncoder:
    def encode(
        self,
        raw_observation: RawObservation,
    ) -> tuple[PerceptualSlot, ...]:
        return (
            PerceptualSlot(
                local_slot_id="slot-9",
                embedding=(0.2, 0.8),
                spatial_support=None,
                confidence=1.0,
                provenance_id=raw_observation.observation_id,
            ),
        )


def _trace_for_ref(
    object_ref: str,
    *,
    prefix: str,
) -> AcquisitionTrace:
    return AcquisitionTrace(
        trace_id=f"{prefix}-trace",
        steps=(
            AcquisitionActionStep(
                schema_id=f"{prefix}A",
                object_refs=(object_ref,),
                evidence_id=f"{prefix}e1",
            ),
            AcquisitionActionStep(
                schema_id=f"{prefix}B",
                object_refs=(object_ref,),
                evidence_id=f"{prefix}e2",
            ),
            AcquisitionActionStep(
                schema_id=f"{prefix}A",
                object_refs=(object_ref,),
                evidence_id=f"{prefix}e3",
            ),
        ),
    )


def _proposal_contract(
    proposal: ActionModelProposal,
):
    return proposal_to_world_contract(proposal)


def test_structured_and_slot_frontends_lead_to_equivalent_contracts() -> None:
    raw = RawObservation(
        observation_id="perception-frame",
        step=0,
        payload=freeze_raw_value({"opaque": True}),
    )
    structured = StructuredObservationFrontend(
        _StructuredDecoder()
    ).perceive(raw)
    slotted = SlotAttentionFrontend(
        _SlotEncoder()
    ).perceive(raw)

    structured_ref = (
        structured.entities[0].source_identity
        or structured.entities[0].local_ref
    )
    slot_ref = (
        slotted.entities[0].source_identity
        or slotted.entities[0].local_ref
    )

    learner = Locm2Learner()
    left = learner.update(
        (_trace_for_ref(structured_ref, prefix="left-"),),
        (),
    )
    right = learner.update(
        (_trace_for_ref(slot_ref, prefix="right-"),),
        (),
    )

    assert isinstance(left, ActionModelProposal), left
    assert isinstance(right, ActionModelProposal), right
    assert (
        canonicalize_world_contract(
            _proposal_contract(left)
        ).fingerprint
        == canonicalize_world_contract(
            _proposal_contract(right)
        ).fingerprint
    )


def test_persistent_contract_transfer_matches_structure_after_renaming(
    tmp_path: Path,
) -> None:
    learner = Locm2Learner()
    source = learner.update(
        (_trace_for_ref("source-object", prefix="source-"),),
        (),
    )
    target = learner.update(
        (_trace_for_ref("target-object", prefix="target-"),),
        (),
    )
    assert isinstance(source, ActionModelProposal), source
    assert isinstance(target, ActionModelProposal), target

    source_contract = _proposal_contract(source)
    target_contract = _proposal_contract(target)
    store = SQLiteWorldContractStore(
        tmp_path / "contracts.sqlite"
    )
    source_ref = store.admit(source_contract)

    candidates = store.find_transfer_candidates(
        canonicalize_world_contract(target_contract)
    )

    assert len(candidates) == 1
    assert candidates[0].ref == source_ref
    assert candidates[0].contract == source_contract
    assert candidates[0].contract.actions[0].source_schema_id.startswith(
        "source-"
    )
    assert target_contract.actions[0].source_schema_id.startswith(
        "target-"
    )
