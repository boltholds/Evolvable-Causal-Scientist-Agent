from ecsa.world_model.canonical import canonicalize_world_contract
from ecsa.world_model.hypotheses import (
    HypothesisStatus,
    WorldContractHypothesis,
)
from ecsa.world_model.learners.base import (
    AcquisitionActionStep,
    AcquisitionTrace,
    ActionModelProposal,
)
from ecsa.world_model.learners.locm2 import Locm2Learner


def _trace(
    *,
    prefix: str = "",
    object_ref: str = "object-1",
) -> AcquisitionTrace:
    names = {
        "a_on": f"{prefix}a_on",
        "a_off": f"{prefix}a_off",
        "b_on": f"{prefix}b_on",
        "b_off": f"{prefix}b_off",
    }
    sequence = (
        "a_on",
        "a_off",
        "b_on",
        "b_off",
        "a_on",
        "b_on",
        "a_off",
        "b_off",
        "b_on",
        "b_off",
        "a_on",
        "a_off",
        "b_off",
        "a_on",
        "b_on",
        "a_off",
    )
    return AcquisitionTrace(
        trace_id=f"{prefix or 'base'}-trace",
        steps=tuple(
            AcquisitionActionStep(
                schema_id=names[name],
                object_refs=(object_ref,),
                evidence_id=f"{prefix or 'base'}-e{index}",
            )
            for index, name in enumerate(sequence, start=1)
        ),
    )


def _contract(
    proposal: ActionModelProposal,
    contract_id: str,
) -> WorldContractHypothesis:
    return WorldContractHypothesis(
        contract_id=contract_id,
        entity_types=proposal.entity_types,
        predicates=proposal.predicates,
        numeric_fluents=(),
        argument_roles=(),
        actions=proposal.actions,
        affordances=(),
        supporting_evidence_ids=proposal.supporting_evidence_ids,
        contradicting_evidence_ids=(),
        confidence=0.5,
        status=HypothesisStatus.PROPOSED,
    )


def test_locm2_produces_multiple_state_machines_for_one_latent_sort() -> None:
    learner = Locm2Learner()

    analysis = learner.analyze((_trace(),))

    assert len(analysis.sorts) == 1
    [sort] = analysis.sorts
    assert len(sort.machine_transition_sets) >= 3
    proper = [
        frozenset(machine)
        for machine in sort.machine_transition_sets
        if len(machine) < len(sort.all_transitions)
    ]
    assert len(proper) >= 2


def test_locm2_output_translates_to_standard_action_model_proposal() -> None:
    result = Locm2Learner().update((_trace(),), ())

    assert isinstance(result, ActionModelProposal), result
    assert result.learner_id == "locm2"
    assert result.entity_types
    assert result.predicates
    assert result.actions
    assert {
        action.source_schema_id
        for action in result.actions
    } == {
        "a_on",
        "a_off",
        "b_on",
        "b_off",
    }


def test_locm2_is_invariant_to_action_and_object_renaming() -> None:
    learner = Locm2Learner()
    left = learner.update((_trace(),), ())
    right = learner.update(
        (_trace(prefix="renamed_", object_ref="totally-different"),),
        (),
    )

    assert isinstance(left, ActionModelProposal), left
    assert isinstance(right, ActionModelProposal), right

    left_fingerprint = canonicalize_world_contract(
        _contract(left, "left")
    ).fingerprint
    right_fingerprint = canonicalize_world_contract(
        _contract(right, "right")
    ).fingerprint

    assert left_fingerprint == right_fingerprint
