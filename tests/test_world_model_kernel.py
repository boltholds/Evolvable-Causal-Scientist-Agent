from dataclasses import replace

from ecsa.world_model.grounding import (
    ActionOutcomeEvidence,
    GroundedTransition,
    GroundingUpdate,
)
from ecsa.world_model.contracts import freeze_raw_value
from ecsa.world_model.hypotheses import (
    ActionSchemaHypothesis,
    EntityTypeHypothesis,
    HypothesisStatus,
    WorldContractHypothesis,
)
from ecsa.world_model.kernel import WorldModelAcquisitionKernel


def _contract(
    contract_id: str,
    *,
    action_symbol: str,
    confidence: float,
) -> WorldContractHypothesis:
    type_id = f"{contract_id}-type"
    action_id = f"{contract_id}-action"
    entity_type = EntityTypeHypothesis(
        hypothesis_id=type_id,
        member_refs=(f"{contract_id}-entity",),
        supporting_evidence_ids=("type-evidence",),
        contradicting_evidence_ids=(),
        confidence=0.6,
        status=HypothesisStatus.PROPOSED,
    )
    action = ActionSchemaHypothesis(
        hypothesis_id=action_id,
        source_schema_id=action_symbol,
        parameter_type_ids=(type_id,),
        precondition_predicate_ids=(),
        add_effect_predicate_ids=(),
        delete_effect_predicate_ids=(),
        numeric_effect_ids=(),
        symmetric_parameter_groups=(),
        supporting_evidence_ids=("action-evidence",),
        contradicting_evidence_ids=(),
        confidence=0.6,
        status=HypothesisStatus.PROPOSED,
    )
    return WorldContractHypothesis(
        contract_id=contract_id,
        entity_types=(entity_type,),
        predicates=(),
        numeric_fluents=(),
        argument_roles=(),
        actions=(action,),
        affordances=(),
        supporting_evidence_ids=("trace",),
        contradicting_evidence_ids=(),
        confidence=confidence,
        status=HypothesisStatus.PROPOSED,
    )


def test_kernel_preserves_competing_contracts_under_ambiguous_evidence() -> None:
    kernel = WorldModelAcquisitionKernel(max_contracts=4)
    first = _contract("c1", action_symbol="A", confidence=0.55)
    second = _contract("c2", action_symbol="B", confidence=0.45)

    update = kernel.ingest_proposals((first, second))

    assert {
        contract.contract_id
        for contract in kernel.contract_hypotheses()
    } == {"c1", "c2"}
    assert set(update.added_contract_ids) == {"c1", "c2"}


def test_contradicting_evidence_lowers_contract_without_mutating_history() -> None:
    kernel = WorldModelAcquisitionKernel()
    original = _contract(
        "c1",
        action_symbol="A",
        confidence=0.8,
    )
    kernel.ingest_proposals((original,))

    update = kernel.record_contract_evidence(
        "c1",
        evidence_id="contradiction-1",
        supports=False,
    )
    [current] = kernel.contract_hypotheses()

    assert current.confidence < original.confidence
    assert original.contradicting_evidence_ids == ()
    assert current.contradicting_evidence_ids == (
        "contradiction-1",
    )
    assert update.updated_contract_ids == ("c1",)


def test_repeated_strong_contradiction_can_reject_contract() -> None:
    kernel = WorldModelAcquisitionKernel()
    kernel.ingest_proposals(
        (
            _contract(
                "c1",
                action_symbol="A",
                confidence=0.2,
            ),
        )
    )

    kernel.record_contract_evidence(
        "c1",
        evidence_id="x1",
        supports=False,
    )
    [current] = kernel.contract_hypotheses()

    assert current.status is HypothesisStatus.REJECTED


def test_observe_grounding_records_failure_evidence_without_contract_guess() -> None:
    outcome = ActionOutcomeEvidence(
        evidence_id="outcome-evidence",
        schema_id="A17",
        success=False,
        arguments=(freeze_raw_value("x"),),
    )
    grounding = GroundingUpdate(
        transition=GroundedTransition(
            transition_id="t1",
            schema_id="A17",
            feature_deltas=(),
            participation=(),
            outcome_evidence=outcome,
        ),
        identity_hypotheses=(),
    )
    kernel = WorldModelAcquisitionKernel()

    update = kernel.observe_grounding(grounding)

    assert update.new_evidence_ids == ("outcome-evidence",)
    assert kernel.contract_hypotheses() == ()
