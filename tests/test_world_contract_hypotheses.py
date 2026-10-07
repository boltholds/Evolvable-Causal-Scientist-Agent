import pytest

from ecsa.world_model.hypotheses import (
    ActionSchemaHypothesis,
    AffordanceHypothesis,
    ArgumentRoleHypothesis,
    EntityTypeHypothesis,
    HypothesisStatus,
    NumericFluentHypothesis,
    PredicateHypothesis,
    WorldContractHypothesis,
)


def _evidence() -> tuple[str, ...]:
    return ("evidence-1",)


def test_world_contract_hypotheses_are_typed_and_provenanced() -> None:
    entity_type = EntityTypeHypothesis(
        hypothesis_id="type-1",
        member_refs=("entity-a", "entity-b"),
        supporting_evidence_ids=_evidence(),
        contradicting_evidence_ids=(),
        confidence=0.7,
        status=HypothesisStatus.SUPPORTED,
    )
    predicate = PredicateHypothesis(
        hypothesis_id="predicate-1",
        argument_type_ids=("type-1",),
        symmetric=False,
        supporting_evidence_ids=_evidence(),
        contradicting_evidence_ids=(),
        confidence=0.6,
        status=HypothesisStatus.PROPOSED,
    )
    numeric = NumericFluentHypothesis(
        hypothesis_id="numeric-1",
        argument_type_ids=("type-1",),
        supporting_evidence_ids=_evidence(),
        contradicting_evidence_ids=(),
        confidence=0.5,
        status=HypothesisStatus.PROPOSED,
    )
    role = ArgumentRoleHypothesis(
        hypothesis_id="role-1",
        action_schema_id="action-1",
        parameter_index=0,
        entity_type_id="type-1",
        supporting_evidence_ids=_evidence(),
        contradicting_evidence_ids=(),
        confidence=0.5,
        status=HypothesisStatus.PROPOSED,
    )
    action = ActionSchemaHypothesis(
        hypothesis_id="action-1",
        source_schema_id="opaque-A17",
        parameter_type_ids=("type-1",),
        precondition_predicate_ids=(),
        add_effect_predicate_ids=("predicate-1",),
        delete_effect_predicate_ids=(),
        numeric_effect_ids=("numeric-1",),
        symmetric_parameter_groups=(),
        supporting_evidence_ids=_evidence(),
        contradicting_evidence_ids=(),
        confidence=0.6,
        status=HypothesisStatus.SUPPORTED,
    )
    affordance = AffordanceHypothesis(
        hypothesis_id="affordance-1",
        action_schema_id="action-1",
        argument_type_ids=("type-1",),
        success_probability=0.8,
        supporting_evidence_ids=_evidence(),
        contradicting_evidence_ids=(),
        confidence=0.5,
        status=HypothesisStatus.PROPOSED,
    )

    contract = WorldContractHypothesis(
        contract_id="contract-1",
        entity_types=(entity_type,),
        predicates=(predicate,),
        numeric_fluents=(numeric,),
        argument_roles=(role,),
        actions=(action,),
        affordances=(affordance,),
        supporting_evidence_ids=("trace-1",),
        contradicting_evidence_ids=(),
        confidence=0.65,
        status=HypothesisStatus.SUPPORTED,
    )

    assert contract.actions[0].source_schema_id == "opaque-A17"
    assert contract.entity_types[0].member_refs == (
        "entity-a",
        "entity-b",
    )
    assert contract.affordances[0].success_probability == 0.8


def test_hypothesis_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        EntityTypeHypothesis(
            hypothesis_id="type",
            member_refs=("entity",),
            supporting_evidence_ids=("e",),
            contradicting_evidence_ids=(),
            confidence=1.1,
            status=HypothesisStatus.PROPOSED,
        )


def test_hypothesis_rejects_duplicate_evidence() -> None:
    with pytest.raises(ValueError, match="evidence"):
        EntityTypeHypothesis(
            hypothesis_id="type",
            member_refs=("entity",),
            supporting_evidence_ids=("e", "e"),
            contradicting_evidence_ids=(),
            confidence=0.5,
            status=HypothesisStatus.PROPOSED,
        )


def test_action_rejects_invalid_symmetric_parameter_group() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        ActionSchemaHypothesis(
            hypothesis_id="action",
            source_schema_id="A",
            parameter_type_ids=("type-a", "type-b"),
            precondition_predicate_ids=(),
            add_effect_predicate_ids=(),
            delete_effect_predicate_ids=(),
            numeric_effect_ids=(),
            symmetric_parameter_groups=((0, 2),),
            supporting_evidence_ids=("e",),
            contradicting_evidence_ids=(),
            confidence=0.5,
            status=HypothesisStatus.PROPOSED,
        )


def test_world_contract_rejects_dangling_action_type_reference() -> None:
    action = ActionSchemaHypothesis(
        hypothesis_id="action",
        source_schema_id="A",
        parameter_type_ids=("missing",),
        precondition_predicate_ids=(),
        add_effect_predicate_ids=(),
        delete_effect_predicate_ids=(),
        numeric_effect_ids=(),
        symmetric_parameter_groups=(),
        supporting_evidence_ids=("e",),
        contradicting_evidence_ids=(),
        confidence=0.5,
        status=HypothesisStatus.PROPOSED,
    )

    with pytest.raises(ValueError, match="type"):
        WorldContractHypothesis(
            contract_id="contract",
            entity_types=(),
            predicates=(),
            numeric_fluents=(),
            argument_roles=(),
            actions=(action,),
            affordances=(),
            supporting_evidence_ids=("trace",),
            contradicting_evidence_ids=(),
            confidence=0.5,
            status=HypothesisStatus.PROPOSED,
        )
