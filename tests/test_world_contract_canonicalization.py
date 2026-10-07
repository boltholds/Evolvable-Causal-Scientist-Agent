from ecsa.world_model.canonical import (
    canonicalize_world_contract,
)
from ecsa.world_model.hypotheses import (
    ActionSchemaHypothesis,
    AffordanceHypothesis,
    EntityTypeHypothesis,
    HypothesisStatus,
    NumericFluentHypothesis,
    PredicateHypothesis,
    WorldContractHypothesis,
)


def _world(
    prefix: str,
    *,
    action_symbol: str,
    use_parameter_order: tuple[int, int] = (0, 1),
    symmetric_use: bool = False,
    add_extra_effect: bool = False,
) -> WorldContractHypothesis:
    tool_id = f"{prefix}-type-tool"
    sample_id = f"{prefix}-type-sample"
    predicate_id = f"{prefix}-predicate-held"
    extra_predicate_id = f"{prefix}-predicate-extra"
    numeric_id = f"{prefix}-numeric-reading"
    pickup_id = f"{prefix}-action-pickup"
    use_id = f"{prefix}-action-use"

    entity_types = (
        EntityTypeHypothesis(
            hypothesis_id=tool_id,
            member_refs=(f"{prefix}-member-tool",),
            supporting_evidence_ids=(f"{prefix}-e1",),
            contradicting_evidence_ids=(),
            confidence=0.8,
            status=HypothesisStatus.SUPPORTED,
        ),
        EntityTypeHypothesis(
            hypothesis_id=sample_id,
            member_refs=(
                f"{prefix}-member-sample-1",
                f"{prefix}-member-sample-2",
            ),
            supporting_evidence_ids=(f"{prefix}-e2",),
            contradicting_evidence_ids=(),
            confidence=0.8,
            status=HypothesisStatus.SUPPORTED,
        ),
    )
    predicates = [
        PredicateHypothesis(
            hypothesis_id=predicate_id,
            argument_type_ids=(sample_id,),
            symmetric=False,
            supporting_evidence_ids=(f"{prefix}-e3",),
            contradicting_evidence_ids=(),
            confidence=0.7,
            status=HypothesisStatus.SUPPORTED,
        )
    ]
    if add_extra_effect:
        predicates.append(
            PredicateHypothesis(
                hypothesis_id=extra_predicate_id,
                argument_type_ids=(sample_id,),
                symmetric=False,
                supporting_evidence_ids=(f"{prefix}-e4",),
                contradicting_evidence_ids=(),
                confidence=0.7,
                status=HypothesisStatus.SUPPORTED,
            )
        )

    numeric = NumericFluentHypothesis(
        hypothesis_id=numeric_id,
        argument_type_ids=(sample_id,),
        supporting_evidence_ids=(f"{prefix}-e5",),
        contradicting_evidence_ids=(),
        confidence=0.7,
        status=HypothesisStatus.SUPPORTED,
    )
    pickup = ActionSchemaHypothesis(
        hypothesis_id=pickup_id,
        source_schema_id=f"{prefix}-opaque-pick",
        parameter_type_ids=(sample_id,),
        precondition_predicate_ids=(),
        add_effect_predicate_ids=(predicate_id,),
        delete_effect_predicate_ids=(),
        numeric_effect_ids=(),
        symmetric_parameter_groups=(),
        supporting_evidence_ids=(f"{prefix}-e6",),
        contradicting_evidence_ids=(),
        confidence=0.8,
        status=HypothesisStatus.SUPPORTED,
    )
    canonical_use_types = (tool_id, sample_id)
    use_types = tuple(
        canonical_use_types[index]
        for index in use_parameter_order
    )
    use = ActionSchemaHypothesis(
        hypothesis_id=use_id,
        source_schema_id=action_symbol,
        parameter_type_ids=use_types,
        precondition_predicate_ids=(),
        add_effect_predicate_ids=(
            (extra_predicate_id,)
            if add_extra_effect
            else ()
        ),
        delete_effect_predicate_ids=(),
        numeric_effect_ids=(numeric_id,),
        symmetric_parameter_groups=(
            ((0, 1),)
            if symmetric_use
            else ()
        ),
        supporting_evidence_ids=(f"{prefix}-e7",),
        contradicting_evidence_ids=(),
        confidence=0.8,
        status=HypothesisStatus.SUPPORTED,
    )
    affordance = AffordanceHypothesis(
        hypothesis_id=f"{prefix}-affordance-use",
        action_schema_id=use_id,
        argument_type_ids=use_types,
        success_probability=0.9,
        supporting_evidence_ids=(f"{prefix}-e8",),
        contradicting_evidence_ids=(),
        confidence=0.7,
        status=HypothesisStatus.SUPPORTED,
    )

    return WorldContractHypothesis(
        contract_id=f"{prefix}-contract",
        entity_types=entity_types,
        predicates=tuple(predicates),
        numeric_fluents=(numeric,),
        argument_roles=(),
        actions=(pickup, use),
        affordances=(affordance,),
        supporting_evidence_ids=(f"{prefix}-trace",),
        contradicting_evidence_ids=(),
        confidence=0.75,
        status=HypothesisStatus.SUPPORTED,
    )


def test_isomorphic_worlds_with_renamed_symbols_have_same_canonical_contract() -> None:
    world_a = _world("a", action_symbol="USE")
    world_b = _world("b", action_symbol="ZQ4")

    canonical_a = canonicalize_world_contract(world_a)
    canonical_b = canonicalize_world_contract(world_b)

    assert canonical_a.fingerprint == canonical_b.fingerprint
    assert canonical_a.node_signatures == canonical_b.node_signatures


def test_non_isomorphic_action_structure_changes_canonical_contract() -> None:
    world_a = _world("a", action_symbol="USE")
    world_b = _world(
        "b",
        action_symbol="ZQ4",
        add_extra_effect=True,
    )

    assert (
        canonicalize_world_contract(world_a).fingerprint
        != canonicalize_world_contract(world_b).fingerprint
    )


def test_argument_swap_is_not_collapsed_when_evidence_is_directional() -> None:
    world_a = _world("a", action_symbol="A")
    world_b = _world(
        "b",
        action_symbol="B",
        use_parameter_order=(1, 0),
    )

    assert (
        canonicalize_world_contract(world_a).fingerprint
        != canonicalize_world_contract(world_b).fingerprint
    )


def test_argument_swap_is_collapsed_when_contract_marks_relation_symmetric() -> None:
    world_a = _world(
        "a",
        action_symbol="A",
        symmetric_use=True,
    )
    world_b = _world(
        "b",
        action_symbol="B",
        use_parameter_order=(1, 0),
        symmetric_use=True,
    )

    assert (
        canonicalize_world_contract(world_a).fingerprint
        == canonicalize_world_contract(world_b).fingerprint
    )
