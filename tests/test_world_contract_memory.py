from pathlib import Path

from ecsa.world_model.canonical import canonicalize_world_contract
from ecsa.world_model.hypotheses import (
    ActionSchemaHypothesis,
    EntityTypeHypothesis,
    HypothesisStatus,
    WorldContractHypothesis,
)
from ecsa.world_model.memory import SQLiteWorldContractStore


def _contract(
    prefix: str,
    *,
    action_symbol: str,
) -> WorldContractHypothesis:
    type_id = f"{prefix}-type"
    action_id = f"{prefix}-action"
    return WorldContractHypothesis(
        contract_id=f"{prefix}-contract",
        entity_types=(
            EntityTypeHypothesis(
                hypothesis_id=type_id,
                member_refs=(f"{prefix}-entity",),
                supporting_evidence_ids=(f"{prefix}-type-e",),
                contradicting_evidence_ids=(),
                confidence=0.8,
                status=HypothesisStatus.SUPPORTED,
            ),
        ),
        predicates=(),
        numeric_fluents=(),
        argument_roles=(),
        actions=(
            ActionSchemaHypothesis(
                hypothesis_id=action_id,
                source_schema_id=action_symbol,
                parameter_type_ids=(type_id,),
                precondition_predicate_ids=(),
                add_effect_predicate_ids=(),
                delete_effect_predicate_ids=(),
                numeric_effect_ids=(),
                symmetric_parameter_groups=(),
                supporting_evidence_ids=(f"{prefix}-action-e",),
                contradicting_evidence_ids=(),
                confidence=0.8,
                status=HypothesisStatus.SUPPORTED,
            ),
        ),
        affordances=(),
        supporting_evidence_ids=(f"{prefix}-trace",),
        contradicting_evidence_ids=(),
        confidence=0.8,
        status=HypothesisStatus.SUPPORTED,
    )


def test_sqlite_store_round_trips_canonical_contract_and_provenance(
    tmp_path: Path,
) -> None:
    store = SQLiteWorldContractStore(
        tmp_path / "world-contracts.sqlite"
    )
    contract = _contract("source", action_symbol="opaque-A")

    ref = store.admit(contract)
    restored = store.get(ref)

    assert restored == contract
    assert ref.contract_id == contract.contract_id
    assert ref.version == 1
    assert ref.canonical_fingerprint == (
        canonicalize_world_contract(contract).fingerprint
    )


def test_sqlite_store_versions_contract_without_mutating_prior_version(
    tmp_path: Path,
) -> None:
    store = SQLiteWorldContractStore(tmp_path / "memory.sqlite")
    original = _contract("source", action_symbol="A")
    first = store.admit(original)

    revised = WorldContractHypothesis(
        **{
            **original.__dict__,
            "confidence": 0.9,
            "supporting_evidence_ids": (
                *original.supporting_evidence_ids,
                "new-evidence",
            ),
        }
    )
    second = store.admit(revised)

    assert first.version == 1
    assert second.version == 2
    assert store.get(first) == original
    assert store.get(second) == revised


def test_transfer_candidate_matches_structure_without_matching_source_names(
    tmp_path: Path,
) -> None:
    store = SQLiteWorldContractStore(tmp_path / "transfer.sqlite")
    source = _contract("source", action_symbol="PICKUP")
    target_shape = _contract("target", action_symbol="A17")
    source_ref = store.admit(source)

    candidates = store.find_transfer_candidates(
        canonicalize_world_contract(target_shape)
    )

    assert len(candidates) == 1
    assert candidates[0].ref == source_ref
    assert candidates[0].contract.actions[0].source_schema_id == (
        "PICKUP"
    )
    assert (
        candidates[0].ref.canonical_fingerprint
        == canonicalize_world_contract(target_shape).fingerprint
    )
