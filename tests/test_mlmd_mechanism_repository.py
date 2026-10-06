from pathlib import Path

import pytest

from ecsa.adapters.mlmd import MLMDMechanismRepository
from ecsa.mechanisms import (
    ApplicabilityContext,
    EpistemicStatus,
    MechanismKind,
    MechanismRecord,
    MechanismRelation,
    MechanismRelationKind,
    MechanismScope,
    MechanismVersionRef,
    TransferStatus,
)


def record(
    mechanism_id: str,
    version: int,
    *,
    status: EpistemicStatus = EpistemicStatus.ADMITTED,
    contexts: tuple[str, ...] = ("plant-a", "plant-b"),
    relations: tuple[MechanismRelation, ...] = (),
) -> MechanismRecord:
    return MechanismRecord(
        mechanism_id=mechanism_id,
        version=version,
        kind=MechanismKind.PROGRAM,
        epistemic_status=status,
        representation_artifact="program-mechanism:sha256:" + "a" * 64,
        scope=MechanismScope(
            context_ids=contexts,
            regime_ids=("normal",),
            required_assumptions=("deterministic",),
        ),
        transfer_status=(
            TransferStatus.SHARED
            if len(contexts) > 1
            else TransferStatus.CONTEXT_SPECIALIZED
        ),
        parameters=(("arity", "2"),),
        supporting_evidence=("heldout-1",),
        contradicting_evidence=(),
        provenance=(
            "dreamcoder-program:sha256:" + "b" * 64,
            "stitch-abstraction:sha256:" + "c" * 64,
            "vario-transfer:sha256:" + "d" * 64,
        ),
        relations=relations,
    )


def test_mlmd_repository_roundtrips_and_filters_applicability(tmp_path: Path) -> None:
    repository = MLMDMechanismRepository(tmp_path / "mechanisms.sqlite")
    shared = record("shared-gate", 1)
    local = record("local-gate", 1, contexts=("plant-c",))

    repository.admit(shared)
    repository.admit(local)

    assert repository.get(shared.ref) == shared
    assert repository.get(local.ref) == local

    applicable = repository.find_applicable(
        ApplicabilityContext(
            context_id="plant-a",
            regime_id="normal",
            assumptions=("deterministic",),
        )
    )
    assert tuple(item.ref for item in applicable) == (shared.ref,)

    unseen = repository.find_applicable(
        ApplicabilityContext(
            context_id="plant-z",
            regime_id="normal",
            assumptions=("deterministic",),
        )
    )
    assert unseen == ()


def test_mlmd_repository_persists_provenance_and_mechanism_relations(
    tmp_path: Path,
) -> None:
    repository = MLMDMechanismRepository(tmp_path / "mechanisms.sqlite")
    parent = record("parent", 1, contexts=("plant-a",))
    child = record(
        "child",
        1,
        contexts=("plant-a",),
        relations=(
            MechanismRelation(
                MechanismRelationKind.SPECIALIZES,
                parent.ref,
            ),
        ),
    )
    repository.admit(parent)
    repository.admit(child)

    lineage = repository.lineage(child.ref)

    assert lineage.mechanism == child.ref
    assert lineage.provenance_artifact_ids == child.provenance
    assert lineage.relations == child.relations
    assert parent.ref in lineage.related_mechanisms


def test_mlmd_admission_rejects_unqualified_status_and_duplicate_version(
    tmp_path: Path,
) -> None:
    repository = MLMDMechanismRepository(tmp_path / "mechanisms.sqlite")

    with pytest.raises(ValueError, match="ADMITTED"):
        repository.admit(
            record(
                "candidate",
                1,
                status=EpistemicStatus.QUALIFIED,
            )
        )

    admitted = record("stable", 1)
    repository.admit(admitted)
    with pytest.raises(ValueError, match="already exists"):
        repository.admit(admitted)
