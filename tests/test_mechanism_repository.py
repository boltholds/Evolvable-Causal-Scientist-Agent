from pathlib import Path

from ecsa.adapters.mlmd import MLMDMechanismRepository
from ecsa.mechanisms import (
    ApplicabilityContext,
    MechanismKind,
    MechanismRecord,
    MechanismRelation,
    MechanismRelationKind,
    MechanismScope,
    MechanismStatus,
    MechanismTransferStatus,
)


def record(
    *,
    mechanism_id: str = "door-law",
    version: int = 1,
    status: MechanismStatus = MechanismStatus.ADMITTED,
    representation: str = "dreamcoder-program:sha256:" + "a" * 64,
    contexts: tuple[str, ...] = ("factory-a",),
    relations: tuple[MechanismRelation, ...] = (),
) -> MechanismRecord:
    return MechanismRecord(
        mechanism_id=mechanism_id,
        version=version,
        kind=MechanismKind.PROGRAM,
        status=status,
        representation_artifact_id=representation,
        scope=MechanismScope(
            context_ids=contexts,
            regime_ids=("steady",),
            domain_ids=("doors",),
            task_ids=("open-door",),
            required_assumptions=("deterministic", "binary-inputs"),
        ),
        transfer=MechanismTransferStatus.SHARED,
        supporting_evidence_ids=("heldout-e1", "vario-transfer:v1"),
        contradicting_evidence_ids=(),
        provenance_artifact_ids=(
            "dreamcoder-program:p1",
            "stitch-abstraction:s1",
            "vario-transfer:v1",
        ),
        relations=relations,
    )


def applicable_factory_a() -> ApplicabilityContext:
    return ApplicabilityContext(
        context_id="factory-a",
        regime_id="steady",
        domain_id="doors",
        task_id="open-door",
        assumptions=("binary-inputs", "deterministic", "observed-C"),
    )


def test_mlmd_repository_persists_mechanism_and_exact_input_lineage(
    tmp_path: Path,
) -> None:
    repo = MLMDMechanismRepository.sqlite(tmp_path / "mechanisms.sqlite")
    admitted = record()

    repo.admit(admitted)

    assert repo.get("door-law", 1) == admitted
    assert repo.get("door-law") == admitted

    lineage = repo.lineage("door-law", 1)
    assert lineage is not None
    assert lineage.record == admitted
    assert lineage.representation_artifact_id == admitted.representation_artifact_id
    assert lineage.supporting_evidence_ids == admitted.supporting_evidence_ids
    assert lineage.contradicting_evidence_ids == ()
    assert lineage.provenance_artifact_ids == admitted.provenance_artifact_ids
    assert lineage.execution_kind == "MechanismAdmission"


def test_applicability_is_conservative_over_scope_regime_and_assumptions(
    tmp_path: Path,
) -> None:
    repo = MLMDMechanismRepository.sqlite(tmp_path / "mechanisms.sqlite")
    repo.admit(record())

    assert repo.find_applicable(applicable_factory_a()) == (record(),)

    wrong_context = ApplicabilityContext(
        context_id="factory-b",
        regime_id="steady",
        domain_id="doors",
        task_id="open-door",
        assumptions=("binary-inputs", "deterministic"),
    )
    assert repo.find_applicable(wrong_context) == ()

    missing_assumption = ApplicabilityContext(
        context_id="factory-a",
        regime_id="steady",
        domain_id="doors",
        task_id="open-door",
        assumptions=("binary-inputs",),
    )
    assert repo.find_applicable(missing_assumption) == ()


def test_superseding_mechanism_hides_superseded_record_from_reuse(
    tmp_path: Path,
) -> None:
    repo = MLMDMechanismRepository.sqlite(tmp_path / "mechanisms.sqlite")
    old = record(mechanism_id="door-law")
    replacement = record(
        mechanism_id="door-law-v2",
        representation="dreamcoder-program:sha256:" + "b" * 64,
        relations=(
            MechanismRelation(
                MechanismRelationKind.SUPERSEDES,
                "door-law",
                1,
            ),
        ),
    )
    repo.admit(old)
    repo.admit(replacement)

    applicable = repo.find_applicable(applicable_factory_a())

    assert applicable == (replacement,)
    assert repo.get("door-law", 1) == old


def test_deprecation_creates_new_version_and_preserves_old_lineage(
    tmp_path: Path,
) -> None:
    repo = MLMDMechanismRepository.sqlite(tmp_path / "mechanisms.sqlite")
    admitted = record()
    repo.admit(admitted)

    deprecated = repo.deprecate(
        "door-law",
        reason="prospective contradiction e99",
    )

    assert deprecated.version == 2
    assert deprecated.status is MechanismStatus.DEPRECATED
    assert repo.get("door-law", 1) == admitted
    assert repo.get("door-law", 2) == deprecated
    assert repo.get("door-law") == deprecated
    assert repo.find_applicable(applicable_factory_a()) == ()

    lineage = repo.lineage("door-law", 2)
    assert lineage is not None
    assert lineage.execution_kind == "MechanismDeprecation"
    assert lineage.prior_version == ("door-law", 1)
