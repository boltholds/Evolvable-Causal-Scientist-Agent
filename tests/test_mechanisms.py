import pytest

from ecsa.mechanisms import (
    ApplicabilityContext,
    EpistemicStatus,
    MechanismKind,
    MechanismRecord,
    MechanismRelation,
    MechanismRelationKind,
    MechanismRepository,
    MechanismScope,
    MechanismVersionRef,
    TransferStatus,
)


def admitted_record(*, status=EpistemicStatus.ADMITTED) -> MechanismRecord:
    return MechanismRecord(
        mechanism_id="gate",
        version=1,
        kind=MechanismKind.PROGRAM,
        epistemic_status=status,
        representation_artifact="program-mechanism:sha256:" + "a" * 64,
        scope=MechanismScope(
            context_ids=("plant-a", "plant-b"),
            regime_ids=("normal",),
            required_assumptions=("deterministic", "observes:M,C,Y"),
        ),
        transfer_status=TransferStatus.SHARED,
        parameters=(("arity", "2"),),
        supporting_evidence=("heldout-1", "vario-1"),
        contradicting_evidence=(),
        provenance=(
            "dreamcoder-program:sha256:" + "b" * 64,
            "stitch-abstraction:sha256:" + "c" * 64,
            "vario-transfer:sha256:" + "d" * 64,
        ),
        relations=(
            MechanismRelation(
                MechanismRelationKind.ABSTRACTED_FROM,
                MechanismVersionRef("local-gate", 2),
            ),
        ),
    )


def test_scope_requires_explicit_contexts_and_matches_conservatively() -> None:
    with pytest.raises(ValueError, match="context"):
        MechanismScope(
            context_ids=(),
            regime_ids=("normal",),
            required_assumptions=(),
        )

    scope = admitted_record().scope
    assert scope.matches(
        ApplicabilityContext(
            context_id="plant-a",
            regime_id="normal",
            assumptions=("deterministic", "observes:M,C,Y", "extra"),
        )
    )
    assert not scope.matches(
        ApplicabilityContext(
            context_id="unseen-plant",
            regime_id="normal",
            assumptions=("deterministic", "observes:M,C,Y"),
        )
    )
    assert not scope.matches(
        ApplicabilityContext(
            context_id="plant-a",
            regime_id="normal",
            assumptions=("deterministic",),
        )
    )


def test_mechanism_record_is_typed_immutable_and_protocol_is_structural() -> None:
    record = admitted_record()
    assert record.ref == MechanismVersionRef("gate", 1)
    assert record.kind is MechanismKind.PROGRAM
    assert record.transfer_status is TransferStatus.SHARED
    assert isinstance(MechanismRepository, type)
