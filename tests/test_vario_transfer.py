from pathlib import Path

from ecsa.adapters.stitch import StitchAbstractionProposal
from ecsa.adapters.vario import (
    MechanismTransferScope,
    VarioContextEvidence,
    VarioTransferValidator,
)
from ecsa.repair import RepairFamily


def abstraction() -> StitchAbstractionProposal:
    return StitchAbstractionProposal(
        proposal_id="stitch:fn_0:abc123",
        engine_id="stitch",
        family=RepairFamily.ABSTRACTION_TRANSFER,
        artifact_id="stitch-abstraction:sha256:" + "a" * 64,
        source_theory_ids=("m1", "m2", "m3"),
        source_program_artifact_ids=("p1", "p2", "p3"),
        body="(and #0 (not #1))",
        arity=2,
        utility=100,
        compression_ratio=1.5,
        num_uses=3,
    )


def context(
    context_id: str,
    slope: float,
    *,
    n: int = 60,
) -> VarioContextEvidence:
    x = tuple((index + 1) / n for index in range(n))
    noise = tuple(
        0.02 if index % 2 == 0 else -0.02
        for index in range(n)
    )
    y = tuple(
        slope * xv + epsilon
        for xv, epsilon in zip(x, noise)
    )
    return VarioContextEvidence(
        context_id=context_id,
        x=x,
        y=y,
    )


def test_real_vario_detects_context_specialization_for_stitch_abstraction(
    tmp_path: Path,
) -> None:
    validator = VarioTransferValidator(
        cache_root=tmp_path,
        x_degree=1,
    )

    decision = validator.evaluate(
        abstraction(),
        (
            context("c1", 2.0),
            context("c2", 2.0),
            context("c3", 8.0),
        ),
    )

    assert decision.scope is MechanismTransferScope.CONTEXT_SPECIALIZED
    assert decision.partition == (("c1", "c2"), ("c3",))
    assert decision.artifact_id.startswith("vario-transfer:sha256:")

    artifact = validator.load_artifact(decision.artifact_id)
    assert artifact["schema"] == "ecsa.vario-transfer.v1"
    assert artifact["source_url"] == (
        "https://vreeken.groups.cispa.de/prj/vario/vario-v20220815.zip"
    )
    assert artifact["source_sha256"] == (
        "67df25a256622f86fe7c7b469e2928ddcd2252680b18502699786041ca554652"
    )
    assert artifact["source_stitch_artifact_id"] == abstraction().artifact_id
    assert artifact["vario_partition"] == [["c1", "c2"], ["c3"]]
    assert artifact["vario_score"] >= 0.0


def test_real_vario_keeps_invariant_contexts_shared(
    tmp_path: Path,
) -> None:
    validator = VarioTransferValidator(
        cache_root=tmp_path,
        x_degree=1,
    )

    decision = validator.evaluate(
        abstraction(),
        (
            context("c1", 3.0),
            context("c2", 3.0),
            context("c3", 3.0),
        ),
    )

    assert decision.scope is MechanismTransferScope.SHARED
    assert decision.partition == (("c1", "c2", "c3"),)


def test_real_vario_splits_when_every_context_has_a_distinct_mechanism(
    tmp_path: Path,
) -> None:
    validator = VarioTransferValidator(
        cache_root=tmp_path,
        x_degree=1,
    )

    decision = validator.evaluate(
        abstraction(),
        (
            context("c1", 1.0),
            context("c2", 7.0),
            context("c3", 15.0),
        ),
    )

    assert decision.scope is MechanismTransferScope.SPLIT
    assert decision.partition == (("c1",), ("c2",), ("c3",))
