from pathlib import Path

from ecsa.adapters.dreamcoder import (
    BooleanMechanismExample,
    DreamCoderRepairEngine,
)
from ecsa.contracts import (
    Observation,
    PopulationAnomaly,
    PredictiveDistribution,
    TheoryPosterior,
    TheoryRef,
)
from ecsa.repair import (
    RepairCoordinator,
    RepairFamily,
    RepairSuccess,
    TheorySpaceExpansionRequest,
)


def request() -> TheorySpaceExpansionRequest:
    theory = TheoryRef("h1", "artifact:h1")
    prior = TheoryPosterior((("h1", 1.0),))
    predictions = (
        PredictiveDistribution(
            "h1",
            "anomaly-xor",
            (((0,), 1.0), ((1,), 0.0)),
        ),
    )
    anomaly = PopulationAnomaly(
        Observation("anomaly-xor", (1,)),
        0.0,
        (("h1", 0.0),),
    )
    return TheorySpaceExpansionRequest.from_population_anomaly(
        request_id="expand-program",
        anomaly=anomaly,
        active_theories=(theory,),
        prior=prior,
        predictions=predictions,
    )


def xor_examples() -> tuple[BooleanMechanismExample, ...]:
    return (
        BooleanMechanismExample((False, False), False, "e00"),
        BooleanMechanismExample((False, True), True, "e01"),
        BooleanMechanismExample((True, False), True, "e10"),
        BooleanMechanismExample((True, True), False, "e11"),
    )


def test_real_dreamcoder_synthesizes_program_mechanism_and_records_provenance(
    tmp_path: Path,
) -> None:
    engine = DreamCoderRepairEngine(
        xor_examples(),
        cache_root=tmp_path,
        primitives=("not", "and", "or"),
        maximum_mdl=14.0,
        maximum_programs=200_000,
    )

    run = RepairCoordinator().expand(request(), (engine,))

    assert len(run.reports) == 1
    assert isinstance(run.reports[0], RepairSuccess)
    assert len(run.proposals) == 1

    proposal = run.proposals[0]
    assert proposal.engine_id == "dreamcoder"
    assert proposal.family is RepairFamily.PROGRAM_MECHANISM
    assert proposal.parent_theory_ids == ("h1",)
    assert proposal.evidence_experiment_ids == (
        "e00",
        "e01",
        "e10",
        "e11",
        "anomaly-xor",
    )
    assert proposal.artifact_id.startswith("dreamcoder-program:sha256:")

    artifact = engine.load_artifact(proposal.artifact_id)
    assert artifact["schema"] == "ecsa.dreamcoder-program.v1"
    assert artifact["source_repository"] == "https://github.com/ellisk42/ec.git"
    assert artifact["source_revision"] == "cb0e63f5c33cd2de360b791038b0f5272750270e"
    assert artifact["domain"] == "boolean"
    assert artifact["arity"] == 2
    assert artifact["primitives"] == ["not", "and", "or"]
    assert artifact["program"]
    assert artifact["programs_enumerated"] > 0
    assert artifact["mdl"] <= 14.0
    assert artifact["verified_examples"] == 4


def test_dreamcoder_returns_no_proposal_when_dsl_cannot_express_examples(
    tmp_path: Path,
) -> None:
    engine = DreamCoderRepairEngine(
        xor_examples(),
        cache_root=tmp_path,
        primitives=("and",),
        maximum_mdl=8.0,
        maximum_programs=10_000,
    )

    run = RepairCoordinator().expand(request(), (engine,))

    assert isinstance(run.reports[0], RepairSuccess)
    assert run.proposals == ()
