from pathlib import Path

from ecsa.adapters.dreamcoder import (
    BooleanMechanismExample,
    DreamCoderRepairEngine,
    ProgramMechanismProjector,
    QualifiedProgramMechanism,
)
from ecsa.adapters.mlmd import MLMDMechanismRepository
from ecsa.contracts import (
    ExperimentKind,
    ExperimentSpec,
    Intervention,
    Observation,
    PopulationAnomaly,
    PredictiveDistribution,
    TheoryPosterior,
    TheoryRef,
)
from ecsa.mechanisms import (
    ApplicabilityContext,
    MechanismKind,
    MechanismRecord,
    MechanismScope,
    MechanismStatus,
    MechanismTransferStatus,
)
from ecsa.repair import RepairCoordinator, TheorySpaceExpansionRequest
from ecsa.runtime_projection import (
    DreamCoderGrammarProjector,
    MechanismRuntimeProjector,
)


def request(request_id: str) -> TheorySpaceExpansionRequest:
    theory = TheoryRef("baseline", "artifact:baseline")
    prior = TheoryPosterior((("baseline", 1.0),))
    predictions = (
        PredictiveDistribution(
            "baseline",
            request_id,
            (((0,), 1.0), ((1,), 0.0)),
        ),
    )
    anomaly = PopulationAnomaly(
        Observation(request_id, (1,)),
        0.0,
        (("baseline", 0.0),),
    )
    return TheorySpaceExpansionRequest.from_population_anomaly(
        request_id="expand-" + request_id,
        anomaly=anomaly,
        active_theories=(theory,),
        prior=prior,
        predictions=predictions,
    )


def training_examples(prefix: str) -> tuple[BooleanMechanismExample, ...]:
    # Target: Y = M and not C. Keep (0, 1) held out for qualification.
    return (
        BooleanMechanismExample((False, False), False, f"{prefix}-00"),
        BooleanMechanismExample((True, False), True, f"{prefix}-10"),
        BooleanMechanismExample((True, True), False, f"{prefix}-11"),
    )


def full_examples(prefix: str) -> tuple[BooleanMechanismExample, ...]:
    return (
        BooleanMechanismExample((False, False), False, f"{prefix}-00"),
        BooleanMechanismExample((False, True), False, f"{prefix}-01"),
        BooleanMechanismExample((True, False), True, f"{prefix}-10"),
        BooleanMechanismExample((True, True), False, f"{prefix}-11"),
    )


def heldout() -> tuple[ExperimentSpec, Observation]:
    experiment = ExperimentSpec(
        "source-heldout-01",
        ExperimentKind.INTERVENTIONAL,
        (
            Intervention("M", 0, object_id=0),
            Intervention("C", 1, object_id=0),
        ),
        ("Y",),
        object_id=0,
    )
    return experiment, Observation(experiment.experiment_id, (0,))


def source_mechanism(tmp_path: Path):
    engine = DreamCoderRepairEngine(
        training_examples("source"),
        cache_root=tmp_path / "source-dreamcoder",
        primitives=("not", "and", "or"),
        maximum_mdl=10.0,
        maximum_programs=100_000,
    )
    run = RepairCoordinator().expand(
        request("source-anomaly"),
        (engine,),
    )
    assert len(run.proposals) == 1

    qualification = ProgramMechanismProjector(engine).qualify(
        run.proposals[0],
        parent_theory=TheoryRef("baseline", "artifact:baseline"),
        input_variables=("M", "C"),
        output_variable="Y",
        heldout=(heldout(),),
    )
    assert isinstance(qualification, QualifiedProgramMechanism)
    return engine, qualification


def admitted_program(
    qualification: QualifiedProgramMechanism,
) -> MechanismRecord:
    return MechanismRecord(
        mechanism_id="and-not-c",
        version=1,
        kind=MechanismKind.PROGRAM,
        status=MechanismStatus.ADMITTED,
        representation_artifact_id=qualification.theory.artifact_id,
        scope=MechanismScope(
            context_ids=("factory-a", "factory-b"),
            regime_ids=("steady",),
            domain_ids=("boolean-control",),
            task_ids=("predict-Y",),
            required_assumptions=("deterministic", "binary-inputs"),
        ),
        transfer=MechanismTransferStatus.SHARED,
        supporting_evidence_ids=qualification.heldout_experiment_ids,
        provenance_artifact_ids=(
            qualification.source_program_artifact_id,
        ),
    )


def runtime_context(context_id: str) -> ApplicabilityContext:
    return ApplicabilityContext(
        context_id=context_id,
        regime_id="steady",
        domain_id="boolean-control",
        task_id="predict-Y",
        assumptions=("deterministic", "binary-inputs"),
    )


def test_runtime_projection_keeps_mechanism_kinds_typed(tmp_path: Path) -> None:
    repository = MLMDMechanismRepository.sqlite(
        tmp_path / "runtime-views.sqlite"
    )
    scope = MechanismScope(
        context_ids=("factory-a",),
        regime_ids=("steady",),
    )
    for kind, mechanism_id, artifact in (
        (MechanismKind.CAUSAL_MODEL, "causal", "bcs:model:1"),
        (MechanismKind.STATE_FEATURE, "state", "ttt-state:1"),
        (MechanismKind.SYMBOLIC_RULE, "rule", "pddl:operator:1"),
    ):
        repository.admit(
            MechanismRecord(
                mechanism_id=mechanism_id,
                version=1,
                kind=kind,
                status=MechanismStatus.ADMITTED,
                representation_artifact_id=artifact,
                scope=scope,
                transfer=MechanismTransferStatus.SHARED,
            )
        )

    projector = MechanismRuntimeProjector(repository)
    context = ApplicabilityContext(
        context_id="factory-a",
        regime_id="steady",
    )

    assert tuple(
        item.mechanism.mechanism_id
        for item in projector.causal_models(context).entries
    ) == ("causal",)
    assert tuple(
        item.mechanism.mechanism_id
        for item in projector.state_features(context).entries
    ) == ("state",)
    assert tuple(
        item.mechanism.mechanism_id
        for item in projector.symbolic_rules(context).entries
    ) == ("rule",)


def test_admitted_program_is_reused_as_real_dreamcoder_invention(
    tmp_path: Path,
) -> None:
    source_engine, qualification = source_mechanism(tmp_path)
    repository = MLMDMechanismRepository.sqlite(
        tmp_path / "mechanisms.sqlite"
    )
    mechanism = admitted_program(qualification)
    repository.admit(mechanism)

    projection = DreamCoderGrammarProjector(
        repository,
        source_engine,
    ).project(runtime_context("factory-b"))

    assert tuple(entry.mechanism for entry in projection.entries) == (
        mechanism.ref,
    )
    assert len(projection.programs) == 1
    assert projection.entries[0].source_program_artifact_id == (
        qualification.source_program_artifact_id
    )

    cold = DreamCoderRepairEngine(
        full_examples("cold"),
        cache_root=tmp_path / "cold",
        primitives=("not", "and", "or"),
        maximum_mdl=10.0,
        maximum_programs=100_000,
    )
    warm = DreamCoderRepairEngine(
        full_examples("warm"),
        cache_root=tmp_path / "warm",
        primitives=("not", "and", "or"),
        library_programs=projection.programs,
        maximum_mdl=10.0,
        maximum_programs=100_000,
    )

    cold_run = RepairCoordinator().expand(
        request("cold-anomaly"),
        (cold,),
    )
    warm_run = RepairCoordinator().expand(
        request("warm-anomaly"),
        (warm,),
    )
    assert len(cold_run.proposals) == 1
    assert len(warm_run.proposals) == 1

    cold_artifact = cold.load_artifact(
        cold_run.proposals[0].artifact_id
    )
    warm_artifact = warm.load_artifact(
        warm_run.proposals[0].artifact_id
    )

    assert warm_artifact["library_programs"] == list(projection.programs)
    assert warm_artifact["programs_enumerated"] < (
        cold_artifact["programs_enumerated"]
    )
