from pathlib import Path

from bcs.calibration import BooleanCore
from bcs.inference import History

from ecsa.adapters.bcs import BCSInferenceAdapter
from ecsa.adapters.dreamcoder import (
    BooleanMechanismExample,
    DreamCoderProgramPredictionAdapter,
    DreamCoderRepairEngine,
    ProgramMechanismProjector,
    QualifiedProgramMechanism,
    RejectedProgramMechanism,
)
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
from ecsa.repair import RepairCoordinator, TheorySpaceExpansionRequest
from ecsa.science import ScienceKernel


def request() -> TheorySpaceExpansionRequest:
    parent = TheoryRef("zero-world", "artifact:zero-world")
    prior = TheoryPosterior((("zero-world", 1.0),))
    predictions = (
        PredictiveDistribution(
            "zero-world",
            "anomaly-program",
            (((0,), 1.0), ((1,), 0.0)),
        ),
    )
    anomaly = PopulationAnomaly(
        Observation("anomaly-program", (1,)),
        0.0,
        (("zero-world", 0.0),),
    )
    return TheorySpaceExpansionRequest.from_population_anomaly(
        request_id="expand-program-prediction",
        anomaly=anomaly,
        active_theories=(parent,),
        prior=prior,
        predictions=predictions,
    )


def training_examples() -> tuple[BooleanMechanismExample, ...]:
    # Target mechanism: Y = M and not C.
    # (M=0,C=1) is deliberately held out.
    return (
        BooleanMechanismExample((False, False), False, "train-00"),
        BooleanMechanismExample((True, False), True, "train-10"),
        BooleanMechanismExample((True, True), False, "train-11"),
    )


def experiment(experiment_id: str, m: int, c: int) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id,
        ExperimentKind.INTERVENTIONAL,
        (
            Intervention("M", m, object_id=0),
            Intervention("C", c, object_id=0),
        ),
        ("Y",),
        object_id=0,
    )


def synthesize(tmp_path: Path):
    engine = DreamCoderRepairEngine(
        training_examples(),
        cache_root=tmp_path,
        primitives=("not", "and", "or"),
        maximum_mdl=10.0,
        maximum_programs=100_000,
    )
    run = RepairCoordinator().expand(request(), (engine,))
    assert len(run.proposals) == 1
    return engine, run.proposals[0]


def test_dreamcoder_program_passes_heldout_validation_is_admitted_and_enters_ids(
    tmp_path: Path,
) -> None:
    engine, proposal = synthesize(tmp_path)
    adapter = DreamCoderProgramPredictionAdapter(
        engine,
        proposal,
        input_variables=("M", "C"),
        output_variable="Y",
    )

    heldout = experiment("heldout-01", 0, 1)
    prediction = adapter.predict(
        TheoryRef("candidate", proposal.artifact_id),
        heldout,
    )
    assert isinstance(prediction, PredictiveDistribution)
    assert prediction.probability((0,)) == 1.0

    projector = ProgramMechanismProjector(engine)
    qualification = projector.qualify(
        proposal,
        parent_theory=TheoryRef("zero-world", "artifact:zero-world"),
        input_variables=("M", "C"),
        output_variable="Y",
        heldout=((heldout, Observation("heldout-01", (0,))),),
    )
    assert isinstance(qualification, QualifiedProgramMechanism)
    assert qualification.theory.theory_id.startswith(
        "zero-world+dreamcoder-program:"
    )
    assert qualification.theory.artifact_id.startswith(
        "program-mechanism:sha256:"
    )

    posterior = ScienceKernel().admit_theory(
        TheoryPosterior((("zero-world", 1.0),)),
        qualification.theory,
        prior_mass=0.05,
    )
    assert posterior.probability(qualification.theory.theory_id) == 0.05

    zero_core = BooleanCore(
        (0, 0, 0, 0, 0, 0),
        (0, 0),
        (0, 0, 0, 0),
    )
    bcs = BCSInferenceAdapter(
        {"zero-world": zero_core},
        History((), ()),
    )
    program = qualification.prediction_adapter(engine)

    same = experiment("same", 0, 0)
    split = experiment("split", 1, 0)
    candidates = []
    for spec in (same, split):
        predictions = (
            bcs.predict(
                TheoryRef("zero-world", zero_core.artifact_id),
                spec,
            ),
            program.predict(qualification.theory, spec),
        )
        assert all(
            isinstance(value, PredictiveDistribution)
            for value in predictions
        )
        candidates.append(predictions)

    chosen = ScienceKernel().select_experiment(
        posterior,
        tuple(candidates),
    )
    assert chosen.experiment_id == "split"
    assert chosen.information_gain_bits > 0.0


def test_dreamcoder_program_is_rejected_when_heldout_prediction_is_wrong(
    tmp_path: Path,
) -> None:
    engine, proposal = synthesize(tmp_path)
    heldout = experiment("heldout-01", 0, 1)

    result = ProgramMechanismProjector(engine).qualify(
        proposal,
        parent_theory=TheoryRef("zero-world", "artifact:zero-world"),
        input_variables=("M", "C"),
        output_variable="Y",
        heldout=((heldout, Observation("heldout-01", (1,))),),
    )

    assert isinstance(result, RejectedProgramMechanism)
    assert "held-out" in result.reason
