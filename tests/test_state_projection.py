from pathlib import Path

from bcs.calibration import BooleanCore
from bcs.inference import History

from ecsa.adapters.bcs import BCSInferenceAdapter
from ecsa.adapters.ttt import TTTRepairEngine
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
from ecsa.state_projection import (
    QualifiedStateAugmentation,
    StateAlignmentEvidence,
    TTTStatePredictionAdapter,
    TTTStateProjector,
)


def dist(theory: str, experiment: str, p_zero: float) -> PredictiveDistribution:
    return PredictiveDistribution(
        theory,
        experiment,
        (((0,), p_zero), ((1,), 1.0 - p_zero)),
    )


def request() -> TheorySpaceExpansionRequest:
    theories = (TheoryRef("h1", "a1"), TheoryRef("h2", "a2"))
    posterior = TheoryPosterior((("h1", 0.5), ("h2", 0.5)))
    predictions = (
        dist("h1", "anomaly", 1.0),
        dist("h2", "anomaly", 1.0),
    )
    anomaly = PopulationAnomaly(
        Observation("anomaly", (1,)),
        0.0,
        (("h1", 0.0), ("h2", 0.0)),
    )
    return TheorySpaceExpansionRequest.from_population_anomaly(
        request_id="expand-state",
        anomaly=anomaly,
        active_theories=theories,
        prior=posterior,
        predictions=predictions,
    )


def experiment(experiment_id: str, c: int) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id,
        ExperimentKind.INTERVENTIONAL,
        (Intervention("C", c, object_id=0),),
        ("Y",),
        object_id=0,
    )


def test_ttt_projection_can_qualify_aliasing_and_reenter_shared_prediction_space(
    tmp_path: Path,
) -> None:
    engine = TTTRepairEngine(
        cache_root=tmp_path,
        fixture_id="C00-commands_c",
    )
    run = RepairCoordinator().expand(request(), (engine,))
    proposal = run.proposals[0]

    projector = TTTStateProjector(engine)
    parent = TheoryRef("h1", "a1")
    candidate = projector.project(proposal, parent)

    alignments = tuple(
        StateAlignmentEvidence(
            causal_state_key="deliberately-coarse-state",
            input_word=(symbol,),
        )
        for symbol in range(candidate.alphabet_size)
    )
    qualification = projector.qualify(candidate, alignments)

    assert isinstance(qualification, QualifiedStateAugmentation)
    assert qualification.witness.left_state != qualification.witness.right_state
    assert qualification.theory.theory_id.startswith("h1+ttt-state:")
    assert qualification.theory.artifact_id.startswith(
        "state-augmented:sha256:"
    )

    admitted = ScienceKernel().admit_theory(
        TheoryPosterior((("h1", 0.5), ("h2", 0.5))),
        qualification.theory,
        prior_mass=0.05,
    )
    assert admitted.probability(qualification.theory.theory_id) == 0.05

    r = (0, 0, 0, 0, 0, 0)
    m = (0, 0)
    models = {
        "h1": BooleanCore(r, m, (0, 1, 0, 1)),
        "h2": BooleanCore(r, m, (0, 0, 0, 0)),
    }
    bcs = BCSInferenceAdapter(models, History((), ()))
    ttt = TTTStatePredictionAdapter(projector, qualification)

    candidates = []
    for spec in (experiment("c0", 0), experiment("c1", 1)):
        predictions = (
            bcs.predict(TheoryRef("h1", models["h1"].artifact_id), spec),
            bcs.predict(TheoryRef("h2", models["h2"].artifact_id), spec),
            ttt.predict(qualification.theory, spec),
        )
        assert all(
            isinstance(prediction, PredictiveDistribution)
            for prediction in predictions
        )
        candidates.append(predictions)

    chosen = ScienceKernel().select_experiment(
        admitted,
        tuple(candidates),
    )
    assert chosen.experiment_id == "c1"
    assert chosen.information_gain_bits > 0.0
