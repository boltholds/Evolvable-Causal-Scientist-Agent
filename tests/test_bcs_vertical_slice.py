from bcs.calibration import BooleanCore
from bcs.inference import History

from ecsa.adapters.bcs import BCSInferenceAdapter
from ecsa.contracts import (
    ExperimentKind,
    ExperimentSpec,
    Intervention,
    Observation,
    PopulationAnomaly,
    PosteriorUpdate,
    PredictiveDistribution,
    TheoryPosterior,
    TheoryRef,
)
from ecsa.science import ScienceKernel


def theory_models():
    r = (0, 0, 0, 0, 0, 0)
    m = (0, 0)
    button_only = BooleanCore(
        r,
        m,
        (0, 1, 0, 1),
    )  # Reachable M=0 states implement Y := C.
    never_opens = BooleanCore(
        r,
        m,
        (0, 0, 0, 0),
    )  # Y := 0.
    theories = (
        TheoryRef("button_only", button_only.artifact_id),
        TheoryRef("never_opens", never_opens.artifact_id),
    )
    return theories, {
        "button_only": button_only,
        "never_opens": never_opens,
    }


def experiment(experiment_id: str, c: int) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=experiment_id,
        kind=ExperimentKind.INTERVENTIONAL,
        interventions=(Intervention("C", c, object_id=0),),
        outcomes=("Y",),
        object_id=0,
    )


def complete_predictions(adapter, theories, spec):
    results = tuple(
        adapter.predict(theory, spec)
        for theory in theories
    )
    assert all(
        isinstance(result, PredictiveDistribution)
        for result in results
    )
    return results


def test_first_vertical_slice_selects_intervention_and_updates_posterior() -> None:
    theories, models = theory_models()
    posterior = TheoryPosterior.uniform(theories)
    adapter = BCSInferenceAdapter(models, History((), ()))
    kernel = ScienceKernel()

    c0 = complete_predictions(
        adapter,
        theories,
        experiment("clamp-c0", 0),
    )
    c1 = complete_predictions(
        adapter,
        theories,
        experiment("clamp-c1", 1),
    )

    chosen = kernel.select_experiment(
        posterior,
        (c0, c1),
    )
    assert chosen.experiment_id == "clamp-c1"

    result = kernel.update(
        posterior,
        c1,
        Observation("clamp-c1", (0,)),
    )
    assert isinstance(result, PosteriorUpdate)
    assert result.posterior.probability("never_opens") == 1.0
    assert result.posterior.probability("button_only") == 0.0


def test_bcs_population_anomaly_is_explicit() -> None:
    theories, models = theory_models()
    posterior = TheoryPosterior.uniform(theories)
    adapter = BCSInferenceAdapter(models, History((), ()))
    predictions = complete_predictions(
        adapter,
        theories,
        experiment("clamp-c0", 0),
    )

    result = ScienceKernel().update(
        posterior,
        predictions,
        Observation("clamp-c0", (1,)),
    )

    assert isinstance(result, PopulationAnomaly)
    assert result.evidence_probability == 0.0
