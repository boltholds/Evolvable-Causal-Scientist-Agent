"""Minimal ECSA vertical slice using two real Behavioral-Causal-State models."""

from bcs.calibration import BooleanCore
from bcs.inference import History

from ecsa.adapters.bcs import BCSInferenceAdapter
from ecsa.contracts import (
    ExperimentKind,
    ExperimentSpec,
    Intervention,
    Observation,
    PredictiveDistribution,
    TheoryPosterior,
    TheoryRef,
)
from ecsa.science import ScienceKernel


def main() -> None:
    r = (0, 0, 0, 0, 0, 0)
    m = (0, 0)
    models = {
        "button_only": BooleanCore(r, m, (0, 1, 0, 1)),
        "never_opens": BooleanCore(r, m, (0, 0, 0, 0)),
    }
    theories = tuple(
        TheoryRef(name, model.artifact_id)
        for name, model in models.items()
    )
    posterior = TheoryPosterior.uniform(theories)
    adapter = BCSInferenceAdapter(models, History((), ()))
    kernel = ScienceKernel()

    experiments = tuple(
        ExperimentSpec(
            f"clamp-c{c}",
            ExperimentKind.INTERVENTIONAL,
            (Intervention("C", c),),
            ("Y",),
        )
        for c in (0, 1)
    )

    candidate_predictions = []
    for spec in experiments:
        predictions = tuple(
            adapter.predict(theory, spec)
            for theory in theories
        )
        if not all(
            isinstance(result, PredictiveDistribution)
            for result in predictions
        ):
            raise RuntimeError(
                f"inference unavailable for {spec.experiment_id}: {predictions}"
            )
        candidate_predictions.append(predictions)

    chosen = kernel.select_experiment(
        posterior,
        tuple(candidate_predictions),
    )
    print("selected:", chosen)

    selected_predictions = next(
        predictions
        for predictions in candidate_predictions
        if predictions[0].experiment_id == chosen.experiment_id
    )

    observation = Observation(
        chosen.experiment_id,
        (0,),
    )
    result = kernel.update(
        posterior,
        selected_predictions,
        observation,
    )

    print("observation:", observation)
    print("update:", result)


if __name__ == "__main__":
    main()
