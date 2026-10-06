from __future__ import annotations

from collections.abc import Mapping

from ecsa.contracts import (
    ExperimentKind,
    ExperimentSpec,
    PredictionResult,
    PredictionStatus,
    PredictionUnavailable,
    PredictiveDistribution,
    TheoryRef,
)

try:
    from bcs.dynamics import Dynamics
    from bcs.inference import (
        ExactDistribution,
        History,
        InferenceIncomplete,
        InferenceUndefined,
        Interventional,
        evaluate,
    )
    from bcs.simulator import Clamp, Variable
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "BCS integration requires the optional 'bcs' dependency: "
        "pip install -e '.[bcs]'"
    ) from exc


class BCSInferenceAdapter:
    """Thin bridge from ECSA experiment contracts to existing BCS inference."""

    def __init__(
        self,
        models: Mapping[str, Dynamics],
        history: History,
        *,
        budget: int = 1_000_000,
    ) -> None:
        self._models = dict(models)
        self._history = history
        self._budget = budget

    def predict(
        self,
        theory: TheoryRef,
        experiment: ExperimentSpec,
    ) -> PredictionResult:
        if experiment.kind is not ExperimentKind.INTERVENTIONAL:
            raise ValueError(
                "BCS adapter currently supports interventional experiments only"
            )
        try:
            model = self._models[theory.theory_id]
        except KeyError as exc:
            raise KeyError(f"unknown theory: {theory.theory_id}") from exc

        model_artifact_id = getattr(model, "artifact_id", None)
        if (
            model_artifact_id is not None
            and model_artifact_id != theory.artifact_id
        ):
            raise ValueError(
                "theory artifact_id does not match registered BCS model"
            )

        clamps = tuple(
            Clamp(
                intervention.object_id,
                Variable(intervention.variable),
                self._binary_value(intervention.value),
            )
            for intervention in experiment.interventions
        )
        query = Interventional(
            experiment.object_id,
            clamps,
            tuple(Variable(name) for name in experiment.outcomes),
        )
        result = evaluate(
            model,
            self._history,
            query,
            budget=self._budget,
        )

        if isinstance(result, ExactDistribution):
            return PredictiveDistribution(
                theory.theory_id,
                experiment.experiment_id,
                tuple(
                    (tuple(outcome), float(probability))
                    for outcome, probability in result.probabilities
                ),
            )
        if isinstance(result, InferenceUndefined):
            return PredictionUnavailable(
                theory.theory_id,
                experiment.experiment_id,
                PredictionStatus.UNDEFINED,
                result.reason,
            )
        if isinstance(result, InferenceIncomplete):
            return PredictionUnavailable(
                theory.theory_id,
                experiment.experiment_id,
                PredictionStatus.INCOMPLETE,
                result.reason,
            )
        raise TypeError(
            f"unsupported BCS inference result: {type(result).__name__}"
        )

    @staticmethod
    def _binary_value(value: object) -> int:
        if type(value) is not int or value not in (0, 1):
            raise ValueError(
                "BCS interventions require binary integer values"
            )
        return value
