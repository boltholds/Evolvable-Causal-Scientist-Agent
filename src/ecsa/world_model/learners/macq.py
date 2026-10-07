from __future__ import annotations

import importlib.util
from hashlib import sha256

from .base import (
    AcquisitionTrace,
    ActionModelProposal,
    LearnerFailure,
)
from ..hypotheses import (
    ActionSchemaHypothesis,
    EntityTypeHypothesis,
    HypothesisStatus,
    PredicateHypothesis,
    WorldContractHypothesis,
)


def _stable_id(prefix: str, value: str) -> str:
    digest = sha256(value.encode()).hexdigest()[:20]
    return f"{prefix}:{digest}"


class MacqLocmLearner:
    learner_id = "macq-locm"

    def update(
        self,
        traces: tuple[AcquisitionTrace, ...],
        current_contracts: tuple[WorldContractHypothesis, ...],
    ) -> ActionModelProposal | LearnerFailure:
        del current_contracts
        if importlib.util.find_spec("macq") is None:
            return LearnerFailure(
                self.learner_id,
                "macq is not installed; install the world-model-planning extra",
            )
        if len(traces) != 1:
            return LearnerFailure(
                self.learner_id,
                "LOCM backend currently requires exactly one acquisition trace",
            )
        trace = traces[0]
        if len(trace.steps) < 2:
            return LearnerFailure(
                self.learner_id,
                "LOCM trace requires at least two actions",
            )

        try:
            model = self._extract(trace)
            return self._translate(trace, model)
        except Exception as exc:
            return LearnerFailure(
                self.learner_id,
                f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _extract(trace: AcquisitionTrace):
        from macq.extract import Extract, modes
        from macq.observation import ActionObservation
        from macq.trace import (
            Action,
            PlanningObject,
            State,
            Step,
            Trace,
            TraceList,
        )

        objects: dict[str, PlanningObject] = {}

        def planning_object(ref: str) -> PlanningObject:
            value = objects.get(ref)
            if value is None:
                value = PlanningObject("entity", ref)
                objects[ref] = value
            return value

        steps = [
            Step(
                State({}),
                Action(
                    step.schema_id,
                    [planning_object(ref) for ref in step.object_refs],
                ),
                index,
            )
            for index, step in enumerate(trace.steps, start=1)
        ]
        steps.append(Step(State({}), None, len(steps) + 1))
        observed = TraceList([Trace(steps)]).tokenize(ActionObservation)
        return Extract(
            observed,
            modes.LOCM,
            statics={},
            viz=False,
            view=False,
        )

    def _translate(
        self,
        trace: AcquisitionTrace,
        model,
    ) -> ActionModelProposal | LearnerFailure:
        evidence = tuple(
            dict.fromkeys(step.evidence_id for step in trace.steps)
        )
        learned_actions = tuple(
            sorted(model.actions, key=lambda value: value.name)
        )
        if not learned_actions:
            return LearnerFailure(
                self.learner_id,
                "LOCM produced no learned actions",
            )

        sorts = {
            sort
            for action in learned_actions
            for sort in action.param_sorts
        }
        for fluent in model.fluents:
            sorts.update(fluent.param_sorts)

        type_ids = {
            sort: _stable_id("macq-type", sort)
            for sort in sorted(sorts)
        }
        entity_types = tuple(
            EntityTypeHypothesis(
                hypothesis_id=type_ids[sort],
                member_refs=(),
                supporting_evidence_ids=evidence,
                contradicting_evidence_ids=(),
                confidence=0.5,
                status=HypothesisStatus.PROPOSED,
            )
            for sort in sorted(sorts)
        )

        fluent_key_to_id: dict[tuple[str, tuple[str, ...]], str] = {}
        predicate_values: list[PredicateHypothesis] = []
        for fluent in sorted(
            model.fluents,
            key=lambda value: (
                value.name,
                tuple(value.param_sorts),
            ),
        ):
            key = (fluent.name, tuple(fluent.param_sorts))
            predicate_id = _stable_id(
                "macq-predicate",
                repr(key),
            )
            fluent_key_to_id[key] = predicate_id
            predicate_values.append(
                PredicateHypothesis(
                    hypothesis_id=predicate_id,
                    argument_type_ids=tuple(
                        type_ids[sort]
                        for sort in fluent.param_sorts
                    ),
                    symmetric=False,
                    supporting_evidence_ids=evidence,
                    contradicting_evidence_ids=(),
                    confidence=0.5,
                    status=HypothesisStatus.PROPOSED,
                )
            )

        def predicate_ids(values) -> tuple[str, ...]:
            ids: list[str] = []
            for fluent in values:
                key = (fluent.name, tuple(fluent.param_sorts))
                predicate_id = fluent_key_to_id.get(key)
                if predicate_id is None:
                    predicate_id = _stable_id(
                        "macq-predicate",
                        repr(key),
                    )
                    fluent_key_to_id[key] = predicate_id
                    predicate_values.append(
                        PredicateHypothesis(
                            hypothesis_id=predicate_id,
                            argument_type_ids=tuple(
                                type_ids[sort]
                                for sort in fluent.param_sorts
                            ),
                            symmetric=False,
                            supporting_evidence_ids=evidence,
                            contradicting_evidence_ids=(),
                            confidence=0.5,
                            status=HypothesisStatus.PROPOSED,
                        )
                    )
                ids.append(predicate_id)
            return tuple(sorted(set(ids)))

        actions = tuple(
            ActionSchemaHypothesis(
                hypothesis_id=_stable_id(
                    "macq-action",
                    f"{action.name}:{tuple(action.param_sorts)!r}",
                ),
                source_schema_id=action.name,
                parameter_type_ids=tuple(
                    type_ids[sort]
                    for sort in action.param_sorts
                ),
                precondition_predicate_ids=predicate_ids(action.precond),
                add_effect_predicate_ids=predicate_ids(action.add),
                delete_effect_predicate_ids=predicate_ids(action.delete),
                numeric_effect_ids=(),
                symmetric_parameter_groups=(),
                supporting_evidence_ids=evidence,
                contradicting_evidence_ids=(),
                confidence=0.5,
                status=HypothesisStatus.PROPOSED,
            )
            for action in learned_actions
        )

        return ActionModelProposal(
            learner_id=self.learner_id,
            entity_types=entity_types,
            predicates=tuple(
                sorted(
                    predicate_values,
                    key=lambda value: value.hypothesis_id,
                )
            ),
            actions=actions,
            supporting_evidence_ids=evidence,
        )
