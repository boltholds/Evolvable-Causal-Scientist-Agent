from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

from ..hypotheses import (
    ActionSchemaHypothesis,
    EntityTypeHypothesis,
    PredicateHypothesis,
    WorldContractHypothesis,
)


@dataclass(frozen=True)
class AcquisitionActionStep:
    schema_id: str
    object_refs: tuple[str, ...]
    evidence_id: str

    def __post_init__(self) -> None:
        if not self.schema_id or not self.evidence_id:
            raise ValueError("schema_id and evidence_id are required")
        if (
            not isinstance(self.object_refs, tuple)
            or not all(
                isinstance(value, str) and value
                for value in self.object_refs
            )
        ):
            raise ValueError("object refs must be immutable nonempty strings")


@dataclass(frozen=True)
class AcquisitionTrace:
    trace_id: str
    steps: tuple[AcquisitionActionStep, ...]

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id is required")
        if (
            not isinstance(self.steps, tuple)
            or not self.steps
            or not all(
                isinstance(step, AcquisitionActionStep)
                for step in self.steps
            )
        ):
            raise ValueError("trace steps must be nonempty immutable actions")


@dataclass(frozen=True)
class ActionModelProposal:
    learner_id: str
    entity_types: tuple[EntityTypeHypothesis, ...]
    predicates: tuple[PredicateHypothesis, ...]
    actions: tuple[ActionSchemaHypothesis, ...]
    supporting_evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.learner_id:
            raise ValueError("learner_id is required")
        typed = (
            (self.entity_types, EntityTypeHypothesis, "entity types"),
            (self.predicates, PredicateHypothesis, "predicates"),
            (self.actions, ActionSchemaHypothesis, "actions"),
        )
        for values, cls, label in typed:
            if (
                not isinstance(values, tuple)
                or not all(isinstance(value, cls) for value in values)
            ):
                raise ValueError(f"{label} must be immutable typed values")
        if (
            not isinstance(self.supporting_evidence_ids, tuple)
            or not all(
                isinstance(value, str) and value
                for value in self.supporting_evidence_ids
            )
            or len(set(self.supporting_evidence_ids))
            != len(self.supporting_evidence_ids)
        ):
            raise ValueError(
                "supporting evidence ids must be unique immutable strings"
            )


@dataclass(frozen=True)
class LearnerFailure:
    learner_id: str
    reason: str

    def __post_init__(self) -> None:
        if not self.learner_id or not self.reason:
            raise ValueError("learner_id and reason are required")


ActionModelResult: TypeAlias = ActionModelProposal | LearnerFailure


@runtime_checkable
class ActionModelLearner(Protocol):
    learner_id: str

    def update(
        self,
        traces: tuple[AcquisitionTrace, ...],
        current_contracts: tuple[WorldContractHypothesis, ...],
    ) -> ActionModelResult: ...



def proposal_to_world_contract(
    proposal: ActionModelProposal,
) -> WorldContractHypothesis:
    if not isinstance(proposal, ActionModelProposal):
        raise TypeError("proposal must be ActionModelProposal")

    from ..canonical import canonicalize_world_contract
    from ..hypotheses import HypothesisStatus

    provisional = WorldContractHypothesis(
        contract_id=f"proposal:{proposal.learner_id}",
        entity_types=proposal.entity_types,
        predicates=proposal.predicates,
        numeric_fluents=(),
        argument_roles=(),
        actions=proposal.actions,
        affordances=(),
        supporting_evidence_ids=proposal.supporting_evidence_ids,
        contradicting_evidence_ids=(),
        confidence=0.5,
        status=HypothesisStatus.PROPOSED,
    )
    fingerprint = canonicalize_world_contract(provisional).fingerprint
    return WorldContractHypothesis(
        contract_id=f"world-contract:{fingerprint}",
        entity_types=proposal.entity_types,
        predicates=proposal.predicates,
        numeric_fluents=(),
        argument_roles=(),
        actions=proposal.actions,
        affordances=(),
        supporting_evidence_ids=proposal.supporting_evidence_ids,
        contradicting_evidence_ids=(),
        confidence=0.5,
        status=HypothesisStatus.PROPOSED,
    )
