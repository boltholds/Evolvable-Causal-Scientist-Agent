from __future__ import annotations

from dataclasses import dataclass, replace

from .grounding import GroundingUpdate
from .hypotheses import (
    HypothesisStatus,
    WorldContractHypothesis,
)
from .memory import WorldContractStore


@dataclass(frozen=True)
class WorldModelUpdate:
    new_evidence_ids: tuple[str, ...] = ()
    added_contract_ids: tuple[str, ...] = ()
    updated_contract_ids: tuple[str, ...] = ()
    rejected_contract_ids: tuple[str, ...] = ()
    contracts: tuple[WorldContractHypothesis, ...] = ()


class WorldModelAcquisitionKernel:
    def __init__(
        self,
        *,
        store: WorldContractStore | None = None,
        max_contracts: int = 64,
    ) -> None:
        if type(max_contracts) is not int or max_contracts < 1:
            raise ValueError("max_contracts must be positive")
        self.store = store
        self.max_contracts = max_contracts
        self._contracts: dict[str, WorldContractHypothesis] = {}
        self._grounding_evidence_ids: list[str] = []

    def observe_grounding(
        self,
        update: GroundingUpdate,
    ) -> WorldModelUpdate:
        if not isinstance(update, GroundingUpdate):
            raise TypeError("observe_grounding requires GroundingUpdate")
        evidence: list[str] = []
        evidence.extend(
            delta.evidence_id
            for delta in update.transition.feature_deltas
        )
        evidence.extend(
            item.evidence_id
            for item in update.transition.participation
        )
        evidence.append(
            update.transition.outcome_evidence.evidence_id
        )
        for hypothesis in update.identity_hypotheses:
            evidence.extend(hypothesis.evidence_ids)
        unique_new = tuple(
            evidence_id
            for evidence_id in dict.fromkeys(evidence)
            if evidence_id not in self._grounding_evidence_ids
        )
        self._grounding_evidence_ids.extend(unique_new)
        return WorldModelUpdate(
            new_evidence_ids=unique_new,
            contracts=self.contract_hypotheses(),
        )

    def ingest_proposals(
        self,
        proposals: tuple[WorldContractHypothesis, ...],
    ) -> WorldModelUpdate:
        if not isinstance(proposals, tuple) or not all(
            isinstance(value, WorldContractHypothesis)
            for value in proposals
        ):
            raise TypeError(
                "proposals must be an immutable tuple of world contracts"
            )
        added: list[str] = []
        updated: list[str] = []
        for proposal in proposals:
            if proposal.contract_id in self._contracts:
                updated.append(proposal.contract_id)
            else:
                added.append(proposal.contract_id)
            self._contracts[proposal.contract_id] = proposal

        if len(self._contracts) > self.max_contracts:
            retained = sorted(
                self._contracts.values(),
                key=lambda value: (
                    -float(value.confidence),
                    value.contract_id,
                ),
            )[: self.max_contracts]
            self._contracts = {
                value.contract_id: value
                for value in retained
            }

        retained_ids = set(self._contracts)
        return WorldModelUpdate(
            added_contract_ids=tuple(
                value for value in added if value in retained_ids
            ),
            updated_contract_ids=tuple(
                value for value in updated if value in retained_ids
            ),
            contracts=self.contract_hypotheses(),
        )

    def record_contract_evidence(
        self,
        contract_id: str,
        *,
        evidence_id: str,
        supports: bool,
    ) -> WorldModelUpdate:
        if not contract_id or contract_id not in self._contracts:
            raise KeyError(f"unknown world contract: {contract_id}")
        if not evidence_id:
            raise ValueError("evidence_id is required")
        if type(supports) is not bool:
            raise ValueError("supports must be bool")

        current = self._contracts[contract_id]
        if (
            evidence_id in current.supporting_evidence_ids
            or evidence_id in current.contradicting_evidence_ids
        ):
            return WorldModelUpdate(
                contracts=self.contract_hypotheses(),
            )

        if supports:
            confidence = min(
                1.0,
                float(current.confidence) + 0.1,
            )
            status = (
                HypothesisStatus.SUPPORTED
                if current.status is HypothesisStatus.PROPOSED
                and confidence >= 0.5
                else current.status
            )
            revised = replace(
                current,
                supporting_evidence_ids=(
                    *current.supporting_evidence_ids,
                    evidence_id,
                ),
                confidence=confidence,
                status=status,
            )
            rejected = ()
        else:
            confidence = max(
                0.0,
                float(current.confidence) - 0.25,
            )
            status = (
                HypothesisStatus.REJECTED
                if confidence <= 0.05
                else current.status
            )
            revised = replace(
                current,
                contradicting_evidence_ids=(
                    *current.contradicting_evidence_ids,
                    evidence_id,
                ),
                confidence=confidence,
                status=status,
            )
            rejected = (
                (contract_id,)
                if status is HypothesisStatus.REJECTED
                else ()
            )

        self._contracts[contract_id] = revised
        return WorldModelUpdate(
            new_evidence_ids=(evidence_id,),
            updated_contract_ids=(contract_id,),
            rejected_contract_ids=rejected,
            contracts=self.contract_hypotheses(),
        )

    def contract_hypotheses(
        self,
    ) -> tuple[WorldContractHypothesis, ...]:
        return tuple(
            sorted(
                self._contracts.values(),
                key=lambda value: value.contract_id,
            )
        )
