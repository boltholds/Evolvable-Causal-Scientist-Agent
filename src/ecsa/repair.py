from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias

from .contracts import (
    PopulationAnomaly,
    PredictiveDistribution,
    TheoryPosterior,
    TheoryRef,
)


class RepairFamily(StrEnum):
    STATE_MEMORY = "state_memory"
    PARAMETER_MODE = "parameter_mode"
    REGIME_CHANGE = "regime_change"
    FAULT = "fault"
    CAUSAL_STRUCTURE = "causal_structure"
    PROGRAM_MECHANISM = "program_mechanism"
    ABSTRACTION_TRANSFER = "abstraction_transfer"
    LATENT_STATE = "latent_state"


@dataclass(frozen=True)
class TheorySpaceExpansionRequest:
    request_id: str
    anomaly: PopulationAnomaly
    active_theories: tuple[TheoryRef, ...]
    prior: TheoryPosterior
    predictions: tuple[PredictiveDistribution, ...]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id is required")
        if not self.active_theories:
            raise ValueError("active theories are required")
        if not self.predictions:
            raise ValueError("prospective predictions are required")

        theory_ids = tuple(theory.theory_id for theory in self.active_theories)
        if len(set(theory_ids)) != len(theory_ids):
            raise ValueError("active theory ids must be unique")

        prior_ids = tuple(theory_id for theory_id, _ in self.prior.probabilities)
        likelihood_ids = tuple(theory_id for theory_id, _ in self.anomaly.likelihoods)
        prediction_ids = tuple(prediction.theory_id for prediction in self.predictions)

        required = set(theory_ids)
        if set(prior_ids) != required:
            raise ValueError("prior does not match active theories")
        if set(likelihood_ids) != required:
            raise ValueError("anomaly likelihoods do not match active theories")
        if set(prediction_ids) != required or len(prediction_ids) != len(required):
            raise ValueError("predictions do not match active theories")

        experiment_id = self.anomaly.observation.experiment_id
        if any(
            prediction.experiment_id != experiment_id
            for prediction in self.predictions
        ):
            raise ValueError(
                "predictions must belong to the anomalous experiment"
            )

    @classmethod
    def from_population_anomaly(
        cls,
        *,
        request_id: str,
        anomaly: PopulationAnomaly,
        active_theories: tuple[TheoryRef, ...],
        prior: TheoryPosterior,
        predictions: tuple[PredictiveDistribution, ...],
    ) -> "TheorySpaceExpansionRequest":
        return cls(
            request_id=request_id,
            anomaly=anomaly,
            active_theories=active_theories,
            prior=prior,
            predictions=predictions,
        )


@dataclass(frozen=True)
class TheoryProposal:
    proposal_id: str
    engine_id: str
    family: RepairFamily
    artifact_id: str
    parent_theory_ids: tuple[str, ...]
    evidence_experiment_ids: tuple[str, ...]
    summary: str

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.engine_id or not self.artifact_id:
            raise ValueError(
                "proposal_id, engine_id, and artifact_id are required"
            )
        if not isinstance(self.family, RepairFamily):
            raise ValueError("typed repair family required")
        if len(set(self.parent_theory_ids)) != len(self.parent_theory_ids):
            raise ValueError("parent theory ids must be unique")
        if (
            not self.evidence_experiment_ids
            or len(set(self.evidence_experiment_ids))
            != len(self.evidence_experiment_ids)
        ):
            raise ValueError(
                "unique supporting evidence experiment ids are required"
            )
        if not self.summary:
            raise ValueError("proposal summary is required")


class RepairEngine(Protocol):
    engine_id: str
    family: RepairFamily

    def propose(
        self,
        request: TheorySpaceExpansionRequest,
    ) -> tuple[TheoryProposal, ...]: ...


@dataclass(frozen=True)
class RepairSuccess:
    engine_id: str
    family: RepairFamily
    proposals: tuple[TheoryProposal, ...]


@dataclass(frozen=True)
class RepairFailure:
    engine_id: str
    family: RepairFamily
    reason: str


RepairReport: TypeAlias = RepairSuccess | RepairFailure


@dataclass(frozen=True)
class RepairRun:
    request_id: str
    reports: tuple[RepairReport, ...]

    @property
    def proposals(self) -> tuple[TheoryProposal, ...]:
        return tuple(
            proposal
            for report in self.reports
            if isinstance(report, RepairSuccess)
            for proposal in report.proposals
        )


class RepairCoordinator:
    """Fan a population anomaly out to independent repair engines."""

    def expand(
        self,
        request: TheorySpaceExpansionRequest,
        engines: tuple[RepairEngine, ...],
    ) -> RepairRun:
        reports: list[RepairReport] = []
        active_ids = {
            theory.theory_id
            for theory in request.active_theories
        }
        anomaly_experiment_id = request.anomaly.observation.experiment_id
        seen_engine_ids: set[str] = set()

        for engine in engines:
            engine_id = engine.engine_id
            family = engine.family

            if not engine_id:
                raise ValueError("repair engine_id is required")
            if engine_id in seen_engine_ids:
                raise ValueError(f"duplicate repair engine_id: {engine_id}")
            seen_engine_ids.add(engine_id)
            if not isinstance(family, RepairFamily):
                raise ValueError(
                    f"repair engine {engine_id} has an untyped family"
                )

            try:
                proposals = tuple(engine.propose(request))
                self._validate_proposals(
                    engine_id=engine_id,
                    family=family,
                    proposals=proposals,
                    active_ids=active_ids,
                    anomaly_experiment_id=anomaly_experiment_id,
                )
                reports.append(
                    RepairSuccess(engine_id, family, proposals)
                )
            except Exception as exc:
                reports.append(
                    RepairFailure(
                        engine_id,
                        family,
                        f"{type(exc).__name__}: {exc}",
                    )
                )

        return RepairRun(request.request_id, tuple(reports))

    @staticmethod
    def _validate_proposals(
        *,
        engine_id: str,
        family: RepairFamily,
        proposals: tuple[TheoryProposal, ...],
        active_ids: set[str],
        anomaly_experiment_id: str,
    ) -> None:
        proposal_ids: set[str] = set()
        for proposal in proposals:
            if not isinstance(proposal, TheoryProposal):
                raise TypeError("repair engines must return TheoryProposal values")
            if proposal.proposal_id in proposal_ids:
                raise ValueError(
                    f"duplicate proposal_id from engine {engine_id}: "
                    f"{proposal.proposal_id}"
                )
            proposal_ids.add(proposal.proposal_id)
            if proposal.engine_id != engine_id:
                raise ValueError("proposal engine_id does not match source engine")
            if proposal.family is not family:
                raise ValueError("proposal family does not match source engine")
            if not set(proposal.parent_theory_ids).issubset(active_ids):
                raise ValueError("proposal references an inactive parent theory")
            if anomaly_experiment_id not in proposal.evidence_experiment_ids:
                raise ValueError(
                    "proposal must cite the anomaly-triggering experiment"
                )
