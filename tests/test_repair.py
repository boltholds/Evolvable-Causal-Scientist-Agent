from ecsa.contracts import (
    Observation,
    PopulationAnomaly,
    PredictiveDistribution,
    TheoryPosterior,
    TheoryRef,
)
from ecsa.repair import (
    RepairCoordinator,
    RepairFailure,
    RepairFamily,
    RepairSuccess,
    TheoryProposal,
    TheorySpaceExpansionRequest,
)


def dist(theory: str, experiment: str, p_zero: float) -> PredictiveDistribution:
    return PredictiveDistribution(
        theory,
        experiment,
        (((0,), p_zero), ((1,), 1.0 - p_zero)),
    )


def expansion_request() -> TheorySpaceExpansionRequest:
    theories = (TheoryRef("h1", "a1"), TheoryRef("h2", "a2"))
    prior = TheoryPosterior((("h1", 0.7), ("h2", 0.3)))
    predictions = (
        dist("h1", "x", 1.0),
        dist("h2", "x", 1.0),
    )
    anomaly = PopulationAnomaly(
        Observation("x", (1,)),
        0.0,
        (("h1", 0.0), ("h2", 0.0)),
    )
    return TheorySpaceExpansionRequest.from_population_anomaly(
        request_id="expand-x",
        anomaly=anomaly,
        active_theories=theories,
        prior=prior,
        predictions=predictions,
    )


def test_expansion_request_preserves_anomalous_experiment_context() -> None:
    request = expansion_request()

    assert request.request_id == "expand-x"
    assert request.anomaly.observation.experiment_id == "x"
    assert tuple(t.theory_id for t in request.active_theories) == ("h1", "h2")
    assert tuple(p.theory_id for p in request.predictions) == ("h1", "h2")
    assert request.prior.probability("h1") == 0.7


class StateRepairEngine:
    engine_id = "ttt"
    family = RepairFamily.STATE_MEMORY

    def propose(self, request: TheorySpaceExpansionRequest):
        return (
            TheoryProposal(
                proposal_id="state-memory-1",
                engine_id=self.engine_id,
                family=self.family,
                artifact_id="automaton:candidate-1",
                parent_theory_ids=("h1", "h2"),
                evidence_experiment_ids=(request.anomaly.observation.experiment_id,),
                summary="add one bit of behavioral memory",
            ),
        )


class ProgramRepairEngine:
    engine_id = "dreamcoder"
    family = RepairFamily.PROGRAM_MECHANISM

    def propose(self, request: TheorySpaceExpansionRequest):
        return (
            TheoryProposal(
                proposal_id="program-1",
                engine_id=self.engine_id,
                family=self.family,
                artifact_id="program:candidate-1",
                parent_theory_ids=("h2",),
                evidence_experiment_ids=(request.anomaly.observation.experiment_id,),
                summary="replace the failing mechanism with a synthesized program",
            ),
        )


class FailingRepairEngine:
    engine_id = "bocpd"
    family = RepairFamily.REGIME_CHANGE

    def propose(self, request: TheorySpaceExpansionRequest):
        raise RuntimeError("regime detector unavailable")


def test_repair_coordinator_collects_proposals_from_multiple_engines() -> None:
    result = RepairCoordinator().expand(
        expansion_request(),
        (StateRepairEngine(), ProgramRepairEngine()),
    )

    assert tuple(p.proposal_id for p in result.proposals) == (
        "state-memory-1",
        "program-1",
    )
    assert all(isinstance(report, RepairSuccess) for report in result.reports)
    assert tuple(report.engine_id for report in result.reports) == (
        "ttt",
        "dreamcoder",
    )


def test_repair_engine_failure_does_not_suppress_other_proposals() -> None:
    result = RepairCoordinator().expand(
        expansion_request(),
        (FailingRepairEngine(), ProgramRepairEngine()),
    )

    assert tuple(p.proposal_id for p in result.proposals) == ("program-1",)
    assert isinstance(result.reports[0], RepairFailure)
    assert result.reports[0].engine_id == "bocpd"
    assert "regime detector unavailable" in result.reports[0].reason
    assert isinstance(result.reports[1], RepairSuccess)
