from pathlib import Path

from ecsa.adapters.ttt import TTTRepairEngine
from ecsa.contracts import (
    Observation,
    PopulationAnomaly,
    PredictiveDistribution,
    TheoryPosterior,
    TheoryRef,
)
from ecsa.repair import RepairCoordinator, RepairFamily, RepairSuccess, TheorySpaceExpansionRequest


def dist(theory: str, experiment: str, p_zero: float) -> PredictiveDistribution:
    return PredictiveDistribution(
        theory,
        experiment,
        (((0,), p_zero), ((1,), 1.0 - p_zero)),
    )


def expansion_request() -> TheorySpaceExpansionRequest:
    theories = (TheoryRef("h1", "a1"), TheoryRef("h2", "a2"))
    prior = TheoryPosterior((("h1", 0.5), ("h2", 0.5)))
    predictions = (
        dist("h1", "anomaly-x", 1.0),
        dist("h2", "anomaly-x", 1.0),
    )
    anomaly = PopulationAnomaly(
        Observation("anomaly-x", (1,)),
        0.0,
        (("h1", 0.0), ("h2", 0.0)),
    )
    return TheorySpaceExpansionRequest.from_population_anomaly(
        request_id="expand-ttt",
        anomaly=anomaly,
        active_theories=theories,
        prior=prior,
        predictions=predictions,
    )


def test_real_learnlib_ttt_engine_returns_content_addressed_state_memory_proposal(
    tmp_path: Path,
) -> None:
    engine = TTTRepairEngine(
        cache_root=tmp_path,
        fixture_id="C00-commands",
    )

    result = RepairCoordinator().expand(
        expansion_request(),
        (engine,),
    )

    assert len(result.reports) == 1
    assert isinstance(result.reports[0], RepairSuccess)
    assert len(result.proposals) == 1

    proposal = result.proposals[0]
    assert proposal.engine_id == "learnlib_ttt"
    assert proposal.family is RepairFamily.STATE_MEMORY
    assert proposal.parent_theory_ids == ("h1", "h2")
    assert proposal.evidence_experiment_ids == ("anomaly-x",)
    assert proposal.artifact_id.startswith("ttt-mealy:sha256:")

    artifact = engine.load_artifact(proposal.artifact_id)
    assert artifact["fixture_id"] == "C00-commands"
    assert artifact["learner_class"] == "de.learnlib.algorithm.ttt.mealy.TTTLearnerMealy"
    assert artifact["status"] == "passed"
    assert artifact["remaining_counterexample"] is None
    assert artifact["hypothesis_states"] == artifact["minimal_reference_states"] == 4
    assert artifact["mq_count"] > 0
    assert artifact["eq_count"] > 0
    assert "hypothesis" in artifact
