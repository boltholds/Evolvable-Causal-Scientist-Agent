from ecsa.adapters.bocpd import BOCPDRepairEngine
from ecsa.contracts import (
    EvidenceLedger,
    EvidenceRecord,
    Observation,
    PopulationAnomaly,
    PosteriorUpdate,
    PredictiveDistribution,
    TheoryPosterior,
    TheoryRef,
)
from ecsa.repair import (
    RepairCoordinator,
    RepairFamily,
    RepairSuccess,
    TheorySpaceExpansionRequest,
)


def evidence_record(index: int, observed_probability: float) -> EvidenceRecord:
    experiment_id = f"e-{index}"
    posterior = TheoryPosterior((("h1", 1.0),))
    prediction = PredictiveDistribution(
        "h1",
        experiment_id,
        (
            ((0,), observed_probability),
            ((1,), 1.0 - observed_probability),
        ),
    )
    observation = Observation(experiment_id, (0,))
    result = PosteriorUpdate(
        posterior,
        observed_probability,
        (("h1", observed_probability),),
    )
    return EvidenceRecord(
        posterior,
        (prediction,),
        observation,
        result,
    )


def request() -> TheorySpaceExpansionRequest:
    theory = TheoryRef("h1", "artifact:h1")
    prior = TheoryPosterior((("h1", 1.0),))
    predictions = (
        PredictiveDistribution(
            "h1",
            "anomaly-now",
            (((0,), 1.0), ((1,), 0.0)),
        ),
    )
    anomaly = PopulationAnomaly(
        Observation("anomaly-now", (1,)),
        0.0,
        (("h1", 0.0),),
    )
    return TheorySpaceExpansionRequest.from_population_anomaly(
        request_id="expand-regime",
        anomaly=anomaly,
        active_theories=(theory,),
        prior=prior,
        predictions=predictions,
    )


def test_bocpd_proposes_regime_change_after_confirmed_predictive_shift(tmp_path) -> None:
    ledger = EvidenceLedger(
        tuple(
            evidence_record(i, probability)
            for i, probability in enumerate(
                [0.99] * 30 + [0.20] * 8
            )
        )
    )
    engine = BOCPDRepairEngine(
        ledger,
        artifact_root=tmp_path,
        hazard_rate=1 / 50,
        min_reset_drop=8,
        min_post_change_run_length=2,
    )

    run = RepairCoordinator().expand(request(), (engine,))

    assert len(run.reports) == 1
    assert isinstance(run.reports[0], RepairSuccess)
    assert len(run.proposals) == 1
    proposal = run.proposals[0]
    assert proposal.engine_id == "bocpd"
    assert proposal.family is RepairFamily.REGIME_CHANGE
    assert proposal.parent_theory_ids == ("h1",)
    assert proposal.evidence_experiment_ids[-1] == "anomaly-now"
    assert proposal.artifact_id.startswith("bocpd-regime:sha256:")

    artifact = engine.load_artifact(proposal.artifact_id)
    assert artifact["schema"] == "ecsa.bocpd-regime.v1"
    assert artifact["source_repository"] == "https://github.com/fiannai/bocd.git"
    assert artifact["changepoint_step"] >= 25
    assert artifact["latest_map_run_length"] >= 2
    assert artifact["signal"][-1] == 1.0


def test_bocpd_does_not_call_single_unconfirmed_anomaly_a_regime_change(
    tmp_path,
) -> None:
    ledger = EvidenceLedger(
        tuple(
            evidence_record(i, 0.99)
            for i in range(30)
        )
    )
    engine = BOCPDRepairEngine(
        ledger,
        artifact_root=tmp_path,
        hazard_rate=1 / 50,
        min_reset_drop=8,
        min_post_change_run_length=2,
    )

    run = RepairCoordinator().expand(request(), (engine,))

    assert len(run.proposals) == 0
    assert isinstance(run.reports[0], RepairSuccess)
