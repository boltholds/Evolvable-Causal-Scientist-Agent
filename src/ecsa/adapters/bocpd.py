from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from bocd import ConstantHazard, Detector, GaussianModel

from ecsa.contracts import EvidenceLedger
from ecsa.repair import (
    RepairFamily,
    TheoryProposal,
    TheorySpaceExpansionRequest,
)

BOCPD_REPOSITORY = "https://github.com/fiannai/bocd.git"
BOCPD_REVISION = "0c09315e9102ddea885bb7164e47b61a95e99dac"


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


class BOCPDRepairEngine:
    """Detect sustained predictive-regime changes with Bayesian online CPD.

    The signal is population predictive discrepancy:
        1 - P(observed outcome | current theory population)

    Using predictive discrepancy instead of raw observations makes the detector
    comparable across different intervention choices. A change-point proposal
    is emitted only after a run-length reset has been followed by at least
    min_post_change_run_length observations, so a single terminal anomaly is
    not declared a regime change by itself.
    """

    engine_id = "bocpd"
    family = RepairFamily.REGIME_CHANGE

    def __init__(
        self,
        ledger: EvidenceLedger,
        *,
        artifact_root: Path,
        hazard_rate: float = 1 / 200,
        min_reset_drop: int = 20,
        min_post_change_run_length: int = 2,
        max_run_length: int | None = 500,
    ) -> None:
        if not 0.0 < hazard_rate < 1.0:
            raise ValueError("hazard_rate must be in (0,1)")
        if type(min_reset_drop) is not int or min_reset_drop < 1:
            raise ValueError("min_reset_drop must be a positive integer")
        if (
            type(min_post_change_run_length) is not int
            or min_post_change_run_length < 1
        ):
            raise ValueError(
                "min_post_change_run_length must be a positive integer"
            )
        if (
            max_run_length is not None
            and (
                type(max_run_length) is not int
                or max_run_length < 2
            )
        ):
            raise ValueError("max_run_length must be None or an integer >=2")

        self.ledger = ledger
        self.artifact_root = Path(artifact_root)
        self.hazard_rate = float(hazard_rate)
        self.min_reset_drop = min_reset_drop
        self.min_post_change_run_length = min_post_change_run_length
        self.max_run_length = max_run_length

    def propose(
        self,
        request: TheorySpaceExpansionRequest,
    ) -> tuple[TheoryProposal, ...]:
        signal, experiment_ids = self._predictive_discrepancy_series(request)
        if len(signal) < 3:
            return ()

        detector = Detector(
            model=GaussianModel(),
            hazard=ConstantHazard(rate=self.hazard_rate),
            max_run_length=self.max_run_length,
        )
        results = detector.fit(signal)

        reset_steps: list[int] = []
        reset_drops: list[int] = []
        for index in range(1, len(results)):
            drop = (
                results[index - 1].most_likely_run_length
                - results[index].most_likely_run_length
            )
            if drop >= self.min_reset_drop:
                reset_steps.append(index)
                reset_drops.append(drop)

        if not reset_steps:
            return ()

        changepoint_step = reset_steps[-1]
        reset_drop = reset_drops[-1]
        latest_map_run_length = results[-1].most_likely_run_length
        if latest_map_run_length < self.min_post_change_run_length:
            return ()

        artifact = {
            "schema": "ecsa.bocpd-regime.v1",
            "source_repository": BOCPD_REPOSITORY,
            "source_revision": BOCPD_REVISION,
            "signal_kind": "population_predictive_discrepancy",
            "signal": signal,
            "experiment_ids": experiment_ids,
            "hazard_rate": self.hazard_rate,
            "max_run_length": self.max_run_length,
            "min_reset_drop": self.min_reset_drop,
            "min_post_change_run_length": self.min_post_change_run_length,
            "changepoint_step": changepoint_step,
            "changepoint_experiment_id": experiment_ids[changepoint_step],
            "reset_drop": reset_drop,
            "latest_map_run_length": latest_map_run_length,
            "map_run_lengths": [
                result.most_likely_run_length
                for result in results
            ],
            "changepoint_probabilities": [
                result.changepoint_prob
                for result in results
            ],
        }
        digest = _digest(artifact)
        artifact_id = f"bocpd-regime:sha256:{digest}"
        self._write_artifact(digest, artifact)

        evidence_start = max(0, changepoint_step - 1)
        evidence_ids = tuple(experiment_ids[evidence_start:])
        parent_ids = tuple(
            theory.theory_id
            for theory in request.active_theories
        )
        proposal = TheoryProposal(
            proposal_id=f"bocpd-regime:{digest[:16]}",
            engine_id=self.engine_id,
            family=self.family,
            artifact_id=artifact_id,
            parent_theory_ids=parent_ids,
            evidence_experiment_ids=evidence_ids,
            summary=(
                "BOCPD detected a sustained reset in predictive-error run "
                f"length at step {changepoint_step} "
                f"(drop={reset_drop}, latest_run_length={latest_map_run_length})."
            ),
        )
        return (proposal,)

    def load_artifact(self, artifact_id: str) -> dict:
        prefix = "bocpd-regime:sha256:"
        if not artifact_id.startswith(prefix):
            raise ValueError("invalid BOCPD artifact id")
        digest = artifact_id[len(prefix):]
        if len(digest) != 64 or any(
            char not in "0123456789abcdef"
            for char in digest
        ):
            raise ValueError("invalid BOCPD artifact digest")
        path = self._artifact_path(digest)
        if not path.exists():
            raise FileNotFoundError(f"BOCPD artifact not found: {artifact_id}")
        artifact = json.loads(path.read_text())
        if _digest(artifact) != digest:
            raise ValueError("BOCPD artifact digest mismatch")
        return artifact

    def _predictive_discrepancy_series(
        self,
        request: TheorySpaceExpansionRequest,
    ) -> tuple[list[float], list[str]]:
        anomaly_id = request.anomaly.observation.experiment_id
        signal: list[float] = []
        experiment_ids: list[str] = []

        for record in self.ledger.records:
            if record.observation.experiment_id == anomaly_id:
                continue
            probability = float(record.result.evidence_probability)
            self._validate_probability(probability)
            signal.append(1.0 - probability)
            experiment_ids.append(record.observation.experiment_id)

        current_probability = float(request.anomaly.evidence_probability)
        self._validate_probability(current_probability)
        signal.append(1.0 - current_probability)
        experiment_ids.append(anomaly_id)
        return signal, experiment_ids

    @staticmethod
    def _validate_probability(probability: float) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "BOCPD predictive-discrepancy adapter requires evidence "
                "probabilities in [0,1]"
            )

    def _write_artifact(self, digest: str, artifact: dict) -> None:
        path = self._artifact_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        if path.exists():
            if path.read_text() != serialized:
                raise ValueError("conflicting BOCPD artifact content")
            return
        path.write_text(serialized)

    def _artifact_path(self, digest: str) -> Path:
        return self.artifact_root / "bocpd" / f"{digest}.json"
