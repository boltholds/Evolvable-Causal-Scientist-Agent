from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from .contracts import TransferArenaResult


def compute_transfer_gain(*, cold: int, reuse: int) -> float | None:
    if type(cold) is not int or type(reuse) is not int or cold < 0 or reuse < 0:
        raise ValueError("transfer counts must be nonnegative integers")
    if cold == 0:
        return None
    return 1.0 - (reuse / cold)


@dataclass(frozen=True)
class ReactorRunMetrics:
    measurement_actions: int
    distinct_measurements: int
    transfer_candidates_retrieved: int
    transfer_candidates_tested: int
    transfer_candidates_accepted: int
    transfer_candidates_rejected: int
    false_transfer_events: int


class ReactorRunMetricsAccumulator:
    def __init__(self) -> None:
        self._measurement_actions = 0
        self._measurement_keys: set[tuple[int, int]] = set()
        self._retrieved: set[str] = set()
        self._tested: set[str] = set()
        self._accepted: set[str] = set()
        self._rejected: set[str] = set()
        self._false_transfer_events = 0

    def record_measurement(
        self,
        *,
        instrument_uuid: int,
        crystal_uuid: int,
    ) -> None:
        self._measurement_actions += 1
        self._measurement_keys.add((instrument_uuid, crystal_uuid))

    def record_candidate_retrieved(self, ref: str) -> None:
        self._validate_ref(ref)
        self._retrieved.add(ref)

    def record_candidate_tested(self, ref: str) -> None:
        self._validate_ref(ref)
        self._tested.add(ref)

    def record_candidate_accepted(self, ref: str) -> None:
        self._validate_ref(ref)
        self._accepted.add(ref)

    def record_candidate_rejected(
        self,
        ref: str,
        *,
        false_transfer: bool = False,
    ) -> None:
        self._validate_ref(ref)
        self._rejected.add(ref)
        if false_transfer:
            self._false_transfer_events += 1

    def snapshot(self) -> ReactorRunMetrics:
        return ReactorRunMetrics(
            measurement_actions=self._measurement_actions,
            distinct_measurements=len(self._measurement_keys),
            transfer_candidates_retrieved=len(self._retrieved),
            transfer_candidates_tested=len(self._tested),
            transfer_candidates_accepted=len(self._accepted),
            transfer_candidates_rejected=len(self._rejected),
            false_transfer_events=self._false_transfer_events,
        )

    @staticmethod
    def _validate_ref(ref: str) -> None:
        if not isinstance(ref, str) or not ref:
            raise ValueError("mechanism reference must be a nonempty string")



@dataclass(frozen=True)
class SeedTransferMetrics:
    seed: int
    cold_measurement_actions: int
    reuse_measurement_actions: int
    cold_steps: int
    reuse_steps: int
    measurement_transfer_gain: float | None
    step_transfer_gain: float | None


@dataclass(frozen=True)
class TransferSummary:
    per_seed: tuple[SeedTransferMetrics, ...]
    mean_measurement_transfer_gain: float | None
    mean_step_transfer_gain: float | None


def summarize_transfer(result: TransferArenaResult) -> TransferSummary:
    per_seed = tuple(
        SeedTransferMetrics(
            seed=pair.seed,
            cold_measurement_actions=pair.cold.episode.measurement_count,
            reuse_measurement_actions=pair.reuse.episode.measurement_count,
            cold_steps=pair.cold.episode.evaluation.steps,
            reuse_steps=pair.reuse.episode.evaluation.steps,
            measurement_transfer_gain=compute_transfer_gain(
                cold=pair.cold.episode.measurement_count,
                reuse=pair.reuse.episode.measurement_count,
            ),
            step_transfer_gain=compute_transfer_gain(
                cold=pair.cold.episode.evaluation.steps,
                reuse=pair.reuse.episode.evaluation.steps,
            ),
        )
        for pair in result.pairs
    )

    transfer_rows = tuple(item for item in per_seed if item.seed != 0)
    measurement_values = tuple(
        item.measurement_transfer_gain
        for item in transfer_rows
        if item.measurement_transfer_gain is not None
    )
    step_values = tuple(
        item.step_transfer_gain
        for item in transfer_rows
        if item.step_transfer_gain is not None
    )
    return TransferSummary(
        per_seed=per_seed,
        mean_measurement_transfer_gain=(
            fmean(measurement_values)
            if measurement_values
            else None
        ),
        mean_step_transfer_gain=(
            fmean(step_values)
            if step_values
            else None
        ),
    )
