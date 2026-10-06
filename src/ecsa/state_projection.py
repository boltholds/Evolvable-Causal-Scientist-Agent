from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TypeAlias

from .adapters.ttt import TTTRepairEngine
from .contracts import (
    ExperimentKind,
    ExperimentSpec,
    PredictionStatus,
    PredictionUnavailable,
    PredictiveDistribution,
    TheoryRef,
)
from .repair import RepairFamily, TheoryProposal


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class StateAlignmentEvidence:
    causal_state_key: str
    input_word: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.causal_state_key:
            raise ValueError("causal_state_key is required")
        if not isinstance(self.input_word, tuple):
            raise ValueError("input_word must be immutable")
        if any(type(symbol) is not int or symbol < 0 for symbol in self.input_word):
            raise ValueError("input symbols must be nonnegative integers")


@dataclass(frozen=True)
class StateAliasingWitness:
    causal_state_key: str
    left_word: tuple[int, ...]
    left_state: int
    right_word: tuple[int, ...]
    right_state: int

    def __post_init__(self) -> None:
        if self.left_state == self.right_state:
            raise ValueError("aliasing witness requires distinct behavioral states")


@dataclass(frozen=True)
class StateAugmentationCandidate:
    source_proposal_id: str
    parent_theory: TheoryRef
    machine_artifact_id: str
    fixture_id: str
    initial_state: int
    transitions: tuple[tuple[tuple[int, int], ...], ...]
    minimal_state_count: int

    @property
    def alphabet_size(self) -> int:
        return len(self.transitions[0])

    @property
    def state_count(self) -> int:
        return len(self.transitions)

    def state_after(self, input_word: tuple[int, ...]) -> int:
        state = self.initial_state
        for symbol in input_word:
            if type(symbol) is not int or not 0 <= symbol < self.alphabet_size:
                raise ValueError("input symbol outside TTT alphabet")
            state = self.transitions[state][symbol][0]
        return state


@dataclass(frozen=True)
class QualifiedStateAugmentation:
    candidate: StateAugmentationCandidate
    theory: TheoryRef
    witness: StateAliasingWitness


@dataclass(frozen=True)
class StateProjectionRejected:
    candidate: StateAugmentationCandidate
    reason: str


StateProjectionResult: TypeAlias = QualifiedStateAugmentation | StateProjectionRejected


class TTTStateProjector:
    """Project a minimal TTT Mealy machine into a candidate state feature."""

    def __init__(self, engine: TTTRepairEngine) -> None:
        self.engine = engine

    def project(
        self,
        proposal: TheoryProposal,
        parent: TheoryRef,
    ) -> StateAugmentationCandidate:
        if proposal.engine_id != self.engine.engine_id:
            raise ValueError("proposal does not come from this TTT engine")
        if proposal.family is not RepairFamily.STATE_MEMORY:
            raise ValueError("TTT projection requires a state-memory proposal")
        if parent.theory_id not in proposal.parent_theory_ids:
            raise ValueError("parent theory is not referenced by the proposal")

        artifact = self.engine.load_artifact(proposal.artifact_id)
        machine = artifact["hypothesis"]
        transitions = self._normalize_transitions(machine["transitions"])
        initial = machine["initial"]
        if type(initial) is not int or not 0 <= initial < len(transitions):
            raise ValueError("invalid TTT initial state")
        minimal = artifact["minimal_reference_states"]
        if minimal != len(transitions):
            raise ValueError("TTT hypothesis is not the validated minimal machine")

        return StateAugmentationCandidate(
            source_proposal_id=proposal.proposal_id,
            parent_theory=parent,
            machine_artifact_id=proposal.artifact_id,
            fixture_id=artifact["fixture_id"],
            initial_state=initial,
            transitions=transitions,
            minimal_state_count=minimal,
        )

    def qualify(
        self,
        candidate: StateAugmentationCandidate,
        alignments: tuple[StateAlignmentEvidence, ...],
    ) -> StateProjectionResult:
        if not alignments:
            return StateProjectionRejected(
                candidate,
                "no causal/behavioral state alignments supplied",
            )

        grouped: dict[str, list[tuple[tuple[int, ...], int]]] = {}
        for evidence in alignments:
            state = candidate.state_after(evidence.input_word)
            grouped.setdefault(evidence.causal_state_key, []).append(
                (evidence.input_word, state)
            )

        witness: StateAliasingWitness | None = None
        for key, values in grouped.items():
            for index, (left_word, left_state) in enumerate(values):
                for right_word, right_state in values[index + 1 :]:
                    if left_state != right_state:
                        witness = StateAliasingWitness(
                            causal_state_key=key,
                            left_word=left_word,
                            left_state=left_state,
                            right_word=right_word,
                            right_state=right_state,
                        )
                        break
                if witness is not None:
                    break
            if witness is not None:
                break

        if witness is None:
            return StateProjectionRejected(
                candidate,
                "no causal-state aliasing across distinct TTT residual states",
            )

        wire = {
            "schema": "ecsa.state-augmented-theory.v1",
            "parent_theory_id": candidate.parent_theory.theory_id,
            "parent_artifact_id": candidate.parent_theory.artifact_id,
            "source_proposal_id": candidate.source_proposal_id,
            "ttt_machine_artifact_id": candidate.machine_artifact_id,
            "fixture_id": candidate.fixture_id,
            "feature": "ttt_residual_state",
            "witness": {
                "causal_state_key": witness.causal_state_key,
                "left_word": list(witness.left_word),
                "left_state": witness.left_state,
                "right_word": list(witness.right_word),
                "right_state": witness.right_state,
            },
        }
        digest = _digest(wire)
        theory = TheoryRef(
            theory_id=(
                f"{candidate.parent_theory.theory_id}+ttt-state:{digest[:12]}"
            ),
            artifact_id=f"state-augmented:sha256:{digest}",
        )
        self._write_projection_artifact(digest, wire)
        return QualifiedStateAugmentation(candidate, theory, witness)

    def load_projection_artifact(self, artifact_id: str) -> dict:
        prefix = "state-augmented:sha256:"
        if not artifact_id.startswith(prefix):
            raise ValueError("invalid state-augmentation artifact id")
        digest = artifact_id[len(prefix) :]
        path = self._projection_path(digest)
        if not path.exists():
            raise FileNotFoundError(f"state projection artifact not found: {artifact_id}")
        wire = json.loads(path.read_text())
        if _digest(wire) != digest:
            raise ValueError("state projection artifact digest mismatch")
        return wire

    def _write_projection_artifact(self, digest: str, wire: dict) -> None:
        path = self._projection_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(wire, sort_keys=True, separators=(",", ":")) + "\n"
        if path.exists() and path.read_text() != serialized:
            raise ValueError("conflicting state projection artifact")
        if not path.exists():
            path.write_text(serialized)

    def _projection_path(self, digest: str) -> Path:
        return self.engine.cache_root / "state-projections" / f"{digest}.json"

    @staticmethod
    def _normalize_transitions(
        raw: object,
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        if not isinstance(raw, list) or not raw:
            raise ValueError("TTT transition table is required")
        rows: list[tuple[tuple[int, int], ...]] = []
        width: int | None = None
        state_count = len(raw)
        for row in raw:
            if not isinstance(row, list) or not row:
                raise ValueError("invalid TTT transition row")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError("TTT transition rows must have equal width")
            normalized_row: list[tuple[int, int]] = []
            for transition in row:
                if not isinstance(transition, list) or len(transition) != 2:
                    raise ValueError("invalid TTT transition")
                destination, output = transition
                if type(destination) is not int or not 0 <= destination < state_count:
                    raise ValueError("invalid TTT destination state")
                if type(output) is not int or not 0 <= output <= 255:
                    raise ValueError("invalid TTT encoded observation")
                normalized_row.append((destination, output))
            rows.append(tuple(normalized_row))
        return tuple(rows)


class TTTStatePredictionAdapter:
    """Predict supported ECSA interventions from a qualified TTT state feature."""

    def __init__(
        self,
        projector: TTTStateProjector,
        qualification: QualifiedStateAugmentation,
        *,
        history_symbols: tuple[int, ...] = (),
    ) -> None:
        self.projector = projector
        self.qualification = qualification
        self.history_symbols = history_symbols
        qualification.candidate.state_after(history_symbols)

    def predict(
        self,
        theory: TheoryRef,
        experiment: ExperimentSpec,
    ):
        if theory != self.qualification.theory:
            raise ValueError("theory does not match this TTT state projection")
        if experiment.kind is not ExperimentKind.INTERVENTIONAL:
            return self._undefined(theory, experiment, "TTT projection supports interventions only")

        symbol = self._symbol_for_experiment(experiment)
        if symbol is None:
            return self._undefined(
                theory,
                experiment,
                "experiment is outside the frozen TTT fixture alphabet",
            )

        candidate = self.qualification.candidate
        state = candidate.state_after(self.history_symbols)
        _, encoded_output = candidate.transitions[state][symbol]
        try:
            outcome = self._decode_outcome(
                encoded_output,
                experiment.object_id,
                experiment.outcomes,
            )
        except ValueError as exc:
            return self._undefined(theory, experiment, str(exc))

        return PredictiveDistribution(
            theory.theory_id,
            experiment.experiment_id,
            ((outcome, 1.0),),
        )

    def _symbol_for_experiment(self, experiment: ExperimentSpec) -> int | None:
        fixture = self.projector.engine.load_fixture(
            self.qualification.candidate.fixture_id
        )
        target_clamps: list[list[object]] = []
        for intervention in experiment.interventions:
            value = intervention.value
            if type(value) is not int or value not in (0, 1):
                return None
            target_clamps.append(
                [intervention.object_id, intervention.variable, value]
            )
        target_clamps.sort(key=lambda item: (item[0], item[1], item[2]))

        for index, symbol in enumerate(fixture["alphabet"]):
            actions = symbol.get("actions")
            clamps = [list(item) for item in symbol.get("clamps", [])]
            clamps.sort(key=lambda item: (item[0], item[1], item[2]))
            if actions == ["hold", "hold"] and clamps == target_clamps:
                return index
        return None

    @staticmethod
    def _decode_outcome(
        encoded: int,
        object_id: int,
        outcomes: tuple[str, ...],
    ) -> tuple[int, ...]:
        if object_id not in (0, 1):
            raise ValueError("TTT fixture exposes only object 0 and 1")
        offsets = {"R": 0, "M": 1, "C": 2, "Y": 3}
        bits = tuple((encoded >> shift) & 1 for shift in range(7, -1, -1))
        result: list[int] = []
        for variable in outcomes:
            if variable not in offsets:
                raise ValueError(f"TTT fixture does not expose outcome {variable}")
            result.append(bits[object_id * 4 + offsets[variable]])
        return tuple(result)

    @staticmethod
    def _undefined(
        theory: TheoryRef,
        experiment: ExperimentSpec,
        reason: str,
    ) -> PredictionUnavailable:
        return PredictionUnavailable(
            theory.theory_id,
            experiment.experiment_id,
            PredictionStatus.UNDEFINED,
            reason,
        )
