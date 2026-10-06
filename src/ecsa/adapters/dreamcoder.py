from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TypeAlias

from ecsa.contracts import (
    ExperimentKind,
    ExperimentSpec,
    Observation,
    PredictionStatus,
    PredictionUnavailable,
    PredictiveDistribution,
    TheoryRef,
)
from ecsa.repair import (
    RepairFamily,
    TheoryProposal,
    TheorySpaceExpansionRequest,
)

DREAMCODER_REPOSITORY = "https://github.com/ellisk42/ec.git"
DREAMCODER_REVISION = "cb0e63f5c33cd2de360b791038b0f5272750270e"
_ALLOWED_BOOLEAN_PRIMITIVES = {"not", "and", "or", "xor"}


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BooleanMechanismExample:
    inputs: tuple[bool, ...]
    output: bool
    experiment_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, tuple) or not self.inputs:
            raise ValueError("nonempty immutable boolean inputs required")
        if not all(type(value) is bool for value in self.inputs):
            raise ValueError("boolean mechanism inputs must be bool")
        if type(self.output) is not bool:
            raise ValueError("boolean mechanism output must be bool")
        if not self.experiment_id:
            raise ValueError("experiment_id is required")


class DreamCoderRepairEngine:
    """Synthesize a boolean program mechanism with DreamCoder's core enumerator."""

    engine_id = "dreamcoder"
    family = RepairFamily.PROGRAM_MECHANISM

    def __init__(
        self,
        examples: tuple[BooleanMechanismExample, ...],
        *,
        cache_root: Path,
        primitives: tuple[str, ...] = ("not", "and", "or"),
        maximum_mdl: float = 14.0,
        maximum_programs: int = 200_000,
        timeout_seconds: float = 20.0,
        evaluation_timeout: float = 0.05,
        source_root: Path | None = None,
    ) -> None:
        if not isinstance(examples, tuple) or not examples:
            raise ValueError("at least one boolean mechanism example is required")
        if not all(isinstance(example, BooleanMechanismExample) for example in examples):
            raise ValueError("typed BooleanMechanismExample values required")
        arity = len(examples[0].inputs)
        if any(len(example.inputs) != arity for example in examples):
            raise ValueError("all mechanism examples must have the same arity")
        if len({example.inputs for example in examples}) != len(examples):
            raise ValueError("mechanism example inputs must be unique")
        if len({example.experiment_id for example in examples}) != len(examples):
            raise ValueError("mechanism example experiment ids must be unique")

        if (
            not isinstance(primitives, tuple)
            or not primitives
            or len(set(primitives)) != len(primitives)
        ):
            raise ValueError("unique immutable primitive names are required")
        unsupported = set(primitives) - _ALLOWED_BOOLEAN_PRIMITIVES
        if unsupported:
            raise ValueError(f"unsupported boolean primitives: {sorted(unsupported)}")

        if maximum_mdl <= 0:
            raise ValueError("maximum_mdl must be positive")
        if type(maximum_programs) is not int or maximum_programs < 1:
            raise ValueError("maximum_programs must be a positive integer")
        if timeout_seconds <= 0 or evaluation_timeout <= 0:
            raise ValueError("timeouts must be positive")

        self.examples = examples
        self.cache_root = Path(cache_root)
        self.primitives = primitives
        self.maximum_mdl = float(maximum_mdl)
        self.maximum_programs = maximum_programs
        self.timeout_seconds = float(timeout_seconds)
        self.evaluation_timeout = float(evaluation_timeout)
        self.source_root = Path(source_root) if source_root is not None else None
        self.arity = arity

    def propose(
        self,
        request: TheorySpaceExpansionRequest,
    ) -> tuple[TheoryProposal, ...]:
        result = self._run_worker()
        if result.get("status") == "not_found":
            return ()
        if result.get("status") != "found":
            raise RuntimeError(f"unexpected DreamCoder worker status: {result!r}")

        artifact = {
            "schema": "ecsa.dreamcoder-program.v1",
            "source_repository": DREAMCODER_REPOSITORY,
            "source_revision": DREAMCODER_REVISION,
            "domain": "boolean",
            "arity": self.arity,
            "primitives": list(self.primitives),
            "examples": [
                {
                    "inputs": list(example.inputs),
                    "output": example.output,
                    "experiment_id": example.experiment_id,
                }
                for example in self.examples
            ],
            "program": result["program"],
            "request": result["request"],
            "log_prior": result["log_prior"],
            "mdl": result["mdl"],
            "programs_enumerated": result["programs_enumerated"],
            "verified_examples": len(self.examples),
            "maximum_mdl": self.maximum_mdl,
            "maximum_programs": self.maximum_programs,
        }
        digest = _digest(artifact)
        artifact_id = f"dreamcoder-program:sha256:{digest}"
        self._write_artifact(digest, artifact)

        evidence_ids = [example.experiment_id for example in self.examples]
        anomaly_id = request.anomaly.observation.experiment_id
        if anomaly_id not in evidence_ids:
            evidence_ids.append(anomaly_id)

        parent_ids = tuple(
            theory.theory_id
            for theory in request.active_theories
        )
        proposal = TheoryProposal(
            proposal_id=f"dreamcoder-program:{digest[:16]}",
            engine_id=self.engine_id,
            family=self.family,
            artifact_id=artifact_id,
            parent_theory_ids=parent_ids,
            evidence_experiment_ids=tuple(evidence_ids),
            summary=(
                "DreamCoder synthesized a boolean mechanism "
                f"with MDL={result['mdl']:.6f} after enumerating "
                f"{result['programs_enumerated']} programs."
            ),
        )
        return (proposal,)

    def load_artifact(self, artifact_id: str) -> dict:
        prefix = "dreamcoder-program:sha256:"
        if not artifact_id.startswith(prefix):
            raise ValueError("invalid DreamCoder artifact id")
        digest = artifact_id[len(prefix):]
        if len(digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in digest
        ):
            raise ValueError("invalid DreamCoder artifact digest")
        path = self._artifact_path(digest)
        if not path.exists():
            raise FileNotFoundError(f"DreamCoder artifact not found: {artifact_id}")
        artifact = json.loads(path.read_text())
        if _digest(artifact) != digest:
            raise ValueError("DreamCoder artifact digest mismatch")
        return artifact

    def evaluate_program(
        self,
        artifact_id: str,
        inputs: tuple[bool, ...],
    ) -> bool:
        artifact = self.load_artifact(artifact_id)
        if len(inputs) != artifact["arity"]:
            raise ValueError("program input arity mismatch")
        if not all(type(value) is bool for value in inputs):
            raise ValueError("DreamCoder program inputs must be bool")
        payload = {
            "mode": "evaluate",
            "primitives": artifact["primitives"],
            "program": artifact["program"],
            "inputs": list(inputs),
        }
        result = self._invoke_worker(
            payload,
            timeout_seconds=self.evaluation_timeout + 5.0,
        )
        if result.get("status") != "ok" or type(result.get("output")) is not bool:
            raise RuntimeError(f"unexpected DreamCoder evaluation result: {result!r}")
        return result["output"]

    def _run_worker(self) -> dict:
        source_root = self._ensure_source()
        payload = {
            "arity": self.arity,
            "primitives": list(self.primitives),
            "examples": [
                {
                    "inputs": list(example.inputs),
                    "output": example.output,
                }
                for example in self.examples
            ],
            "maximum_mdl": self.maximum_mdl,
            "maximum_programs": self.maximum_programs,
            "timeout_seconds": self.timeout_seconds,
            "evaluation_timeout": self.evaluation_timeout,
        }
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(source_root)
            if not existing_pythonpath
            else str(source_root) + os.pathsep + existing_pythonpath
        )
        environment["ECSA_DREAMCODER_SOURCE_ROOT"] = str(source_root)
        return self._invoke_worker(
            payload,
            timeout_seconds=self.timeout_seconds + 5.0,
            environment=environment,
        )

    def _invoke_worker(
        self,
        payload: dict,
        *,
        timeout_seconds: float,
        environment: dict[str, str] | None = None,
    ) -> dict:
        source_root = self._ensure_source()
        if environment is None:
            environment = os.environ.copy()
            existing_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(source_root)
                if not existing_pythonpath
                else str(source_root) + os.pathsep + existing_pythonpath
            )
            environment["ECSA_DREAMCODER_SOURCE_ROOT"] = str(source_root)
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ecsa.adapters._dreamcoder_worker",
                ],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("DreamCoder worker timed out") from exc
        if process.returncode != 0:
            raise RuntimeError(
                "DreamCoder worker failed: "
                + (process.stderr.strip() or process.stdout.strip())
            )
        try:
            return json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"DreamCoder worker returned invalid JSON: {process.stdout!r}"
            ) from exc

    def _ensure_source(self) -> Path:
        if self.source_root is not None:
            return self.source_root

        root = self.cache_root / (
            "dreamcoder-source-" + DREAMCODER_REVISION[:12]
        )
        if not root.exists():
            root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    DREAMCODER_REPOSITORY,
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "checkout",
                    "--detach",
                    DREAMCODER_REVISION,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        current = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if current != DREAMCODER_REVISION:
            raise ValueError(
                "cached DreamCoder source is not at the pinned revision"
            )
        return root

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
                raise ValueError("conflicting DreamCoder artifact content")
            return
        path.write_text(serialized)

    def _artifact_path(self, digest: str) -> Path:
        return self.cache_root / "dreamcoder" / f"{digest}.json"



class DreamCoderProgramPredictionAdapter:
    """Project a synthesized Boolean DreamCoder program into ECSA predictions."""

    def __init__(
        self,
        engine: DreamCoderRepairEngine,
        proposal_or_artifact: TheoryProposal | str,
        *,
        input_variables: tuple[str, ...],
        output_variable: str,
        theory_id: str | None = None,
    ) -> None:
        artifact_id = (
            proposal_or_artifact.artifact_id
            if isinstance(proposal_or_artifact, TheoryProposal)
            else proposal_or_artifact
        )
        artifact = engine.load_artifact(artifact_id)
        if artifact.get("domain") != "boolean":
            raise ValueError("only Boolean DreamCoder artifacts are supported")
        if (
            not isinstance(input_variables, tuple)
            or not input_variables
            or len(set(input_variables)) != len(input_variables)
            or not all(isinstance(name, str) and name for name in input_variables)
        ):
            raise ValueError("unique nonempty input variable names are required")
        if len(input_variables) != artifact["arity"]:
            raise ValueError("input variable count does not match program arity")
        if not output_variable:
            raise ValueError("output_variable is required")

        self.engine = engine
        self.program_artifact_id = artifact_id
        self.input_variables = input_variables
        self.output_variable = output_variable
        self.theory_id = theory_id

    def predict(
        self,
        theory: TheoryRef,
        experiment: ExperimentSpec,
    ):
        if self.theory_id is not None and theory.theory_id != self.theory_id:
            raise ValueError("theory does not match this program mechanism")
        if experiment.kind is not ExperimentKind.INTERVENTIONAL:
            return self._undefined(theory, experiment, "program mechanism supports interventions only")
        if experiment.outcomes != (self.output_variable,):
            return self._undefined(
                theory,
                experiment,
                "experiment outcome does not match program mechanism output",
            )
        if any(
            intervention.object_id != experiment.object_id
            for intervention in experiment.interventions
        ):
            return self._undefined(
                theory,
                experiment,
                "program mechanism supports one object-local intervention context",
            )

        values: dict[str, bool] = {}
        for intervention in experiment.interventions:
            if intervention.variable in values:
                return self._undefined(theory, experiment, "duplicate program input intervention")
            raw = intervention.value
            if type(raw) is bool:
                value = raw
            elif type(raw) is int and raw in (0, 1):
                value = bool(raw)
            else:
                return self._undefined(
                    theory,
                    experiment,
                    "program mechanism requires Boolean input interventions",
                )
            values[intervention.variable] = value

        if set(values) != set(self.input_variables):
            return self._undefined(
                theory,
                experiment,
                "experiment does not provide exactly the program input variables",
            )

        output = self.engine.evaluate_program(
            self.program_artifact_id,
            tuple(values[name] for name in self.input_variables),
        )
        return PredictiveDistribution(
            theory.theory_id,
            experiment.experiment_id,
            (((int(output),), 1.0),),
        )

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


@dataclass(frozen=True)
class QualifiedProgramMechanism:
    theory: TheoryRef
    source_program_artifact_id: str
    input_variables: tuple[str, ...]
    output_variable: str
    heldout_experiment_ids: tuple[str, ...]

    def prediction_adapter(
        self,
        engine: DreamCoderRepairEngine,
    ) -> DreamCoderProgramPredictionAdapter:
        return DreamCoderProgramPredictionAdapter(
            engine,
            self.source_program_artifact_id,
            input_variables=self.input_variables,
            output_variable=self.output_variable,
            theory_id=self.theory.theory_id,
        )


@dataclass(frozen=True)
class RejectedProgramMechanism:
    proposal_id: str
    reason: str


ProgramMechanismQualification: TypeAlias = (
    QualifiedProgramMechanism | RejectedProgramMechanism
)


class ProgramMechanismProjector:
    """Qualify a synthesized program on prospective held-out evidence."""

    def __init__(self, engine: DreamCoderRepairEngine) -> None:
        self.engine = engine

    def qualify(
        self,
        proposal: TheoryProposal,
        *,
        parent_theory: TheoryRef,
        input_variables: tuple[str, ...],
        output_variable: str,
        heldout: tuple[tuple[ExperimentSpec, Observation], ...],
    ) -> ProgramMechanismQualification:
        if proposal.engine_id != self.engine.engine_id:
            raise ValueError("proposal does not come from this DreamCoder engine")
        if proposal.family is not RepairFamily.PROGRAM_MECHANISM:
            raise ValueError("program projection requires PROGRAM_MECHANISM proposal")
        if parent_theory.theory_id not in proposal.parent_theory_ids:
            raise ValueError("parent theory is not referenced by the proposal")
        if not heldout:
            return RejectedProgramMechanism(
                proposal.proposal_id,
                "at least one held-out prospective observation is required",
            )

        artifact = self.engine.load_artifact(proposal.artifact_id)
        training_ids = {
            example["experiment_id"]
            for example in artifact["examples"]
        }
        heldout_ids: list[str] = []
        provisional = TheoryRef(
            theory_id=f"candidate:{proposal.proposal_id}",
            artifact_id=proposal.artifact_id,
        )
        adapter = DreamCoderProgramPredictionAdapter(
            self.engine,
            proposal,
            input_variables=input_variables,
            output_variable=output_variable,
        )

        validations: list[dict] = []
        for experiment, observation in heldout:
            if observation.experiment_id != experiment.experiment_id:
                raise ValueError("held-out observation/experiment id mismatch")
            if experiment.experiment_id in training_ids:
                return RejectedProgramMechanism(
                    proposal.proposal_id,
                    "held-out experiment overlaps DreamCoder synthesis evidence",
                )
            prediction = adapter.predict(provisional, experiment)
            if not isinstance(prediction, PredictiveDistribution):
                return RejectedProgramMechanism(
                    proposal.proposal_id,
                    "held-out experiment is outside the program prediction domain",
                )
            probability = prediction.probability(observation.outcome)
            if probability < 1.0:
                return RejectedProgramMechanism(
                    proposal.proposal_id,
                    "program failed held-out prospective validation",
                )
            heldout_ids.append(experiment.experiment_id)
            validations.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "observation": list(observation.outcome),
                    "predicted_probability": probability,
                }
            )

        wire = {
            "schema": "ecsa.program-mechanism.v1",
            "parent_theory_id": parent_theory.theory_id,
            "parent_artifact_id": parent_theory.artifact_id,
            "source_proposal_id": proposal.proposal_id,
            "source_program_artifact_id": proposal.artifact_id,
            "input_variables": list(input_variables),
            "output_variable": output_variable,
            "heldout_validations": validations,
        }
        digest = _digest(wire)
        theory = TheoryRef(
            theory_id=(
                f"{parent_theory.theory_id}+dreamcoder-program:{digest[:12]}"
            ),
            artifact_id=f"program-mechanism:sha256:{digest}",
        )
        self._write_projection(digest, wire)
        return QualifiedProgramMechanism(
            theory=theory,
            source_program_artifact_id=proposal.artifact_id,
            input_variables=input_variables,
            output_variable=output_variable,
            heldout_experiment_ids=tuple(heldout_ids),
        )

    def load_projection_artifact(self, artifact_id: str) -> dict:
        prefix = "program-mechanism:sha256:"
        if not artifact_id.startswith(prefix):
            raise ValueError("invalid program-mechanism artifact id")
        digest = artifact_id[len(prefix):]
        path = self._projection_path(digest)
        if not path.exists():
            raise FileNotFoundError(f"program-mechanism artifact not found: {artifact_id}")
        wire = json.loads(path.read_text())
        if _digest(wire) != digest:
            raise ValueError("program-mechanism artifact digest mismatch")
        return wire

    def _write_projection(self, digest: str, wire: dict) -> None:
        path = self._projection_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            wire,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        if path.exists():
            if path.read_text() != serialized:
                raise ValueError("conflicting program-mechanism artifact content")
            return
        path.write_text(serialized)

    def _projection_path(self, digest: str) -> Path:
        return self.engine.cache_root / "program-projections" / f"{digest}.json"
