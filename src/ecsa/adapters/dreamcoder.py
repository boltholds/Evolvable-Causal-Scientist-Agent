from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

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
                timeout=self.timeout_seconds + 5.0,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("DreamCoder synthesis timed out") from exc
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
