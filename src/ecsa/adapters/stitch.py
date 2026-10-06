from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from ecsa.adapters.dreamcoder import QualifiedProgramMechanism
from ecsa.repair import RepairFamily

STITCH_REPOSITORY = "https://github.com/mlb2251/stitch.git"
STITCH_REVISION = "350804b7b35807c78bd21c313785ae5152ae2985"


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


class ProgramArtifactProvider(Protocol):
    def load_artifact(self, artifact_id: str) -> dict: ...


@dataclass(frozen=True)
class StitchAbstractionProposal:
    proposal_id: str
    engine_id: str
    family: RepairFamily
    artifact_id: str
    source_theory_ids: tuple[str, ...]
    source_program_artifact_ids: tuple[str, ...]
    body: str
    arity: int
    utility: int
    compression_ratio: float
    num_uses: int


@dataclass(frozen=True)
class StitchAbstractionRun:
    proposals: tuple[StitchAbstractionProposal, ...]
    original_cost: int | None = None
    final_cost: int | None = None
    compression_ratio: float | None = None


class StitchAbstractionEngine:
    """Learn reusable abstractions from qualified program mechanisms only."""

    engine_id = "stitch"
    family = RepairFamily.ABSTRACTION_TRANSFER

    def __init__(
        self,
        mechanisms: tuple[QualifiedProgramMechanism, ...],
        *,
        program_artifacts: ProgramArtifactProvider,
        cache_root: Path,
        max_arity: int = 2,
        iterations: int = 1,
        threads: int = 1,
        source_root: Path | None = None,
    ) -> None:
        if not isinstance(mechanisms, tuple) or not mechanisms:
            raise ValueError("at least one qualified program mechanism is required")
        if not all(isinstance(item, QualifiedProgramMechanism) for item in mechanisms):
            raise ValueError("Stitch accepts only QualifiedProgramMechanism values")
        theory_ids = tuple(item.theory.theory_id for item in mechanisms)
        if len(set(theory_ids)) != len(theory_ids):
            raise ValueError("source theory ids must be unique")
        artifact_ids = tuple(item.source_program_artifact_id for item in mechanisms)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("source program artifact ids must be unique")
        if type(max_arity) is not int or max_arity < 0:
            raise ValueError("max_arity must be a nonnegative integer")
        if type(iterations) is not int or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")

        self.mechanisms = mechanisms
        self.program_artifacts = program_artifacts
        self.cache_root = Path(cache_root)
        self.max_arity = max_arity
        self.iterations = iterations
        self.threads = threads
        self.source_root = Path(source_root) if source_root is not None else None

    def compress(self) -> StitchAbstractionRun:
        # A reusable abstraction must span at least two independently qualified
        # mechanism artifacts; do not ask Stitch for a single-program shortcut.
        if len(self.mechanisms) < 2:
            return StitchAbstractionRun(())

        programs = tuple(
            self._load_program(mechanism)
            for mechanism in self.mechanisms
        )
        binary = self._ensure_binary()
        self.cache_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix="stitch-run-",
            dir=self.cache_root,
        ) as tmp:
            work = Path(tmp)
            input_path = work / "programs.json"
            output_path = work / "out.json"
            input_path.write_text(json.dumps(list(programs)) + "\n")

            process = subprocess.run(
                [
                    str(binary),
                    str(input_path),
                    "--out",
                    str(output_path),
                    "--max-arity",
                    str(self.max_arity),
                    "--iterations",
                    str(self.iterations),
                    "--threads",
                    str(self.threads),
                    "--silent",
                    "--rewrite-check",
                ],
                cwd=binary.parent.parent.parent,
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    "Stitch compression failed: "
                    + (process.stderr.strip() or process.stdout.strip())
                )
            if not output_path.exists():
                raise RuntimeError("Stitch did not produce its JSON output")
            result = json.loads(output_path.read_text())

        abstractions = result.get("abstractions")
        if not isinstance(abstractions, list):
            raise ValueError("invalid Stitch output: abstractions must be a list")

        source_theory_ids = tuple(
            mechanism.theory.theory_id
            for mechanism in self.mechanisms
        )
        source_program_ids = tuple(
            mechanism.source_program_artifact_id
            for mechanism in self.mechanisms
        )
        rewritten = result.get("rewritten")
        if rewritten is None:
            rewritten = list(programs)
        if not isinstance(rewritten, list) or not all(
            isinstance(value, str) for value in rewritten
        ):
            raise ValueError("invalid Stitch output: rewritten programs")

        proposals: list[StitchAbstractionProposal] = []
        for index, abstraction in enumerate(abstractions):
            if not isinstance(abstraction, dict):
                raise ValueError("invalid Stitch abstraction entry")
            body = abstraction.get("body")
            arity = abstraction.get("arity")
            utility = abstraction.get("utility")
            ratio = abstraction.get("compression_ratio")
            num_uses = abstraction.get("num_uses")
            name = abstraction.get("name")
            if (
                not isinstance(body, str)
                or type(arity) is not int
                or type(utility) is not int
                or not isinstance(ratio, (int, float))
                or isinstance(ratio, bool)
                or type(num_uses) is not int
                or not isinstance(name, str)
            ):
                raise ValueError("invalid Stitch abstraction metrics")

            wire = {
                "schema": "ecsa.stitch-abstraction.v1",
                "source_repository": STITCH_REPOSITORY,
                "source_revision": STITCH_REVISION,
                "source_theory_ids": list(source_theory_ids),
                "source_program_artifact_ids": list(source_program_ids),
                "original_programs": list(programs),
                "rewritten_programs": list(rewritten),
                "original_cost": result.get("original_cost"),
                "final_cost": result.get("final_cost"),
                "compression_ratio": result.get("compression_ratio"),
                "abstraction_index": index,
                "abstraction": abstraction,
                "stitch_args": {
                    "max_arity": self.max_arity,
                    "iterations": self.iterations,
                    "threads": self.threads,
                },
            }
            digest = _digest(wire)
            artifact_id = f"stitch-abstraction:sha256:{digest}"
            self._write_artifact(digest, wire)
            proposals.append(
                StitchAbstractionProposal(
                    proposal_id=f"stitch:{name}:{digest[:16]}",
                    engine_id=self.engine_id,
                    family=self.family,
                    artifact_id=artifact_id,
                    source_theory_ids=source_theory_ids,
                    source_program_artifact_ids=source_program_ids,
                    body=body,
                    arity=arity,
                    utility=utility,
                    compression_ratio=float(ratio),
                    num_uses=num_uses,
                )
            )

        return StitchAbstractionRun(
            tuple(proposals),
            original_cost=result.get("original_cost"),
            final_cost=result.get("final_cost"),
            compression_ratio=(
                float(result["compression_ratio"])
                if isinstance(result.get("compression_ratio"), (int, float))
                and not isinstance(result.get("compression_ratio"), bool)
                else None
            ),
        )

    def load_artifact(self, artifact_id: str) -> dict:
        prefix = "stitch-abstraction:sha256:"
        if not artifact_id.startswith(prefix):
            raise ValueError("invalid Stitch artifact id")
        digest = artifact_id[len(prefix):]
        if len(digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in digest
        ):
            raise ValueError("invalid Stitch artifact digest")
        path = self._artifact_path(digest)
        if not path.exists():
            raise FileNotFoundError(f"Stitch artifact not found: {artifact_id}")
        wire = json.loads(path.read_text())
        if _digest(wire) != digest:
            raise ValueError("Stitch artifact digest mismatch")
        return wire

    def _load_program(
        self,
        mechanism: QualifiedProgramMechanism,
    ) -> str:
        artifact = self.program_artifacts.load_artifact(
            mechanism.source_program_artifact_id
        )
        if artifact.get("schema") != "ecsa.dreamcoder-program.v1":
            raise ValueError(
                "qualified mechanism source is not a DreamCoder program artifact"
            )
        program = artifact.get("program")
        if not isinstance(program, str) or not program:
            raise ValueError("DreamCoder artifact does not contain a program")
        return program

    def _ensure_binary(self) -> Path:
        root = self._ensure_source()
        binary = root / "target" / "release" / "compress"
        if binary.exists():
            return binary
        cargo = shutil.which("cargo")
        if cargo is None:
            raise RuntimeError("Rust cargo is required to build Stitch")
        process = subprocess.run(
            [cargo, "build", "--release", "--bin", "compress"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise RuntimeError(
                "failed to build Stitch: "
                + (process.stderr.strip() or process.stdout.strip())
            )
        if not binary.exists():
            raise RuntimeError("Stitch build completed without compress binary")
        return binary

    def _ensure_source(self) -> Path:
        if self.source_root is not None:
            return self.source_root

        root = self.cache_root / (
            "stitch-source-" + STITCH_REVISION[:12]
        )
        if not root.exists():
            root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    STITCH_REPOSITORY,
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
                    STITCH_REVISION,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        current = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if current != STITCH_REVISION:
            raise ValueError(
                "cached Stitch source is not at the pinned revision"
            )
        return root

    def _write_artifact(self, digest: str, wire: dict) -> None:
        path = self._artifact_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            wire,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        if path.exists():
            if path.read_text() != serialized:
                raise ValueError("conflicting Stitch artifact content")
            return
        path.write_text(serialized)

    def _artifact_path(self, digest: str) -> Path:
        return self.cache_root / "stitch" / f"{digest}.json"
