from __future__ import annotations

import importlib.util
import json
import subprocess
from hashlib import sha256
from pathlib import Path
from types import ModuleType

from ecsa.repair import (
    RepairFamily,
    TheoryProposal,
    TheorySpaceExpansionRequest,
)

BCS_REPOSITORY = "https://github.com/boltholds/Behavioral-Causal-State.git"
BCS_REVISION = "ae8d331e3a1bcac55c740e35720672c7d2bf7106"


def _object_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


class TTTRepairEngine:
    """Real LearnLib TTT repair backend using the existing BCS experiment bridge.

    The engine keeps the BCS TTT implementation external: it checks out the
    pinned BCS source revision into a cache, imports its experiment runner, and
    stores the accepted Mealy hypothesis as a content-addressed local artifact.
    """

    engine_id = "learnlib_ttt"
    family = RepairFamily.STATE_MEMORY

    def __init__(
        self,
        *,
        cache_root: Path,
        fixture_id: str,
        source_root: Path | None = None,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.fixture_id = fixture_id
        self.source_root = Path(source_root) if source_root is not None else None

    def propose(
        self,
        request: TheorySpaceExpansionRequest,
    ) -> tuple[TheoryProposal, ...]:
        runner = self._load_runner()
        fixtures = {
            fixture["id"]: fixture
            for fixture in runner.load_fixtures()
        }
        try:
            fixture = fixtures[self.fixture_id]
        except KeyError as exc:
            raise ValueError(
                f"unknown BCS TTT fixture: {self.fixture_id}"
            ) from exc

        build = runner.prepare_java(self.cache_root / "learnlib")
        trial = runner.run_trial(fixture, 0, build)
        if trial.get("status") != "passed":
            raise RuntimeError(
                f"LearnLib TTT trial did not pass: {trial.get('status')}"
            )
        if trial.get("remaining_counterexample") is not None:
            raise RuntimeError(
                "LearnLib TTT returned a hypothesis with a remaining counterexample"
            )

        hypothesis = trial["hypothesis"]
        digest = _object_digest(hypothesis)
        artifact_id = f"ttt-mealy:sha256:{digest}"
        artifact = {
            "schema": "ecsa.ttt-mealy.v1",
            "source_repository": BCS_REPOSITORY,
            "source_revision": BCS_REVISION,
            "fixture_id": self.fixture_id,
            "learner_class": trial["learner_class"],
            "status": trial["status"],
            "hypothesis": hypothesis,
            "hypothesis_states": trial["hypothesis_states"],
            "minimal_reference_states": trial["minimal_reference_states"],
            "alphabet_size": trial["alphabet_size"],
            "mq_count": trial["mq_count"],
            "eq_count": trial["eq_count"],
            "counterexamples": trial["counterexamples"],
            "remaining_counterexample": trial["remaining_counterexample"],
        }
        self._write_artifact(artifact_id, artifact)

        parents = tuple(
            theory.theory_id
            for theory in request.active_theories
        )
        evidence = (
            request.anomaly.observation.experiment_id,
        )
        proposal = TheoryProposal(
            proposal_id=(
                f"learnlib-ttt:{self.fixture_id}:{digest[:16]}"
            ),
            engine_id=self.engine_id,
            family=self.family,
            artifact_id=artifact_id,
            parent_theory_ids=parents,
            evidence_experiment_ids=evidence,
            summary=(
                f"LearnLib TTT learned a "
                f"{trial['hypothesis_states']}-state Mealy machine "
                f"for {self.fixture_id} "
                f"(alphabet={trial['alphabet_size']}, "
                f"MQ={trial['mq_count']}, EQ={trial['eq_count']})."
            ),
        )
        return (proposal,)

    def load_fixture(self, fixture_id: str | None = None) -> dict:
        target = fixture_id or self.fixture_id
        runner = self._load_runner()
        for fixture in runner.load_fixtures():
            if fixture["id"] == target:
                return fixture
        raise ValueError(f"unknown BCS TTT fixture: {target}")

    def load_artifact(self, artifact_id: str) -> dict:
        digest = self._artifact_digest(artifact_id)
        path = self._artifact_path(digest)
        if not path.exists():
            raise FileNotFoundError(
                f"TTT artifact not found: {artifact_id}"
            )
        artifact = json.loads(path.read_text())
        if artifact.get("schema") != "ecsa.ttt-mealy.v1":
            raise ValueError("unsupported TTT artifact schema")
        actual = _object_digest(artifact.get("hypothesis"))
        if actual != digest:
            raise ValueError("TTT artifact hypothesis digest mismatch")
        return artifact

    def _load_runner(self) -> ModuleType:
        root = self._ensure_source()
        runner_path = root / "experiments" / "ttt" / "runner.py"
        if not runner_path.exists():
            raise FileNotFoundError(
                f"BCS TTT runner missing at {runner_path}"
            )
        module_name = (
            "_ecsa_bcs_ttt_"
            + sha256(str(runner_path).encode()).hexdigest()[:16]
        )
        spec = importlib.util.spec_from_file_location(
            module_name,
            runner_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError("cannot load BCS TTT runner")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _ensure_source(self) -> Path:
        if self.source_root is not None:
            return self.source_root

        root = self.cache_root / (
            "bcs-source-" + BCS_REVISION[:12]
        )
        if not root.exists():
            root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    BCS_REPOSITORY,
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
                    BCS_REVISION,
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
        if current != BCS_REVISION:
            raise ValueError(
                "cached BCS TTT source is not at the pinned revision"
            )
        return root

    def _write_artifact(
        self,
        artifact_id: str,
        artifact: dict,
    ) -> None:
        digest = self._artifact_digest(artifact_id)
        path = self._artifact_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
        )
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != artifact:
                raise ValueError(
                    "conflicting content for TTT artifact id"
                )
            return
        path.write_text(payload + "\n")

    def _artifact_path(self, digest: str) -> Path:
        return self.cache_root / "artifacts" / f"{digest}.json"

    @staticmethod
    def _artifact_digest(artifact_id: str) -> str:
        prefix = "ttt-mealy:sha256:"
        if not artifact_id.startswith(prefix):
            raise ValueError("invalid TTT artifact id")
        digest = artifact_id[len(prefix):]
        if len(digest) != 64 or any(
            c not in "0123456789abcdef"
            for c in digest
        ):
            raise ValueError("invalid TTT artifact digest")
        return digest
