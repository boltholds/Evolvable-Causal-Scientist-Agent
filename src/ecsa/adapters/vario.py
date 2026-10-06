from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from ecsa.adapters.stitch import StitchAbstractionProposal
from ecsa.repair import RepairFamily

VARIO_SOURCE_URL = (
    "https://vreeken.groups.cispa.de/prj/vario/vario-v20220815.zip"
)
VARIO_SOURCE_SHA256 = (
    "67df25a256622f86fe7c7b469e2928ddcd2252680b18502699786041ca554652"
)


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(payload).hexdigest()


class MechanismTransferScope(StrEnum):
    SHARED = "shared"
    CONTEXT_SPECIALIZED = "context_specialized"
    SPLIT = "split"


@dataclass(frozen=True)
class VarioContextEvidence:
    context_id: str
    x: tuple[float, ...]
    y: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.context_id:
            raise ValueError("context_id is required")
        if not isinstance(self.x, tuple) or not isinstance(self.y, tuple):
            raise ValueError("context observations must be immutable tuples")
        if len(self.x) != len(self.y) or len(self.x) < 4:
            raise ValueError("x and y must have equal length >= 4")
        for value in self.x + self.y:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError("VARIO evidence must contain finite numeric values")

    @property
    def digest(self) -> str:
        return _digest(
            {
                "context_id": self.context_id,
                "x": [float(value) for value in self.x],
                "y": [float(value) for value in self.y],
            }
        )


@dataclass(frozen=True)
class VarioTransferDecision:
    scope: MechanismTransferScope
    partition: tuple[tuple[str, ...], ...]
    artifact_id: str
    vario_score: float
    source_stitch_artifact_id: str


class VarioTransferValidator:
    """Validate whether a Stitch abstraction is invariant across contexts.

    Official VARIO discovers a partition of contexts whose fitted mechanism
    coefficients are invariant within groups. ECSA maps that partition to a
    transfer scope:

    - one group: SHARED
    - multiple groups with at least one reused group: CONTEXT_SPECIALIZED
    - every context singleton: SPLIT

    The three-way scope is ECSA terminology layered on the official VARIO
    partition; it is not claimed as VARIO's own output vocabulary.
    """

    def __init__(
        self,
        *,
        cache_root: Path,
        x_degree: int = 1,
        rscript: str = "Rscript",
    ) -> None:
        if type(x_degree) is not int or x_degree < 1:
            raise ValueError("x_degree must be a positive integer")
        self.cache_root = Path(cache_root)
        self.x_degree = x_degree
        self.rscript = rscript

    def evaluate(
        self,
        abstraction: StitchAbstractionProposal,
        contexts: tuple[VarioContextEvidence, ...],
    ) -> VarioTransferDecision:
        if not isinstance(abstraction, StitchAbstractionProposal):
            raise ValueError("typed StitchAbstractionProposal required")
        if abstraction.family is not RepairFamily.ABSTRACTION_TRANSFER:
            raise ValueError("VARIO expects an ABSTRACTION_TRANSFER proposal")
        if not isinstance(contexts, tuple) or len(contexts) < 2:
            raise ValueError("at least two contexts are required")
        if not all(isinstance(item, VarioContextEvidence) for item in contexts):
            raise ValueError("typed VarioContextEvidence values required")
        context_ids = tuple(item.context_id for item in contexts)
        if len(set(context_ids)) != len(context_ids):
            raise ValueError("context ids must be unique")

        rscript_path = shutil.which(self.rscript)
        if rscript_path is None:
            raise RuntimeError(
                "Rscript is required for the official VARIO backend"
            )
        source_root = self._ensure_source()

        self.cache_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="vario-run-",
            dir=self.cache_root,
        ) as tmp:
            work = Path(tmp)
            data_dir = work / "contexts"
            data_dir.mkdir()
            for index, evidence in enumerate(contexts):
                path = data_dir / f"context-{index:04d}.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("x1", "y"))
                    writer.writerows(
                        zip(
                            (float(value) for value in evidence.x),
                            (float(value) for value in evidence.y),
                        )
                    )

            script_path = work / "run_vario.R"
            script_path.write_text(self._r_script())
            process = subprocess.run(
                [
                    rscript_path,
                    str(script_path),
                    str(source_root),
                    str(data_dir),
                    str(self.x_degree),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    "official VARIO execution failed: "
                    + (process.stderr.strip() or process.stdout.strip())
                )
            labels, score = self._parse_output(process.stdout)

        partition = self._canonical_partition(context_ids, labels)
        scope = self._scope(partition, len(contexts))
        wire = {
            "schema": "ecsa.vario-transfer.v1",
            "source_url": VARIO_SOURCE_URL,
            "source_sha256": VARIO_SOURCE_SHA256,
            "source_license": None,
            "source_license_note": (
                "The official VARIO v20220815 archive contains no LICENSE file; "
                "ECSA does not redistribute the source."
            ),
            "source_stitch_proposal_id": abstraction.proposal_id,
            "source_stitch_artifact_id": abstraction.artifact_id,
            "source_stitch_body": abstraction.body,
            "x_degree": self.x_degree,
            "contexts": [
                {
                    "context_id": evidence.context_id,
                    "sample_count": len(evidence.x),
                    "evidence_sha256": evidence.digest,
                }
                for evidence in contexts
            ],
            "vario_partition": [list(group) for group in partition],
            "vario_score": score,
            "ecsa_scope": scope.value,
            "scope_mapping": (
                "ECSA mapping: one VARIO group=shared; multiple groups with "
                "reuse=context_specialized; all singleton groups=split."
            ),
        }
        digest = _digest(wire)
        artifact_id = f"vario-transfer:sha256:{digest}"
        self._write_artifact(digest, wire)

        return VarioTransferDecision(
            scope=scope,
            partition=partition,
            artifact_id=artifact_id,
            vario_score=score,
            source_stitch_artifact_id=abstraction.artifact_id,
        )

    def load_artifact(self, artifact_id: str) -> dict:
        prefix = "vario-transfer:sha256:"
        if not artifact_id.startswith(prefix):
            raise ValueError("invalid VARIO artifact id")
        digest = artifact_id[len(prefix):]
        if len(digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in digest
        ):
            raise ValueError("invalid VARIO artifact digest")
        path = self._artifact_path(digest)
        if not path.exists():
            raise FileNotFoundError(f"VARIO artifact not found: {artifact_id}")
        wire = json.loads(path.read_text())
        if _digest(wire) != digest:
            raise ValueError("VARIO artifact digest mismatch")
        return wire

    def _ensure_source(self) -> Path:
        target = self.cache_root / (
            "vario-source-" + VARIO_SOURCE_SHA256[:12]
        )
        root = target / "Vario_R_kdd22_submitted"
        if root.exists():
            return root

        target.mkdir(parents=True, exist_ok=True)
        archive_bytes = urlopen(VARIO_SOURCE_URL, timeout=90).read()
        actual = sha256(archive_bytes).hexdigest()
        if actual != VARIO_SOURCE_SHA256:
            raise ValueError(
                "official VARIO source archive checksum mismatch"
            )

        with ZipFile(BytesIO(archive_bytes)) as archive:
            names = archive.namelist()
            if any(
                Path(name).is_absolute() or ".." in Path(name).parts
                for name in names
            ):
                raise ValueError("unsafe path in VARIO source archive")
            archive.extractall(target)

        if not root.exists():
            raise RuntimeError("official VARIO archive layout changed")
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
                raise ValueError("conflicting VARIO artifact content")
            return
        path.write_text(serialized)

    def _artifact_path(self, digest: str) -> Path:
        return self.cache_root / "vario" / f"{digest}.json"

    @staticmethod
    def _parse_output(stdout: str) -> tuple[tuple[int, ...], float]:
        partition_line = None
        score_line = None
        for line in stdout.splitlines():
            if line.startswith("ECSA_PARTITION="):
                partition_line = line.split("=", 1)[1].strip()
            elif line.startswith("ECSA_SCORE="):
                score_line = line.split("=", 1)[1].strip()
        if partition_line is None or score_line is None:
            raise RuntimeError(
                "VARIO wrapper did not emit partition/score markers: "
                + stdout[-2000:]
            )
        labels = tuple(
            int(value)
            for value in partition_line.split(",")
            if value
        )
        score = float(score_line)
        if not labels or not math.isfinite(score):
            raise RuntimeError("invalid VARIO partition/score output")
        return labels, score

    @staticmethod
    def _canonical_partition(
        context_ids: tuple[str, ...],
        labels: tuple[int, ...],
    ) -> tuple[tuple[str, ...], ...]:
        if len(labels) != len(context_ids):
            raise RuntimeError("VARIO partition length does not match contexts")
        groups: dict[int, list[str]] = {}
        order: list[int] = []
        for context_id, label in zip(context_ids, labels):
            if label not in groups:
                groups[label] = []
                order.append(label)
            groups[label].append(context_id)
        return tuple(tuple(groups[label]) for label in order)

    @staticmethod
    def _scope(
        partition: tuple[tuple[str, ...], ...],
        context_count: int,
    ) -> MechanismTransferScope:
        if len(partition) == 1:
            return MechanismTransferScope.SHARED
        if len(partition) == context_count:
            return MechanismTransferScope.SPLIT
        return MechanismTransferScope.CONTEXT_SPECIALIZED

    @staticmethod
    def _r_script() -> str:
        return textwrap.dedent(
            r'''
            args <- commandArgs(trailingOnly=TRUE)
            root <- args[[1]]
            data_dir <- args[[2]]
            x_degree <- as.integer(args[[3]])

            suppressPackageStartupMessages(library(rlist))
            suppressPackageStartupMessages(library(combinat))
            suppressPackageStartupMessages(library(dplyr))
            suppressPackageStartupMessages(library(car))

            util_files <- sort(list.files(
              file.path(root, "R", "utils"),
              pattern="\\.R$",
              full.names=TRUE
            ))
            for (path in util_files) {
              sys.source(path, envir=.GlobalEnv)
            }

            core_files <- sort(list.files(
              file.path(root, "R"),
              pattern="\\.R$",
              full.names=TRUE
            ))
            core_files <- core_files[basename(core_files) != "init.R"]
            for (path in core_files) {
              sys.source(path, envir=.GlobalEnv)
            }

            context_files <- sort(list.files(
              data_dir,
              pattern="^context-[0-9]+\\.csv$",
              full.names=TRUE
            ))
            data_E1_En <- lapply(context_files, read.csv)
            set.seed(1)
            config <- list(
              verbose=FALSE,
              greedy_k=FALSE,
              meta_computing=FALSE,
              version_mdl=TRUE,
              version_simp=TRUE,
              conf_precomputed=list()
            )

            result <- Vario_Pi_search(
              data_E1_En,
              c(x_degree),
              config
            )
            ranking <- result$Pi_ranking
            ranking_kind <- "mdl"
            if (
              isTRUE(result$invariant_conservative) &&
              length(result$Pi_ranking_correctone) > 0
            ) {
              ranking <- result$Pi_ranking_correctone
              ranking_kind <- "invariance-corrected"
            }
            best <- ranking[[1]]
            if (!is.null(best$Pi_history)) {
              partition <- best$Pi_history$E_part
              score <- best$score_sum
            } else {
              partition <- best$E_part
              score <- best$score_sum
            }
            cat(
              paste0(
                "ECSA_PARTITION=",
                paste(partition, collapse=","),
                "\n"
              )
            )
            cat(
              paste0(
                "ECSA_SCORE=",
                format(score, digits=17, scientific=FALSE),
                "\n"
              )
            )
            '''
        )
