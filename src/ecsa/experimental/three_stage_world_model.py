"""Three-stage frozen world-model audit: baseline, TextWorld SFT, BehR.

The fixture is deterministic and synthetic. Only witnessed X/Y alternatives
serve as labels; no world-law oracle enters learner, projector or comparator.
Actual 7B checkpoints are loaded explicitly by the CLI, never by import.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch import Tensor

from ecsa.world_model.text_relations import TextRelationSample

from .verbalization_study import StudyLaw
from .verified_relation_kan import (
    NegativeTraining, PoolKind, ProjectionKind, StudyConfig, StudyRow,
    WitnessedPair, _metrics, benchmark_one, fit_kan_with_witnesses,
    fixture_replay_witnesses, mine_repeat_supported_pairs,
)
from .world_model_encoders import (
    Candidate, CausalLMConfig, FrozenCausalLMTokenPort, LikelihoodProbe,
    SPECS, evaluate_next_state_likelihood,
)


class Stage(StrEnum):
    BASE = "base"
    TEXTWORLD_SFT = "textworld_sft"
    BEHR = "behr"


class Scale(StrEnum):
    SMOKE = "smoke"
    FULL = "full"


@dataclass(frozen=True)
class PinnedModel:
    stage: Stage
    candidate: Candidate
    revision: str


PINNED: tuple[PinnedModel, ...] = (
    PinnedModel(Stage.BASE, Candidate.BASE_QWEN25,
                "d149729398750b98c0af14eb82c78cfe92750796"),
    PinnedModel(Stage.TEXTWORLD_SFT, Candidate.TEXTWORLD_SFT,
                "b052a201ae867c3058efba17c9af9cb1635d1f09"),
    PinnedModel(Stage.BEHR, Candidate.BEHR_TEXTWORLD,
                "6a4326a60540cc33ffa42ee7a16fc2bad8f6f613"),
)


@dataclass(frozen=True)
class RunSettings:
    scale: Scale
    config: StudyConfig
    seeds: tuple[int, ...]
    pools: tuple[PoolKind, ...]
    projections: tuple[ProjectionKind, ...]
    dimensions: tuple[int, ...]
    likelihood_pairs: int


def settings(scale: Scale) -> RunSettings:
    if scale is Scale.SMOKE:
        return RunSettings(scale, StudyConfig(24, 8, 12, 3, 20, 0.009, 5),
                           (0,), (PoolKind.SENTENCE_FULL,),
                           (ProjectionKind.PCA,), (2, 8), 3)
    if scale is Scale.FULL:
        return RunSettings(scale, StudyConfig(48, 16, 24, 3, 100, 0.009, 10),
                           (0, 1, 2), tuple(PoolKind), tuple(ProjectionKind),
                           (2, 8, 16, 32), 24)
    raise ValueError(f"unknown scale: {scale}")


def pinned(stage: Stage) -> PinnedModel:
    return next(spec for spec in PINNED if spec.stage is stage)


def split_pairs(law: StudyLaw, seed: int, config: StudyConfig,
                ) -> tuple[tuple[WitnessedPair, ...],
                           tuple[WitnessedPair, ...],
                           tuple[WitnessedPair, ...]]:
    inputs = (
        (seed * 101 + 9, config.train_count, False, seed * 197 + 11),
        (seed * 101 + 59, config.calibration_count, False, seed * 197 + 19),
        (seed * 101 + 109, config.heldout_count, True, seed * 197 + 29),
    )
    splits: list[tuple[WitnessedPair, ...]] = []
    ids: list[set[str]] = []
    for fixture_seed, count, heldout, miner_seed in inputs:
        witnesses = fixture_replay_witnesses(
            law, fixture_seed, count, heldout=heldout,
            repetitions=config.replays_per_x,
        )
        ids.append({w.experiment_id for w in witnesses})
        mined = mine_repeat_supported_pairs(
            witnesses, seed=miner_seed, minimum_replays=config.replays_per_x,
        )
        if len(mined.samples) != count or mined.audit.no_disjoint_alternative:
            raise RuntimeError("incomplete witnessed fixture cohort")
        splits.append(mined.samples)
    if (ids[0] & ids[1]) or (ids[0] & ids[2]) or (ids[1] & ids[2]):
        raise ValueError("train/calibration/heldout intervention IDs overlap")
    return splits[0], splits[1], splits[2]


def likelihood_witnesses(
    law: StudyLaw, seed: int, config: StudyConfig,
) -> tuple[tuple[TextRelationSample, ...], tuple[TextRelationSample, ...]]:
    _, _, heldout = split_pairs(law, seed, config)
    true: list[TextRelationSample] = []
    alternative: list[TextRelationSample] = []
    for pair in heldout:
        base = (pair.positive_experiment_id, pair.schema_id, pair.x_text,
                pair.positive_experiment_id + ":before",
                pair.positive_experiment_id + ":after")
        true.append(TextRelationSample(base[0], base[1], base[2], pair.y_true,
                                       base[3], base[4], True))
        alternative.append(TextRelationSample(base[0], base[1], base[2],
                                              pair.y_alternative, base[3], base[4], True))
    return tuple(true), tuple(alternative)


@dataclass(frozen=True)
class StructuredVectors:
    x: np.ndarray
    positive: np.ndarray
    negative: np.ndarray


STRUCTURED_WIDTH = 10
OPERATORS = (">", ">=", "<", "<=", "==", "!=")


def structured_vectors(pairs: tuple[WitnessedPair, ...]) -> StructuredVectors:
    """Transparent X/Y features, no world-law response or hidden model labels."""
    if not pairs:
        raise ValueError("at least one witnessed pair required")
    xs: list[list[float]] = []
    positives: list[list[float]] = []
    negatives: list[list[float]] = []
    for pair in pairs:
        before = json.loads(pair.x_text)["before"]
        u = float(before["readings"]["primary"])
        v = float(before["readings"]["reference"])
        op = before["predicate"]["operator"]
        if op not in OPERATORS:
            raise ValueError("unrecognized observed operator")
        xs.append([u, v, u - v, u * v] + [float(op == x) for x in OPERATORS])
        def y_features(text: str) -> list[float]:
            measurement = float(json.loads(text)["after"]["measurement"])
            # Exact output is observed, not predicted by an oracle.
            return [measurement] + [0.0] * (STRUCTURED_WIDTH - 1)
        positives.append(y_features(pair.y_true))
        negatives.append(y_features(pair.y_alternative))
    x = np.asarray(xs, dtype=np.float32)
    p = np.asarray(positives, dtype=np.float32)
    n = np.asarray(negatives, dtype=np.float32)
    if (x.shape != p.shape or p.shape != n.shape or
            x.shape[1] != STRUCTURED_WIDTH or
            not all(np.isfinite(a).all() for a in (x, p, n))):
        raise ValueError("invalid structured numeric vectors")
    return StructuredVectors(x, p, n)


@dataclass(frozen=True)
class StructuredNormalizer:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray

    @classmethod
    def fit(cls, train: StructuredVectors) -> StructuredNormalizer:
        def scale(x: np.ndarray) -> np.ndarray:
            sigma = x.std(axis=0)
            return np.where(sigma > 1e-4, sigma, 1.0).astype(np.float32)
        return cls(train.x.mean(axis=0), scale(train.x),
                   train.positive.mean(axis=0), scale(train.positive))

    def transform(self, data: StructuredVectors) -> tuple[Tensor, Tensor, Tensor]:
        # Frozen train-only means/variances, shared between true and wrong Y.
        x = (data.x - self.x_mean) / self.x_scale
        p = (data.positive - self.y_mean) / self.y_scale
        n = (data.negative - self.y_mean) / self.y_scale
        return tuple(torch.from_numpy(v.astype(np.float32)) for v in (x, p, n))


@dataclass(frozen=True)
class StructuredScore:
    law: str
    seed: int
    train_pairs: int
    calibration_pairs: int
    heldout_pairs: int
    heldout_auroc: float
    heldout_paired_accuracy: float
    heldout_brier: float
    heldout_contrastive_nll: float
    calibration_loss: float
    kan_parameters: int
    latent_width: int


def run_structured_control(scale: Scale, *, source_revision: str = "") -> dict[str, object]:
    run = settings(scale)
    scores: list[StructuredScore] = []
    for law in (StudyLaw.PRECISION, StudyLaw.OPERATOR):
        for seed in run.seeds:
            tr, cal, held = split_pairs(law, seed, run.config)
            train = structured_vectors(tr)
            normalizer = StructuredNormalizer.fit(train)
            model, loss = fit_kan_with_witnesses(
                normalizer.transform(train),
                normalizer.transform(structured_vectors(cal)),
                dimension=STRUCTURED_WIDTH, seed=seed, config=run.config,
            )
            auc, accuracy, brier, nll = _metrics(
                model, normalizer.transform(structured_vectors(held)),
            )
            scores.append(StructuredScore(
                law.value, seed, len(tr), len(cal), len(held),
                auc, accuracy, brier, nll, loss,
                sum(p.numel() for p in model.parameters()), STRUCTURED_WIDTH,
            ))
    return {
        "scale": scale.value, "source_revision": source_revision,
        "config": asdict(run.config),
        "control": "observed_numeric_and_operator_only",
        "uses_oracle_labels": False,
        "uses_pretrained_encoder": False,
        "identity_overlap": 0,
        "rows": [asdict(s) for s in scores],
        "limitations": [
            "The control includes an explicit u-v feature and therefore has a favorable inductive bias.",
            "Synthetic deterministic world only; compare against checkpoints on identical splits, not independent trials.",
        ],
    }


def run_candidate_study(
    port: FrozenCausalLMTokenPort, stage: Stage, scale: Scale,
    *, source_revision: str = "",
) -> dict[str, object]:
    run = settings(scale)
    spec = pinned(stage)
    if port.model_id != SPECS[spec.candidate].model_id or port.revision != spec.revision:
        raise ValueError("loaded checkpoint does not match pinned stage revision")
    if not port.loaded_checkpoint_by_port:
        raise ValueError("mock-injected port is not a real checkpoint")
    kan: list[StudyRow] = []
    likelihood: list[LikelihoodProbe] = []
    for law in (StudyLaw.PRECISION, StudyLaw.OPERATOR):
        for seed in run.seeds:
            kan.extend(benchmark_one(
                port=port, law=law, seed=seed, config=run.config,
                pools=run.pools, projections=run.projections,
                dimensions=run.dimensions,
                pairing=(NegativeTraining.WITNESSED,),
            ))
            true, wrong = likelihood_witnesses(law, seed, run.config)
            likelihood.append(evaluate_next_state_likelihood(
                port, true, wrong, law=law.value, seed=seed,
                maximum=run.likelihood_pairs,
            ))
    return {
        "stage": stage.value, "model": port.model_id,
        "revision": port.revision, "scale": scale.value,
        "source_revision": source_revision,
        "config": asdict(run.config),
        "rows": [asdict(x) for x in kan],
        "likelihood": [asdict(x) for x in likelihood],
        "replays_are_deterministic_fixture_only": True,
        "no_oracle_label_in_negative_miner": True,
        "no_kan_to_encoder_gradient": True,
        "limitations": [
            "Synthetic out-of-distribution JSON; no evidence of causal identification or TextWorld task success.",
            "Teacher-forced NLL compares complete Y text; surface priors and token frequencies may confound it.",
            "Different checkpoints may tokenize the same text differently; compare paired choices within a checkpoint first.",
            "Only the KAN heads are optimized; the encoders and their features stay frozen.",
        ],
    }


@dataclass(frozen=True)
class ComparisonCell:
    law: str
    dimension: int
    seed: int
    base_auroc: float
    sft_auroc: float
    behr_auroc: float
    sft_minus_base: float
    behr_minus_sft: float
    behr_minus_base: float


@dataclass(frozen=True)
class LikelihoodComparison:
    law: str
    seed: int
    pairs_examined: int
    base_accuracy: float
    sft_accuracy: float
    behr_accuracy: float
    base_margin: float
    sft_margin: float
    behr_margin: float


def compare_reports(
    stage_reports: dict[Stage, dict[str, object]], control: dict[str, object],
) -> dict[str, object]:
    if set(stage_reports) != set(Stage):
        raise ValueError("comparison requires all three stages")
    base = stage_reports[Stage.BASE]
    scale = base["scale"]
    if (control["scale"] != scale or control["config"] != base["config"] or
            control["source_revision"] != base["source_revision"]):
        raise ValueError("structured control configuration differs")
    if control["uses_oracle_labels"] is not False or control["identity_overlap"] != 0:
        raise ValueError("structured control is not an isolated witnessed baseline")
    lookup: dict[Stage, dict[tuple[str, int, str, str, str, int], dict[str, object]]] = {}
    likelihood: dict[Stage, dict[tuple[str, int], dict[str, object]]] = {}
    for stage, report in stage_reports.items():
        spec = pinned(stage)
        if (report["model"] != SPECS[spec.candidate].model_id or
                report["revision"] != spec.revision or
                report["stage"] != stage.value or
                report["scale"] != scale or report["config"] != base["config"] or
                report["source_revision"] != base["source_revision"] or
                report["no_kan_to_encoder_gradient"] is not True or
                report["replays_are_deterministic_fixture_only"] is not True or
                report["no_oracle_label_in_negative_miner"] is not True):
            raise ValueError(f"stage metadata mismatch: {stage.value}")
        rows: dict[tuple[str, int, str, str, str, int], dict[str, object]] = {}
        for item in report["rows"]:
            key = (item["law"], item["seed"], item["pool"],
                   item["projection"], item["negative_training"], item["dimension"])
            if key in rows or item["negative_training"] != NegativeTraining.WITNESSED.value:
                raise ValueError("duplicate or unsafe KAN cell")
            if item["identity_overlap"] != 0:
                raise ValueError("train/test identity overlap")
            rows[key] = item
        lookup[stage] = rows
        probes: dict[tuple[str, int], dict[str, object]] = {}
        for item in report["likelihood"]:
            key = (item["law"], item["seed"])
            if (key in probes or item["pairs_examined"] < 1 or
                    item["pair_accuracy"] is None or item["mean_nll_margin"] is None):
                raise ValueError("invalid likelihood report")
            probes[key] = item
        likelihood[stage] = probes
    keys = set(lookup[Stage.BASE])
    if not keys or any(set(rows) != keys for rows in lookup.values()):
        raise ValueError("KAN experiment cells differ among stages")
    for key in keys:
        for attribute in ("train_pairs", "calibration_pairs", "heldout_pairs",
                          "kan_parameters", "identity_overlap"):
            if len({lookup[s][key][attribute] for s in Stage}) != 1:
                raise ValueError(f"KAN cohort mismatch: {attribute}")
    grouped: dict[tuple[str, int, int], list[tuple[float, float, float]]] = {}
    for key in sorted(keys):
        law, seed, _, _, _, dimension = key
        values = tuple(float(lookup[s][key]["heldout_auroc"]) for s in Stage)
        grouped.setdefault((law, dimension, seed), []).append(values)
    cells = []
    for (law, dimension, seed), scores in sorted(grouped.items()):
        a, b, c = (mean(x[index] for x in scores) for index in (0, 1, 2))
        cells.append(ComparisonCell(law, dimension, seed, a, b, c,
                                    b-a, c-b, c-a))
    lk_keys = set(likelihood[Stage.BASE])
    if not lk_keys or any(set(p) != lk_keys for p in likelihood.values()):
        raise ValueError("likelihood cohorts differ among stages")
    probes = []
    for law, seed in sorted(lk_keys):
        values = [likelihood[s][(law, seed)] for s in Stage]
        counts = {x["pairs_examined"] for x in values}
        if len(counts) != 1:
            raise ValueError("different likelihood pair counts")
        probes.append(LikelihoodComparison(
            law, seed, int(values[0]["pairs_examined"]),
            *(float(x["pair_accuracy"]) for x in values),
            *(float(x["mean_nll_margin"]) for x in values),
        ))
    control_rows = control["rows"]
    if {(r["law"], r["seed"]) for r in control_rows} != lk_keys:
        raise ValueError("structured-control law/seed cohorts differ")
    if len(control_rows) != len(lk_keys):
        raise ValueError("duplicate structured-control row")
    expected_rows = len(settings(Scale(scale)).seeds) * 2 * len(settings(Scale(scale)).pools) * len(settings(Scale(scale)).projections) * len(settings(Scale(scale)).dimensions)
    if len(keys) != expected_rows:
        raise ValueError("partial KAN experiment")
    return {
        "scale": scale, "source_revision": base["source_revision"],
        "config": base["config"],
        "stage_models": {s.value: {"model": stage_reports[s]["model"],
                                   "revision": stage_reports[s]["revision"]}
                         for s in Stage},
        "matched_kan_cells_per_stage": len(keys),
        "kan_by_law_dimension_seed": [asdict(c) for c in cells],
        "likelihood_by_law_seed": [asdict(c) for c in probes],
        "structured_control_by_law_seed": control_rows,
        "cautions": [
            "Mean KAN AUROC aggregates architecture cells; those cells are not independent samples.",
            "Treat seed as the replication unit; do not infer significance from 96 KAN cells.",
            "SFT-minus-base and BehR-minus-SFT isolate sequential checkpoint differences, not causal effects of training algorithms.",
            "Likelihood scores depend on the chosen template and tokenization; margins are within-checkpoint comparisons.",
            "Structured control uses privileged parsed numeric features, and tests head learnability rather than encoder quality.",
            "These fixtures are synthetic and not a TextWorld or deployed-agent benchmark.",
        ],
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    study = sub.add_parser("study")
    study.add_argument("--stage", required=True, choices=[s.value for s in Stage])
    study.add_argument("--scale", required=True, choices=[s.value for s in Scale])
    study.add_argument("--output", type=Path, required=True)
    study.add_argument("--max-length", type=int, default=1024)
    study.add_argument("--source-revision", default="")
    study.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    study.add_argument("--4bit", dest="quantized", action="store_true")
    for command in ("control", "compare"):
        p = sub.add_parser(command)
        p.add_argument("--directory", type=Path, required=True)
        if command == "control":
            p.add_argument("--source-revision", default="")
            p.add_argument("--scale", required=True, choices=[s.value for s in Scale])
    args = parser.parse_args()
    if args.action == "study":
        stage = Stage(args.stage)
        spec = pinned(stage)
        if args.device != "cuda" or not args.quantized:
            parser.error("three-stage 7B inference requires --device cuda --4bit")
        port = FrozenCausalLMTokenPort(CausalLMConfig(
            candidate=spec.candidate, revision=spec.revision,
            device=args.device, quantized_4bit=args.quantized,
            max_length=args.max_length,
        ))
        report = run_candidate_study(port, stage, Scale(args.scale),
                                     source_revision=args.source_revision)
        write_report(args.output, report)
        print(f"Validated {stage.value}: {len(report['rows'])} KAN cells; {len(report['likelihood'])} likelihood probes")
    elif args.action == "control":
        scale = Scale(args.scale)
        write_report(args.directory / "structured_control.json", run_structured_control(
            scale, source_revision=args.source_revision))
        print("Validated structured numeric/operator control")
    else:
        directory = args.directory
        inputs = {stage: json.loads((directory / f"{stage.value}.json").read_text(encoding="utf8"))
                  for stage in Stage}
        control = json.loads((directory / "structured_control.json").read_text(encoding="utf8"))
        output = compare_reports(inputs, control)
        write_report(directory / "comparison.json", output)
        print(f"Validated three stages: {output['matched_kan_cells_per_stage']} matched KAN cells")


if __name__ == "__main__":
    main()
