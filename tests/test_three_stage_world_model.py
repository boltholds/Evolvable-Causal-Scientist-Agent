"""Offline contracts for three-stage checkpoint, replay and structured controls."""
from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from ecsa.experimental.three_stage_world_model import (
    PINNED, STRUCTURED_WIDTH, Stage, Scale, StructuredNormalizer,
    compare_reports, likelihood_witnesses, run_structured_control,
    settings, split_pairs, structured_vectors,
)
from ecsa.experimental.verified_relation_kan import NegativeTraining
from ecsa.experimental.verbalization_study import StudyLaw
from ecsa.experimental.world_model_encoders import Candidate, SPECS


RUNNER = Path(__file__).resolve().parents[1] / "scripts/benchmarks/run_three_stage_world_model.sh"


def test_three_stage_registry_is_pinned_and_all_7b():
    assert [spec.stage for spec in PINNED] == list(Stage)
    assert [spec.candidate for spec in PINNED] == [
        Candidate.BASE_QWEN25, Candidate.TEXTWORLD_SFT, Candidate.BEHR_TEXTWORLD,
    ]
    assert PINNED[1].revision == "b052a201ae867c3058efba17c9af9cb1635d1f09"
    assert SPECS[Candidate.TEXTWORLD_SFT].model_id == "X1AOX1A/WorldModel-Textworld-Qwen2.5-7B"
    assert settings(Scale.SMOKE).config.train_count == 24
    assert settings(Scale.FULL).config.train_count == 48
    assert len(settings(Scale.FULL).seeds) == 3


def test_witnessed_likelihood_uses_same_heldout_data_without_oracle_negatives():
    config = settings(Scale.SMOKE).config
    for law in (StudyLaw.PRECISION, StudyLaw.OPERATOR):
        train, calibration, heldout = split_pairs(law, 0, config)
        true, wrong = likelihood_witnesses(law, 0, config)
        assert len(true) == len(wrong) == len(heldout) == config.heldout_count
        assert all(a.x_text == b.x_text for a, b in zip(true, wrong))
        assert all(a.y_text != b.y_text for a, b in zip(true, wrong))
        assert [a.y_text for a in true] == [x.y_true for x in heldout]
        assert [b.y_text for b in wrong] == [x.y_alternative for x in heldout]
        assert not ({x.positive_experiment_id for x in train} &
                    {x.positive_experiment_id for x in heldout})
        assert not ({x.positive_experiment_id for x in calibration} &
                    {x.positive_experiment_id for x in heldout})


def test_structured_control_has_no_oracle_and_learns_normalizer_on_train_only():
    train, _, heldout = split_pairs(StudyLaw.PRECISION, 0, settings(Scale.SMOKE).config)
    train_features = structured_vectors(train)
    heldout_features = structured_vectors(heldout)
    assert train_features.x.shape == (24, STRUCTURED_WIDTH)
    assert train_features.positive.shape == train_features.negative.shape
    before = json.loads(train[0].x_text)["before"]
    assert train_features.x[0, 0] == pytest.approx(before["readings"]["primary"])
    assert train_features.x[0, 2] == pytest.approx(
        before["readings"]["primary"] - before["readings"]["reference"], abs=1e-5
    )
    assert np.all(train_features.x[:, 4:].sum(axis=1) == 1)
    assert np.all(train_features.positive[:, 1:] == 0)
    fit = StructuredNormalizer.fit(train_features)
    old = fit.x_mean.copy()
    a, p, n = fit.transform(heldout_features)
    assert a.shape == p.shape == n.shape == (12, STRUCTURED_WIDTH)
    assert np.array_equal(old, fit.x_mean)
    assert not np.allclose(p.numpy(), n.numpy())
    assert all(np.isfinite(x.numpy()).all() for x in (a, p, n))


def test_structured_control_offline_training_is_finite_and_separate_from_model():
    report = run_structured_control(Scale.SMOKE, source_revision="test-commit")
    assert report["source_revision"] == "test-commit"
    assert report["uses_oracle_labels"] is False
    assert report["uses_pretrained_encoder"] is False
    assert len(report["rows"]) == 2
    assert {row["law"] for row in report["rows"]} == {"precision", "operator"}
    for row in report["rows"]:
        assert 0 <= row["heldout_auroc"] <= 1
        assert row["heldout_pairs"] == 12
        assert row["kan_parameters"] > 0


def fake_data():
    run = settings(Scale.SMOKE)
    control = run_structured_control(Scale.SMOKE, source_revision="revision-1")
    reports = {}
    for index, spec in enumerate(PINNED):
        rows = []
        probes = []
        for law in (StudyLaw.PRECISION, StudyLaw.OPERATOR):
            _, _, heldout = split_pairs(law, 0, run.config)
            probes.append({"law": law.value, "seed": 0, "pairs_examined": 3,
                           "pair_accuracy": 0.4 + index * 0.1,
                           "mean_nll_margin": 0.01 * (index + 1)})
            for dimension in run.dimensions:
                rows.append({"law": law.value, "seed": 0,
                             "pool": "sentence_full", "projection": "train_only_pca",
                             "negative_training": NegativeTraining.WITNESSED.value,
                             "dimension": dimension,
                             "train_pairs": run.config.train_count,
                             "calibration_pairs": run.config.calibration_count,
                             "heldout_pairs": run.config.heldout_count,
                             "heldout_auroc": 0.4 + index * 0.1 + dimension / 1000,
                             "kan_parameters": 1101 if dimension == 2 else 3669,
                             "identity_overlap": 0})
        reports[spec.stage] = {
            "stage": spec.stage.value, "scale": "smoke",
            "source_revision": "revision-1", "model": SPECS[spec.candidate].model_id,
            "revision": spec.revision, "config": asdict(run.config),
            "rows": rows, "likelihood": probes,
            "no_kan_to_encoder_gradient": True,
            "replays_are_deterministic_fixture_only": True,
            "no_oracle_label_in_negative_miner": True,
        }
    return reports, control


def test_comparator_matches_every_cell_and_uses_seed_as_replication():
    reports, control = fake_data()
    output = compare_reports(reports, control)
    assert output["matched_kan_cells_per_stage"] == 4
    assert len(output["kan_by_law_dimension_seed"]) == 4
    assert len(output["likelihood_by_law_seed"]) == 2
    assert all(x["sft_minus_base"] == pytest.approx(0.1)
               for x in output["kan_by_law_dimension_seed"])
    assert all(x["behr_minus_sft"] == pytest.approx(0.1)
               for x in output["kan_by_law_dimension_seed"])
    assert all(x["pairs_examined"] == 3 for x in output["likelihood_by_law_seed"])
    assert len(output["structured_control_by_law_seed"]) == 2


def test_comparator_rejects_swapped_checkpoint_unsafe_labels_and_mismatches():
    reports, control = fake_data()
    changed = copy.deepcopy(reports)
    changed[Stage.BEHR]["revision"] = "unvalidated"
    with pytest.raises(ValueError, match="metadata"):
        compare_reports(changed, control)
    changed = copy.deepcopy(reports)
    changed[Stage.BEHR]["rows"][0]["negative_training"] = "shuffled_y_unsafe_control"
    with pytest.raises(ValueError, match="unsafe"):
        compare_reports(changed, control)
    changed = copy.deepcopy(reports)
    changed[Stage.BEHR]["rows"][0]["heldout_pairs"] = 6
    with pytest.raises(ValueError, match="cohort"):
        compare_reports(changed, control)
    changed = copy.deepcopy(reports)
    changed[Stage.BEHR]["likelihood"][0]["pairs_examined"] = 2
    with pytest.raises(ValueError, match="pair counts"):
        compare_reports(changed, control)
    changed = copy.deepcopy(reports)
    changed[Stage.BEHR]["source_revision"] = "other code"
    with pytest.raises(ValueError, match="metadata"):
        compare_reports(changed, control)


def test_shell_runner_syntax_and_rejects_bad_mode_without_cuda():
    assert RUNNER.is_file()
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    failed = subprocess.run(["bash", str(RUNNER), "nonsense"],
                            capture_output=True, text=True)
    assert failed.returncode == 2
    text = RUNNER.read_text()
    assert text.count("run_stage ") == 3
    assert text.index("run_stage base ") < text.index("run_stage textworld_sft ") < text.index("run_stage behr ")
    assert "--device cuda --4bit" in text
    assert "EXPECTED_ROWS=96" in text
