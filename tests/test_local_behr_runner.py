"""No GPU required: local BehR comparison runner source-level contract."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "benchmarks" / "run_behr_vs_qwen25_7b.sh"


def test_runner_shell_syntax():
    assert RUNNER.is_file(), "Local runner missing from repository"
    subprocess.run(["bash", "-n", str(RUNNER)], check=True, capture_output=True, text=True)


def test_runner_pins_both_checkpoint_revisions_and_only_4bit_cuda():
    content = RUNNER.read_text()
    assert 'BEHR_REV="6a4326a60540cc33ffa42ee7a16fc2bad8f6f613"' in content
    assert 'BASE_REV="d149729398750b98c0af14eb82c78cfe92750796"' in content
    assert content.count("run_one ")==2, "Exactly two sequential model calls expected"
    assert "--device cuda --4bit" in content
    assert 'run_one "behr"' in content
    assert 'run_one "qwen_base"' in content
    assert "SHUFFLED_Y" not in content  # runner uses explicit typed CLI values


def test_runner_expected_row_counts_and_same_kan_observations():
    text = RUNNER.read_text()
    assert "EXPECTED_ROWS=32" in text
    assert "EXPECTED_ROWS=192" in text
    for arg in (
        "--laws precision operator", "--pool sentence_full token_mean",
        "--projection random train_only_pca",
        "--pairing witnessed shuffled_y_unsafe_control",
        "--replays 3",
    ):
        assert arg in text, arg
    assert "assert behr[\"config\"]==base[\"config\"]" in text
    assert "assert set(a)==set(b)" in text
    assert "kan_parameters" in text
    assert "no_kan_to_encoder_gradient" in text


def test_runner_rejects_unknown_mode_without_inference():
    outcome = subprocess.run(
        ["bash", str(RUNNER), "invalid"],
        check=False, capture_output=True, text=True,
    )
    assert outcome.returncode==2
    assert "Usage:" in outcome.stderr
