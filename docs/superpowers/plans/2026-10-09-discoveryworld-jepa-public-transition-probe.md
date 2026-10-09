# DiscoveryWorld JEPA Public Transition Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement each task with tests first.

**Goal:** Run ECSA action-conditioned JEPA on public logged DiscoveryWorld transitions with honest independent-seed and ablation controls.

**Architecture:** Read the strict public Arena A observation/action logs, produce fixed-width hash sketches, adapt existing ActionJEPA to arbitrary action width, compare three JEPA arms with raw Ridge, and serialize a reproducible report.

**Tech Stack:** Python 3.12+, NumPy, PyTorch, pytest, pinned optional DiscoveryWorld for CI smoke.

**Spec:** `docs/superpowers/specs/2026-10-09-discoveryworld-jepa-public-transition-probe-design.md`

## Global Constraints

- Do not ingest or inspect scorecards or hidden world state while preparing features or training.
- Never treat cold/reuse replay of one seed as two independent seeds.
- No claim of real-time DiscoveryWorld policy improvement or causal counterfactual discovery in this offline probe.
- Preserve the original three-action JEPA default and existing test suite.

## Review Focus

- Log step alignment (`pre[t]`, action `t`, `post[t+1]`), phase duplicates: raise rather than shift labels.
- ScoringInfo/criticalQuestions leak in nested public payload: reject.
- Dynamic action width is consumed throughout JEPA predictor and training.
- Replay split labels remain independent by official seed: reject single-seed study.
- Cross-arm latent scale mismatch: report within-arm improvement, not absolute MSE comparisons.

## Task 1: Public DiscoveryWorld replay

**Files:** `src/ecsa/benchmarks/discoveryworld/jepa_replay.py`, `tests/test_discoveryworld_jepa_probe.py`

**Interfaces:** `ReplayConfig`, `encode_public_json`, `load_episode`, `load_splits`.

- [x] Write failing tests for deterministic public features, no oracle, log alignment and seed-level study.
- [x] Run RED: import failed for nonexistent `jepa_replay`.
- [x] Implement strict aligned public JSON replay and stable hash vectorizer.
- [x] Run GREEN for the new replay tests.

## Task 2: Reuse existing JEPA with dynamic action width

**Files:** `src/ecsa/experimental/action_jepa.py`, `tests/test_discoveryworld_jepa_probe.py`

**Interfaces:** `ActionJEPA(observation_dim, config, *, action_dim=None)` and `train_jepa(data,config,...)`.

- [x] Test default 3 and nondefault action widths, with wrong-width rejection.
- [x] RED initially failed because constructor ignored custom action width.
- [x] Add configurable width while retaining existing default.
- [x] Run GREEN and earlier synthetic tests.

## Task 3: Learning and matched controls

**Files:** `src/ecsa/benchmarks/discoveryworld/jepa_probe.py`, `tests/test_discoveryworld_jepa_probe.py`

**Interfaces:** `ProbeConfig`, `score_actions`, `run_probe`, `run_probe_from_logs`, module CLI.

- [x] Test JEPA/zero-action/shuffle/raw Ridge on disjoint seed partitions.
- [x] RED: nonexistent probe module import failed.
- [x] Implement matched replay diagnostic and JSON export.
- [x] Run GREEN for 9 probe tests and regression tests.

## Task 4: Real environment guard and user commands

**Files:** `tests/test_discoveryworld_jepa_real.py`, `docs/experiments/2026-10-09-discoveryworld-jepa-replay.md`, `.github/workflows/discoveryworld-jepa-replay.yml`

- [ ] Add a real pinned DiscoveryWorld public observation adapter test with no scorecard access.
- [ ] Document smoke vs heldout study commands, the negative controls and evaluation limits.
- [ ] Run offline full local suite; verify optional real integration in CI.
- [ ] Commit in an isolated feature branch based on `feat/jepa-mechanism-router-20261009`.
