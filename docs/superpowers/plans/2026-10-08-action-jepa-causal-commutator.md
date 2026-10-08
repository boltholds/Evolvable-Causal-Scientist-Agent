# Action-conditioned JEPA and causal commutator implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task by task. Tests are written before implementation.

**Goal:** Train an action-conditioned latent world model and check whether an independent scientist can discover an intervention-order causal dependency without privileged simulator state or LLM features.

**Architecture:** Simulator-only truth → typed observation transition store → trainable online encoder + EMA-target predictor → independent commutator scientist → grader and null controls. Keep scientific truth out of learner/scientist modules.

**Tech Stack:** Python 3.12, NumPy, PyTorch 2.x, pytest, reproducible JSON CLI.

**Spec:** `docs/superpowers/specs/2026-10-08-action-jepa-causal-commutator-design.md`

## Global constraints

- Neither model nor scientist may read `Mechanism` or simulator hidden variables.
- Learned encoder is trained from scratch; do not load LLM hidden states or semantic targets.
- Test with repeated, paired interventions; monitor collapse and chance-level action scores.
- Baselines must share the same samples and evaluator statistic.
- All claims provisional outside the toy simulator; no unverified claim of success.

## Review focus

- Identical action pair must be rejected, not falsely diagnosed as commuting.
- Collapsed encoder must return abstention, not a negative mechanism finding.
- Sensor nuisance must be sampled independently for the repeated-sequence null.
- A shuffled-action control must not accidentally use original action tokens at training time.
- Heldout and train transitions must have separate trial-seed identities with common geometry.

### Task 1: Simulator / transition port

**Files:** `src/ecsa/experimental/jepa_world.py`; `tests/test_action_jepa.py`.

**Interfaces:** `OpaqueSwitchWorld.reset(trial_seed=...)`, `step(Action)`, and `collect_transitions` produce a `TransitionDataset` with exactly `before,actions,after`.

- [x] Write tests for deterministic paired resets, distinct mechanisms, finite shapes and unlabeled transitions.
- [x] Run tests; verify import fails before implementation.
- [x] Implement typed simulator and deterministic trial-seed sampling.
- [x] Run test module; confirm green.

### Task 2: Learned dynamics and controls

**Files:** `src/ecsa/experimental/action_jepa.py`; `tests/test_action_jepa.py`.

**Interfaces:** `train_jepa(data, config, device, shuffle_actions)` returns frozen-for-eval model/report; `evaluate_prediction` returns MSE, action-choice score, latent spread and rank.

- [x] Write failing contracts for shape, no target grads, noncollapse, heldout diagnostic and action shuffling.
- [x] Implement EMA teacher, online predictor, variance/covariance controls and training.
- [x] Verify finite output and green tests.

### Task 3: Independent scientist and graded arena

**Files:** `src/ecsa/experimental/jepa_causal_scientist.py`; `src/ecsa/experimental/jepa_commutator_arena.py`; `scripts/benchmarks/run_action_jepa.py`; `tests/test_action_jepa.py`.

**Interfaces:** `collect_commutator_observations` produces matched evidence; `score_commutator` uses only encoder and evidence; `select_intervention_pair` proposes a pair; `run_arena` compares baselines on identical evidence.

- [x] Write tests for intervention order, null condition, abstention, controls and CLI report.
- [x] Implement independent scorer with typed ports and evidence IDs.
- [x] Verify smoke runs and report schema.

### Task 4: Scientific documentation and CI

**Files:** `docs/experiments/2026-10-08-action-jepa-commutator-arena.md`; `.github/workflows/neural-action-jepa.yml`.

- [x] Run CPU 8-seed study (0–3 exploratory; 4–7 confirmation); preserve raw reports.
- [x] Verify all targeted tests and no local syntax errors.
- [ ] Commit and run remote CI without changing unrelated ECSA modules.
- [ ] Review scientific overclaims and promote via PR or documented branch state.
