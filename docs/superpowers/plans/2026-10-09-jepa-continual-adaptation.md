# Action-JEPA Continual Adaptation Implementation Plan

> **For agentic workers:** Use red-green tests for each task, then run the complete relevant suite. Native sequential execution.

**Goal:** Add a reproducible `additive -> gated -> additive` arena that tests adaptation, catastrophic forgetting, and checkpoint-based recall without oracle regime labels.

**Architecture:** Reuse action_jepa, blind intervention tests and calibrated detector. Add an evaluator-only recurrent world, predictor-only fine-tuning with/without replay, and an arena that records matched heldout evaluation, intervention cost and abstentions.

**Tech Stack:** Python >=3.12, numpy, torch, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-09-jepa-continual-adaptation-design.md`.

## Global Constraints
- The scientific optimizer never imports simulator Mechanism or change-point.
- Replay draws *only* from prechange training examples; heldout, calibration, confirmation never enter training.
- Predictor-only adaptation; frozen encoder/EMA target for comparable latent scoring.
- Reproducible action schedules, disjoint evidence IDs, no oracle-triggered repair.
- Truth labels may appear only in evaluator outputs.

## Review Focus
1. A false alarm must never trigger gated adaptation.
2. A monitor missing the change must report adaptation as skipped, not silently inject gated samples.
3. Replay cannot include any validation or heldout examples.
4. Returned additive transitions cannot be evaluated in a changed embedding basis.
5. Reject malformed data/negative budgets/invalid split boundaries.

## Task 1: Recurrence fixture and slicing
- [ ] Write tests for additive/gated/additive transitions, reproducible masked observations, invalid phases and no mechanism labels in datasets.
- [ ] Verify expected RED; implement `jepa_recurrence_world.py` and helpers; verify GREEN.

## Task 2: Frozen-coordinate predictor adaptation
- [ ] Write tests that adapting gated-only and replay uses identical steps, modifies predictor but not encoder, enforces training-only sample caps, exposes learning traces and tests no-update zero budget.
- [ ] Verify expected RED; implement `jepa_adaptation.py`; verify GREEN.

## Task 3: Blinded integrated scientist arena
- [ ] Write tests for full lifecycle, missing/early alarm, independent intervention banks, heldout scoring, memory and action accounting, same-model heldout and return recognition with no oracle labels in scientist/adaptation modules.
- [ ] Verify expected RED; implement `jepa_adaptation_arena.py`; verify GREEN.

## Task 4: CLI, documentation, CI
- [ ] Test CLI smoke JSON and invalid arguments; verify RED.
- [ ] Implement `scripts/benchmarks/run_jepa_adaptation.py`, experiment doc and test workflow; verify GREEN.
- [ ] Run focused tests, previous JEPA tests, full local available suite, and GPU invocation guard.
- [ ] Review diff for leakage; commit branch, open PR without auto-merge.
