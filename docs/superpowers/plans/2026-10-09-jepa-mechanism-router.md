# Blinded JEPA Mechanism Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. TDD RED→GREEN at each task; preserve scientific negatives.

**Goal:** Connect saved JEPA predictors, ScienceKernel, and the existing optional BOCPD backend with online evidence-grounded regime selection and candidate admission.

**Architecture:** Immutable checkpoint ports with a common anchored encoder, fitted 2D predictive distributions, a Bayesian router with open-set abstention, and an evaluator-only three-law sequential arena. Keep the real mechanism labels out of the router.

**Tech Stack:** Python 3.12+, NumPy, PyTorch, existing ECSA `ScienceKernel` and existing optional `bocd`; pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-jepa-mechanism-router-design.md`.

## Global constraints

No new dependencies; no privileged simulator variables in the router. Separate calibration, training, online, and promotion-validation cohorts. A route alarm is not a proof. Frozen shared JEPA latent frame. Fixed action/provenance types. Do not merge stacked PRs without approval.

## Review focus

1. All known experts fail for stochastic reasons — require repeated residual evidence and independent validation before promotion.
2. A newly trained candidate improves a known expert — do not declare a new mechanism.
3. Candidate validation reuses source training, calibration, or proposal evidence — reject overlapping provenance.
4. Proposal fires before actual mechanism switch — validate at alert time and reject unsupported admission.
5. Returning known or newly admitted mechanism has ambiguous posterior — abstain or request EIG experiment, never infer regime from simulator phase labels.

## Task 1: Checkpoint adapter and online probability routing

- [x] Write tests for typed port, shared encoder fingerprints, calibrated compatible-outcome update, known recurrence, action EIG and open-set abstention; verify RED on missing module.
- [x] Implement `src/ecsa/experimental/jepa_mechanism_router.py` using existing scientific kernel APIs.
- [x] Run tests and verify GREEN.

## Task 2: Novelty proposal and independent admission

- [x] Write tests for proposal without promotion, heldout superiority, old-mode improvement rejection, evidence overlap, rejected-proposal recovery; verify RED before the feature.
- [x] Implement provenance gates and `ScienceKernel.admit_theory()` transition; verify GREEN.

## Task 3: Blinded evaluator and benchmark runner

- [x] Write tests for third simulator law, unchanged common sensors, prefix-stable streams and six-epoch online arena.
- [x] Implement evaluator-only third law, causal-time-consistent intervention acquisition, and `scripts/benchmarks/run_jepa_router.py`.
- [x] Verify smoke and full mode on CPU without injecting simulator labels into router.

## Task 4: Scientific verification and handoff

- [x] Run affected pytest suite, compile source, record CPU seed cohorts and a separate heldout cohort with source checksums.
- [x] Add `neural-jepa-mechanism-router.yml` CI, docs and self-review.
- [ ] Verify remote GitHub CI on the stacked PR; obtain CUDA confirmation from user separately.
