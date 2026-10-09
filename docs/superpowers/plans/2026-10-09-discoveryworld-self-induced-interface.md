# DiscoveryWorld Self-Induced Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Tasks follow RED→GREEN with unchanged holdout.

**Goal:** Learn and revise public object/action contracts without semantic hand labels and benchmark their representations with honest shared-target metrics.

**Architecture:** An experimental `InterfaceLearner` observes public Arena A logs and proposes evidence-backed action roles, effects and preconditions. `LearnedInterfaceEncoder` transforms public observations and actions into separate, train-only learned feature spaces, consumed by the existing Action-JEPA replay probe. The policy and authoritative `WorldModelAcquisitionKernel` are not changed.

**Tech Stack:** Python 3.12+, numpy, optional torch, pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-discoveryworld-self-induced-interface-design.md`

## Global constraints

- No hidden scorecard/scoring information in model training, calibration, or source observation/feature code.
- Official seed 4 reserved as test, seed 3 validation, seeds 0–2 train. No seed fitting after split.
- Offline support never becomes a causal admission automatically.
- The target encoder sees `o[t+1]` and does not receive action tokens.
- Keep the previous hashing probe and synthetic JEPA experiments operational.

## Review focus

- Duplicate public UUID aliases: merge compatible partial records; reject conflicting data.
- Duplicate action/observation IDs: reject rather than use ambiguous training rows.
- No observed state change: report unknown relative improvement, not +100%.
- Action candidates with unknown applicability: label action-contrast as observational, never `do(a)`.
- Schema drift: archive prior immutable proposed contract and require revalidation.

## Task 1 — Induce evidence-based action contracts

**Create:** `src/ecsa/benchmarks/discoveryworld/interface_induction.py`.
**Test:** `tests/test_discoveryworld_interface_induction.py`.

- [x] RED: unimplemented learner fails the unknown-schema and object-binding tests.
- [x] GREEN: emit type-safe action hypotheses and effect/precondition candidates using public evidence IDs.
- [x] Check outcome unknown is not fabricated into success.

## Task 2 — Validate and revise contracts

- [x] RED: validation required heldout independence/statuses.
- [x] GREEN: status `SUPPORTED`, `REJECTED`, or `INSUFFICIENT`; no causal auto-admission.
- [x] RED/GREEN: public field schema changes trigger a new proposal while immutable old snapshot remains available.

## Task 3 — Feed learned interface into JEPA

- [x] RED: learned interface split missing.
- [x] GREEN: fit train-seed vocabulary; typed action arguments become learned role-relative vectors.
- [x] RED/GREEN: state and target representations independent of executed action.

## Task 4 — Repair observational controls and shared-feature tests

**Modify:** `src/ecsa/benchmarks/discoveryworld/jepa_probe.py`.
**Test:** `tests/test_discoveryworld_probe_v2.py`.

- [x] RED/GREEN: all logged alternative action types, type macro metrics, applicability caveat.
- [x] RED/GREEN: frozen JEPA latent to common observable next-state deltas; compare to persistence and Ridge.
- [x] RED/GREEN: zero-change targets yield null improvement and rank warning is explicit.

## Task 5 — CI and repeatable study

- [x] Run entire available local test scope and report skipped optional tests (84 passed, one optional real-DiscoveryWorld test skipped).
- [x] Run both representations against generated three-seed public-log fixtures (synthetic compatibility fixtures, not benchmark scores).
- [ ] Publish a new stacked branch and CI workflow, leaving `main` intact.
- [ ] Run official DiscoveryWorld seeds 0–4 externally before claiming transfer.
