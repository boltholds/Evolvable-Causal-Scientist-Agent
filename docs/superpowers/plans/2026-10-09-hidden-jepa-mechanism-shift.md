# ECSA Hidden-Shift JEPA Implementation Plan

> For agentic workers: native RED→GREEN implementation with verification of every independent gate.

**Goal:** Compare learned action-conditioned latent dynamics to raw, random, shuffled-action and symbolic controls on a previously unannounced additive→gated switch, including blinded detection and separate active confirmation.

**Architecture:** Simulator-only scheduled switch; typed observational predictor and calibrated drift monitor; observation-only causal scientist performs exploratory selection and heldout permutation confirmation; arena orchestrates matched controls and reports law truth in evaluator-only section.

**Tech Stack:** Python 3.12+, NumPy, PyTorch (existing optional `neural-relations`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-hidden-jepa-mechanism-shift-design.md`

## Global Constraints
- No direct Mechanism import in scientist/detector modules.
- Same public observations/actions for all controls; fit/calibration/confirm disjoint.
- Use exact enum/dataclass contracts, not unstructured domain states.
- Both switched and stationary null arms required; no positive result from ground-truth checkpoint.

## Review Focus
- Private shift step only in simulator, never detector inputs.
- Monitor calibration cannot inspect switched samples or learn threshold after alarm.
- Identical evidence IDs in raw/JEPA confirmation, pilot IDs separate.
- Placebo/no-switch rejects no false discovery where possible; report errors.
- Shuffled action model may still discover interaction: do not equate detection with action-conditioned learning.

## Tasks
1. **World.** Write failure-first tests for scheduled switch retaining state and stable sensor surface; add evaluator-only `HiddenShiftWorld` and stream collection.
2. **Residual monitoring.** Tests for independent baseline residual scoring, action-conditioned ridge, rolling/bootstrap calibration and null false alarms. Implement typed `ResidualModel`, `CalibratedShiftMonitor`.
3. **Causal scientist.** Tests for paired null permutation symmetry, noncommutative detection, disjoint pilot/confirm and action-set only interface. Implement hypothesis selection and confirm.
4. **Arena & CLI.** Tests for complete report, identical evidence, frozen JEPA and no-switch negative, reproducibility and smoke CLI. Implement full + smoke runner with controls and documentation.
5. **Review and delivery.** Run offline tests, repeat deterministic seeds, inspect leakage/false alarms, publish branch and PR after passing verification; do not merge without confirmed green checks.
