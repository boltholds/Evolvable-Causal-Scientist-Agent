# ECSA: hidden mechanism shift and active intervention verification

## Research question
Does a **single action-conditioned JEPA** trained only on the additive regime detect unannounced changes of the action-effect mechanism and help an independent observer identify the changed causal interaction? Compare this to raw observations with a matched analytical predictor, frozen random projections with an analytical predictor, an action-shuffled JEPA, and two non-neural observation controls.

## Reproduction

```bash
python -m pip install -e '.[test,neural-relations]'
PYTHONPATH=src python scripts/benchmarks/run_jepa_shift.py \
  --mode smoke --seed 1 --device cpu --output results/jepa_shift_smoke.json
PYTHONPATH=src python scripts/benchmarks/run_jepa_shift.py \
  --mode full --seed 8 --device cuda --output results/jepa_shift_seed8.json
python -m pytest -q tests/test_jepa_shift.py tests/test_action_jepa.py
```

The local CPU reproduction tests the research protocol without requiring CUDA. `full` uses 700 training steps for the correct JEPA and its shuffled-action control, 2048 prechange additive training transitions, 1024 independent additive calibration transitions, 256 additive heldout transitions, and 160 chronological one-step monitoring trials. The hidden change takes effect **after 80 monitoring actions**. Every algorithm sees the same opaque 24-dimensional observations and public action identities; no model sees the `Mechanism`, hidden gate, hidden level, or shift time.

For reproducibility, monitor actions are planned independently of outcomes and matched across shifted and no-change streams. Each observation trial resets the simulator to an opaque seed. This is not a continuous trajectory of physical motion. A resettable simulator clone is allowed for paired interventions.

## Algorithm

1. Train JEPA and action-shuffled JEPA only on additive `observation, action, next observation` transitions, without mechanism supervision. Freeze both models throughout monitoring.
2. Fit raw- and random-feature **action-interaction ridge** predictors on the same additive data. Include an observation-level finite-difference action-template control.
3. Calibrate each predictor's transition errors using independent additive data. Normalize errors separately by observable action identity and bootstrap the maximum rolling mean **over the entire predeclared monitoring horizon**, not just one window (window 16 in full; alpha=.01). The procedure approximates null false-alarm control under conditionally exchangeable residuals; it is not distribution-free.
4. Investigate the **first chronological threshold crossing**. Even if it occurred before the true shift, evaluate interventions in a simulator clone of the mechanism at that **alert epoch**, not at the end of the trace. Both selection and confirmation see only public observations.
5. Propose two competing hypotheses: `order_independence` and `order_dependence`. Use 16 pilot trials for all three unordered public-action pairs and choose the pair with the largest observed noise-adjusted effect. Evaluate this **single selected pair** with 64 independent confirmation trials using a one-sided paired sign-flip permutation test (4095 draws, alpha=.01). Under a commutative mechanism, forward/backward and forward/repeated observations are exchangeable under identical trial resets.
6. Evaluate a **no-change placebo** with the same monitoring action sequence and independent paired confirmatory intervention evidence if it raises an alarm. Track false alarms separately from false *causal discoveries*.
7. A stronger **periodic symbolic baseline** uses no neural model: every 16 monitored steps it actively tests all available unordered pairs on raw observation vectors, with an independent confirmatory test and Bonferroni correction across all scheduled polls. It incurs a large additional *intervention cost*, which must be reported. The simpler finite-difference template is retained as `delta_template`.

## Interpretation and limitations

- Grading can distinguish alarms before and after the hidden switch; the scientist never knows the boundary.
- All pilot and confirmation interventions are disjoint; controls that alert in the same regime see the same acquired experimental evidence.
- Detection is not by itself a causal discovery. Only confirmed action **order dependence** of the selected pair is claimed; this does not identify a unique structural causal model.
- Absence of significant evidence is **not proof** that the null mechanism is correct.
- The benchmark permits replication of arbitrary initial simulator trials and a frozen-in-time experimental laboratory, which cannot generally be assumed in physical plants.
- The first-alarm scientist is not yet re-armed after negative confirmation; a false early alarm can prevent detection of a genuine later switch.
- Raw/ridge can beat JEPA; outcome and intervention budget comparisons, not a predetermined neural winner, determine research value.
- Fixed sensor geometry across training and test is a serious limitation; no cross-environment or real-world transfer claim.

## CPU reference outcomes — 13 fixed seeds

Offline full-size benchmarks were executed using the same implementation on CPU. Seed 0–8 were exploratory development runs; the code was frozen (SHA256 recorded in `2026-10-09-jepa-hidden-shift-results.json`) **before** seeds 9–12, which were evaluated as fresh confirmatory runs without retuning. These are small synthetic replication counts, not statistical evidence of superiority beyond this fixture.

| Detector / policy | Post-change confirmed / 13 | Early alarm | Null-world alarm | Median delay when detected | Median **additional** intervention actions |
|---|---:|---:|---:|---:|---:|
| Action-conditioned JEPA | 10 | 1 | 1 | 8.5 | 672 |
| Action-shuffled JEPA | 0 | 0 | 0 | — | 0 |
| Frozen random encoder + ridge | 6 | 0 | 0 | 26 | 0 (672 for alerted runs) |
| Raw observations + action-conditioned ridge | **13** | 0 | 0 | 19 | 672 |
| Finite-difference action template | 0 | 0 | 0 | — | 0 |
| Proactive symbolic action-order permutation | **13** | 0 | 0 | 16 | **4032** |

Delays are measured after the ground-truth switch **only among detected runs**, not assigned to failures. Intervention steps exclude the 160 monitoring observations and the 2048 baseline training transitions. The symbolic baseline does not train on the 2048 observations but tests all three public action pairs periodically; its results must not be compared solely on detection percentage or delay without accounting for intervention cost. Post-hoc future information is not used when deciding an early alarm: the seed-8 early JEPA false alarm is correctly rejected by *prechange* confirmatory experiments. The no-change world also raised one JEPA alarm but no confirmatory causal discovery.

On the four new heldout seeds 9–12, JEPA detected/confirmed 3/4; raw/ridge and proactive symbolic each 4/4. The controlled JEPA model shows earlier alerts on successful runs, but **raw/ridge was more reliable** in this setup. The strongest defensible statement is that the JEPA contains dynamics predictive enough to support meaningful shift monitoring and intervention tests, not that it is the best or necessary representation learner.

Run-level evidence, false alarms, selected pairs and action-choice accuracies are retained in the machine-readable JSON summary. This is a **preliminary synthetic bench**, not a proof of scientific autonomy. Important next tests: switch away from fixed sensor geometry, compare representation-only controls under equal intervention budgets, pre-register independent seeds and new mechanisms, implement monitor re-arm and posterior updates after a refuted hypothesis, and test adaptation/retention after a confirmed shift.
