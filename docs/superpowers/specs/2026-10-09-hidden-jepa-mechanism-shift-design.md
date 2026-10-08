# ECSA: hidden mechanism shift, prediction surprise and active verification

## Research question
Can one frozen, action-conditioned JEPA trained on the *additive* regime detect an unannounced switch to *gated* dynamics from observed transitions, propose competing explanations, choose an informative intervention and confirm the change with new evidence? Does it outperform comparably evaluated controls?

## Scope and isolation
This is a **synthetic scientific benchmark**, not evidence of general causal discovery. The simulator alone knows `Mechanism` and change-point. Observation/action-only APIs are passed to learner and scientist; neither gets a law label, hidden `level`, hidden `gate`, shift timestamp, or a privileged sensor decoder. The additive environment, switched environment, and null environment share a fixed sensor map with independent sequences and seeds. One JEPA is pretrained once on additive transitions; its weights are frozen during monitoring, experiment selection and verification. No post-change training or access to future interventions is permitted at detection time.

## Data and evidence protocol
1. Train one JEPA from additive `before,action,after` samples, plus a matched action-shuffled JEPA. Both get the same prechange observations, target data and step budget; no sample labels.
2. Fit observation-only raw and frozen-random-feature ridge next-state baselines using the same prechange training transitions and calibrate each detector using disjoint additive transitions. A symbolic observation-level baseline tests action-order dependence via matched resets without learned neural features; account for its intervention budget.
3. Run a blinded observation/action stream with a private (evaluator-only) switch after a fixed action count. Run an additional **no-switch** stream to measure false alarms. No detection algorithm may inspect the step clock or simulator attributes.
4. A calibrated rolling error monitor, using per-action prechange calibration and null bootstrap **maximum over the whole monitoring horizon**, raises alerts from transition residuals; report delay and false alarm rate, with no uncalibrated threshold tuning to the postchange traces.
5. Only if alerted, propose `H0` (no detectable action-order effect) versus `H1` (some action pair has nonzero order effect). Acquire a small exploratory pilot for all unordered pairs of available public actions, choose the pair with largest observed noise-adjusted effect, and confirm on a separate, independently acquired dataset using a one-sided paired permutation test with a fixed alpha. Do not interpret failure to reject H0 as proof H0.
6. The ground-truth mechanism and switch timestamp are used **only by the external evaluator**, after the agent returns its report. Keep pilot and confirm evidence IDs disjoint, and do not reuse postchange stream observations for calibration.

## Outputs and controls
Report reproducible seeds, model configuration, train/calibration/monitor counts, change-point only in evaluator-facing section, residual baseline, alarm step/delay, null false alarms, selected pair, pilot/confirmation evidence IDs, p-value, effect/noise magnitudes, confirmation decision, heldout action-choice, and limitations. Controls are matched shuffled-action JEPA, random-feature/ridge, raw-feature/ridge and symbolic observation-level commutator. No withheld-label leakage, no artificial weakness of baselines, and no results presented as general mechanism discovery.

## Risk controls
- Shift timing cannot be used by detector.
- No model input contains mechanism ID, hidden state or change-point.
- Unknown/invalid feature dimensions or nonfinite residuals raise errors.
- Calibration is isolated from monitoring and hypothesis confirmation.
- Action-pair selection does not use confirmation data; no post-hoc favorable pair swapping.
- False positives on no-switch stream are explicitly reported.
- Strong raw/symbolic results are a valid outcome, not an experiment failure.
