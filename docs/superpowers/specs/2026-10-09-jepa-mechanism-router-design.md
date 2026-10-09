# Blinded JEPA Mechanism Router — Design

Date: 2026-10-09. Experimental ECSA module built on the preceding hidden-change and continual-adaptation branches.

## Intent and boundaries

One observer should reuse frozen predictive checkpoints for familiar transition laws, abstain when evidence is ambiguous, select a discriminating public action using expected information gain, propose a *new predictive mechanism* if **all** saved models repeatedly fail, and admit it only after an independent, heldout test. The learner must never see the simulator's mechanism, phase index, changepoint, or privileged level/gate variables.

## Design choices

- `JepaPredictivePort` snapshots existing `ActionJEPA` checkpoints. Their encoder fingerprints must be identical: all experts predict in one frozen latent coordinate system.
- `MechanismRouter` models a shared measurable event: discretized next-state projections of the **same** observed latent target. One projection and quantization scheme is fitted on the initial stored checkpoint's separate calibration data, before processing the online stream. Each expert provides a diagonal-Gaussian histogram over the same 36 outcomes. `ScienceKernel.update()` updates `TheoryPosterior`; a small symmetric hazard allows regime changes instead of irreversible posterior collapse.
- Individual expert residuals are calibrated on heldout examples from that expert's own previously established epoch. An open-set proposal requires all experts to fall into their calibrated low-tail on at least 3 of the last 18 observations. These empirical tail checks are heuristic repeated tests, not anytime-valid p-values.
- BOCPD via the existing `bocd` optional extra can surface an auxiliary changepoint diagnostic. It requests disambiguation but does not justify admission. No new package dependency is required.
- `select_experiment()` reuses `ScienceKernel.score_experiment()`. The evaluator subsequently gathers new data **at the alarm epoch**, not in a later world. Candidate training, candidate calibration, and independent validation are disjoint. `ScienceKernel.admit_theory()` is called only after the candidate improves heldout predictive MSE by the prespecified ratio and is not merely a better predictor for an already known calibration regime.
- An unsuccessful independent test withdraws the pending proposal and returns control to known-hypothesis inference.

## Arena

Single fixed geometry, opaque noisy observation vectors, actions `pulse`, `flip`, `wait`, six epochs: `Additive → Gated → Additive → Reverse-Gated → Additive → Reverse-Gated`. The third law acts on the same sensors and actions but changes pulse sign when the internal gate is active. Labels and transition boundaries exist only in the evaluator. A base JEPA and a separately adapted gated expert form the initial bank. No third expert exists until the online evidence causes a proposal. Fresh transitions train its candidate; disjoint transitions validate it. The sixth epoch tests reuse without retraining.

## Key tests and metrics

- Strict unknown/known identity and provenance validation; no duplicate checkpoint or evidence IDs; identical shared latent coordinates.
- No future data, no outcome leakage from simulator, independent admission evidence.
- Explicit negative test: a better version of an **existing** model must not be called a new causal mechanism.
- Diagnostic and performance outcomes: known selection count by epoch, first novel proposal delay, rejected/accepted proposals, false duplicate novelty proposals, active training/calibration/confirmation actions, and recurrence selections.
- CPU smoke, CPU full and user-run CUDA, seed-matched repetitions. Do not count many windows as independent replicates; holdout seeds, not late-window observations, are experimental replicates.

## Explicit limits

This is **predictive mechanism routing**, not identification of a unique causal graph. Finite-bin Gaussian likelihoods are approximate and not proven calibrated probabilities. Experts are prepared with mode-specific example cohorts initially; autonomous unbounded growth is not solved. The experiment uses simulator resets at alarm time to acquire independent evidence; arbitrary physical systems cannot necessarily do that. A third mechanism proposal does not establish a full causal explanation. A strong non-neural raw-observation baseline is not yet included in this new arena and must be added before any comparative superiority claim.
