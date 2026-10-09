# Shared Outcome Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Make JEPA, symbolic learners, KAN and simple statistical predictors comparable on the same fixed, public, heldout targets instead of representation-specific latent or hashed MSE.

**Architecture:** Fit a domain-neutral SharedOutcomeFrame from *training* PerceptualObservation/InteractionTransition objects only. Project every model prediction into this immutable role-relative target frame, with explicit masks and calibration. Score all backends using exactly the same heldout examples, observable action outcomes and train-only feature scaling; separately expose abstention, unknown features, and whether action contracts were independently interventionally confirmed.

**Tech Stack:** Python >=3.12, numpy/scikit-compatible linear algebra or existing numpy, existing ECSA world_model and ActionJEPA experimental port, pytest >=8. No required benchmark dependencies in core.

**Spec:** \`docs/superpowers/specs/2026-10-09-universal-causal-experiment-loop-design.md\`, sections 3, 6–10.

## Global Constraints

- Discover target feature vocabulary, numeric scaling, observed type and role names **only from training data**; validation/test must never fit or modify the frame.
- No benchmark-specific nouns, field names, action IDs, simulator internals or scorecard in \`src/ecsa/world_model/shared_evaluation.py\`.
- Distinguish observed missingness, unobserved effect and actual no-change; track unseen features explicitly.
- The common target frame is independent of any JEPA encoder, hashed-state representation or external LLM.
- All predictors must score the same heldout observations with identical target masks and denominators; failed/abstained predictions must be reported as coverage changes, never silently omitted.
- Observational predictions are not evidence of causal identification. Report confirmations from Slice 2 separately.
- Preserve legacy JEPA probe and synthetic arenas, and measure per-action and per-feature results.
- Run RED→GREEN for every test set and commit each reviewable task.

## Review Focus

1. New heldout-only feature: report \`unseen_feature_count\` without fitting/relabeling (Task 1: \`test_test_only_feature_does_not_extend_vocabulary\`).
2. Numeric field scales differ by orders of magnitude: use **train-only** normalization with explicit zero-variance handling (Task 1: \`test_numeric_scalers_frozen_and_constant_safe\`).
3. Missing next observation: mask coordinate without manufacturing unchanged outcome; preserve coverage (Task 1: \`test_missing_is_not_persistence\`).
4. A model abstains on the most difficult rows: do not let denominator shrink or inflate apparent accuracy (Task 3: \`test_abstention_does_not_change_common_denominator\`).
5. JEPA latent space differs from RawRidge feature space: reject direct latent-MSE comparison and score only projected common targets (Task 2: \`test_neural_and_linear_backend_share_target_fingerprint\`).

---

## Task 1: Immutable common public target frame

**Files:**
- Create: \`src/ecsa/world_model/shared_evaluation.py\`
- Test: \`tests/test_shared_outcome_frame.py\`

**Interfaces:**
- \`PublicOutcomeTransition(transition: InteractionTransition, before: PerceptualObservation, after: PerceptualObservation)\`.
- \`TargetKind(StrEnum)\`: \`NUMERIC_DELTA\`, \`BOOLEAN_CHANGE\`, \`CATEGORICAL_CHANGE\`, \`ACTION_SUCCESS\`.
- \`OutcomeCoordinate(key: str, role: str, kind: TargetKind, train_center: float, train_scale: float)\`.
- \`SharedOutcomeFrame.fit(train: tuple[PublicOutcomeTransition,...]) -> SharedOutcomeFrame\` (class method).
- \`SharedOutcomeFrame.project(item: PublicOutcomeTransition) -> SharedTarget\`.
- \`SharedTarget(values: tuple[float,...], observed_mask: tuple[bool,...], unseen_feature_count: int, coordinate_ids: tuple[str,...])\`.
- \`SharedOutcomeFrame.fingerprint: str\`: stable hash of training-derived coordinate definitions, not an episode/seed identity.

Projection uses global and role-bound entity features. Public entity IDs may align within a single transition, but do not enter target keys/values; unbound entities use a deterministic permutation-invariant multiset/count projection. If entity alignment is ambiguous, mask the target rather than infer false stability. Numerical deltas are normalized with train-only scale; categories become train-vocabulary change indicators, and action-success is observed iff the RawActionOutcome contains its public boolean.

- [ ] **Step 1 — RED:** Write \`test_training_only_frame_keeps_identical_fingerprint_on_heldout\`, \`test_test_only_feature_does_not_extend_vocabulary\`, \`test_numeric_scalers_frozen_and_constant_safe\`, \`test_missing_is_not_persistence\`, \`test_object_renaming_does_not_change_role_target\`, \`test_categorical_and_numeric_targets_have_distinct_kinds\`.
- [ ] **Step 2 — verify RED:** \`pytest -q tests/test_shared_outcome_frame.py\`; failure must prove the missing frame behavior.
- [ ] **Step 3 — GREEN:** Implement the complete projection and freezing protocol without importing JEPA or any benchmark adapter. Make frame dimensions and frozen masks identical across all backend arms.
- [ ] **Step 4 — verify GREEN:** \`pytest -q tests/test_shared_outcome_frame.py\`.
- [ ] **Step 5 — commit:** \`git add src/ecsa/world_model/shared_evaluation.py tests/test_shared_outcome_frame.py && git commit -m "feat: freeze canonical public outcomes on training data"\`.

## Task 2: Model-agnostic scoring adapter interface

**Files:**
- Create: \`src/ecsa/world_model/predictive_adapters.py\`
- Test: \`tests/test_shared_predictive_adapters.py\`

**Interfaces:**
- \`SharedPredictionPort(Protocol)\` has \`frame_fingerprint: str\` and \`predict_public(observation: PerceptualObservation, action: GroundAction) -> SharedForecast\`.
- \`SharedForecast(values: tuple[float,...], mask: tuple[bool,...], frame_fingerprint: str)\`.
- \`PersistencePort(frame: SharedOutcomeFrame)\`, predicting no numeric/categorical change.
- \`RawRidgePort.fit(frame: SharedOutcomeFrame, train: tuple[PublicOutcomeTransition,...], *, ridge_lambda: float=1.0)\`: predict common normalized public outcomes.
- \`LatentReadoutPort.fit(frame: SharedOutcomeFrame, train: tuple[PublicOutcomeTransition,...], latent_port: LatentPredictionPort, *, ridge_lambda: float=1.0)\`, where \`LatentPredictionPort.predict_latent(before: PerceptualObservation, action: GroundAction) -> tuple[float,...]\` is a separately injected predictor, e.g. JEPA, KAN or frozen symbolic embedding. The linear readout is fitted on train only; no encoding trick may leak the action into its target.

No model-specific loss enters frame construction. Verify \`frame_fingerprint\` is identical for all ports and refuse forecasts with mismatched sizes, frame identities, NaNs or out-of-range binary probabilities.

- [ ] **Step 1 — RED:** Add \`test_neural_and_linear_backend_share_target_fingerprint\`, \`test_latent_readout_uses_train_only\`, \`test_persistence_outputs_exact_common_no_change\`, \`test_raw_ridge_predicts_same_scale_as_latent_port\`, \`test_forecast_wrong_frame_or_invalid_probabilities_rejected\`.
- [ ] **Step 2 — verify RED:** \`pytest -q tests/test_shared_predictive_adapters.py\`.
- [ ] **Step 3 — GREEN:** Implement minimal adapters and immutable shared forecast contract, using fit-once train-only ridge for neural latent readout; don't import ActionJEPA directly in the domain-neutral core module.
- [ ] **Step 4 — verify GREEN:** \`pytest -q tests/test_shared_predictive_adapters.py tests/test_shared_outcome_frame.py\`.
- [ ] **Step 5 — commit:** \`git add src/ecsa/world_model/predictive_adapters.py tests/test_shared_predictive_adapters.py && git commit -m "feat: standardize predictor outputs into common outcome frame"\`.

## Task 3: Matched heldout and per-action scientific scoring

**Files:**
- Modify: \`src/ecsa/world_model/shared_evaluation.py\`
- Test: \`tests/test_shared_scoring.py\`

**Interfaces:**
- \`SharedEvaluation.score(truth: tuple[SharedTarget,...], forecasts: dict[str,tuple[SharedForecast,...]], *, actions: tuple[GroundAction,...], intervention_records: tuple[EffectClaim,...] = ()) -> dict[str,BackendScore]\`.
- \`BackendScore\` records: common \`row_count\`, \`scored_numeric_count\`, \`scored_binary_count\`, \`numeric_normalized_mse\`, \`binary_brier\`, \`binary_change_f1\`, \`action_success_brier\`, \`per_action\`, \`per_feature\`, \`coverage\`, \`abstentions\`, \`unseen_feature_count\`, \`interventional_confirmation_count\`.

Do not include optional decision telemetry in target coordinates. Sum errors on the **same observed target mask** for all ports; abstaining ports report incomplete coverage separately while their matched-sample error is clearly labeled noncomparable unless full coverage is obtained. Never count test-frame feature drift as feature discovery.

- [ ] **Step 1 — RED:** Add \`test_identical_rows_and_target_mask_for_all_backends\`, \`test_abstention_does_not_change_common_denominator\`, \`test_macro_action_metric_prevents_majority_type_dominance\`, \`test_no_action_success_field_means_not_scored\`, \`test_unseen_features_and_zero_variance_are_reported\`, \`test_interventional_and_observational_evidence_never_conflated\`.
- [ ] **Step 2 — verify RED:** \`pytest -q tests/test_shared_scoring.py\`.
- [ ] **Step 3 — GREEN:** Implement the scoring object with per-coordinate validation and well-defined masked metrics, including explicit \`None\` when comparable paired error is not estimable.
- [ ] **Step 4 — verify GREEN:** \`pytest -q tests/test_shared_scoring.py tests/test_shared_predictive_adapters.py\`.
- [ ] **Step 5 — commit:** \`git add src/ecsa/world_model/shared_evaluation.py tests/test_shared_scoring.py && git commit -m "feat: score models on matched public world outcomes"\`.

## Task 4: External benchmark adapter and two-domain test

**Files:**
- Create: \`src/ecsa/benchmarks/discoveryworld/shared_probe.py\` (benchmark *adapter*, never core dependency).
- Create: \`tests/test_shared_scoring_two_domains.py\`.
- Create: \`tests/test_discoveryworld_shared_probe.py\`.
- Create: \`docs/experiments/2026-10-09-common-target-jepa-evaluation.md\`.

**Interfaces:** \`run_shared_probe(source_root: Path, *, train_seeds: tuple[int,...], validation_seed: int, heldout_seed: int, output: Path, representation: str) -> dict\`. The external adapter constructs \`PublicOutcomeTransition\` from existing public Arena A logs and the existing public PerceptionFrontend, freezes one training frame and evaluates the existing JEPA and RawRidge/persistence ports against the same targets. For actual CUDA studies, train JEPA encoders in the experimental package; only project their forecasts into the frozen outcome frame here.

- [ ] **Step 1 — RED:** Add \`test_two_unrelated_raw_worlds_share_core_evaluation_contract\`, \`test_discoveryworld_adapter_has_no_oracle_input\`, \`test_hash_and_learned_interface_use_same_frame_fingerprint\`, \`test_seed4_not_used_for_vocabulary_or_readout_fit\`, \`test_baselines_receive_identical_action_and_sample_budget\`.
- [ ] **Step 2 — verify RED:** \`pytest -q tests/test_shared_scoring_two_domains.py tests/test_discoveryworld_shared_probe.py\`.
- [ ] **Step 3 — GREEN:** Implement the read-only adapter and two fake \`RawEnvironmentPort\` domains, then document the CLI/study settings. Do not change benchmark task policy or claim interactive task success based on offline forecasts.
- [ ] **Step 4 — verify GREEN:** \`pytest -q tests/test_shared_scoring_two_domains.py tests/test_discoveryworld_shared_probe.py\` and \`pytest -q\`; list all skips/failures and make a CI job exercise the real DiscoveryWorld optional dependency when present.
- [ ] **Step 5 — commit:** \`git add src/ecsa/benchmarks/discoveryworld/shared_probe.py tests/test_shared_scoring_two_domains.py tests/test_discoveryworld_shared_probe.py docs/experiments/2026-10-09-common-target-jepa-evaluation.md && git commit -m "test: compare JEPA and baselines in shared observable frame"\`.

## PR and acceptance gate

- Open **implementation PR C** stacked on PR B, no main merge without explicit review.
- Report the fixed common target fingerprint, vocabulary count, train-only scaling, masks, per-action sample distribution, precision of effect attribution and independent holdout values for **each** predictor.
- Only if this evaluation protocol passes may DiscoveryWorld be used to claim cross-representation improvement. Re-run on official seed 0–2 training, seed 3 validation, seed 4 heldout and another domain that shares no scenario semantics; run several model initialization seeds if results are stochastic.
- If JEPA fails to beat persistence/RawRidge, report failure and explore perception/progressive experiment selection, not model-size tuning or cherry-picked seed runs.
- A successful offline target prediction does not imply improvement in task completion. Interactive paired policy trials require a *separate* subsequent experiment and are not silently folded into this PR.
