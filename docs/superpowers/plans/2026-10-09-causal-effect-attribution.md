# Causal Effect Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Replace naive before/after action-effect assignment with evidence-based, competing explanations for background, state, action and interaction dynamics that remain falsifiable under partial observability.

**Architecture:** Project generic PerceptualObservation + GroundingUpdate into *comparable observed feature trials*, including no-change observations. Store disjoint passive and active evidence cohorts; compare outcomes under state-matched alternative actions, then request prospective information-gain tests through the existing ScienceKernel. Only independent, adequately controlled intervention records may raise the confidence status above observational support.

**Tech Stack:** Python >=3.12, current world_model dataclasses, ecsa.science.ScienceKernel, optional numpy for diagnostic summaries, pytest >=8. No extra required benchmark dependency.

**Spec:** \`docs/superpowers/specs/2026-10-09-universal-causal-experiment-loop-design.md\`, sections 3, 5, 7–10.

## Global Constraints

- No benchmark-specific field names, action names, world objects, or hidden/environment-oracle state in this module.
- Keep \`InteractionGrounder\` and \`WorldModelAcquisitionKernel\` authoritative; evidence attribution is a typed extension, not a parallel world model.
- Distinguish action *success*, observable *change*, *predictive* association, and *controlled intervention* evidence; none imply all the others.
- A changed clock/image is classified by cross-action evidence; never hard-code a blacklist.
- Observationally supported prediction is NOT causal proof. Report intervention-specific scope and assumptions, never \`causal_proven=True\`.
- Observation/action data must come from the public \`RawEnvironmentPort\` surface and \`PerceptionFrontend\`.
- Require an independent prospective confirmation cohort and disjoint IDs for an interventional promotion.
- Every task runs RED → GREEN with verified failures and passes, atomic commits and a full regression check.

## Review Focus

1. Feature absent from the post-observation: unknown/missing, not proof of no change (Task 1: \`test_missing_post_feature_is_not_no_change\`).
2. A monotonically changing feature under diverse actions: background or unresolved, not action-dependent (Task 2: \`test_cross_action_counter_prefers_background\`).
3. One action sampled only when a state flag is true: observational confounding blocks interventional promotion (Task 2: \`test_state_confounding_blocks_action_claim\`).
4. Resetless world and passive logs: remain observational/unresolved, no fabricated paired intervention (Task 3: \`test_offline_only_evidence_cannot_gain_interventional_status\`).
5. One cohort reused as proposal/training/confirmation or renamed transition: reject provenance collision and cross-split leakage (Task 3: \`test_confirmation_requires_disjoint_trial_and_episode_ids\`).

---

## Task 1: Generic observed-feature trial projection

**Files:**
- Create: \`src/ecsa/world_model/effect_attribution.py\`
- Test: \`tests/test_effect_attribution_projection.py\`

**Interfaces:**
- Consumes: existing \`InteractionTransition\`, \`GroundingUpdate\`, \`PerceptualObservation\`, \`GroundAction\`, \`ObservedFeature\`, \`FeatureDelta\`.
- Produces:
  - \`FeatureTrial(evidence_id: str, effect_key: str, action_schema_id: str, argument_role: str, context_signature: tuple[str,...], observed: bool, changed: bool | None, value_before: FrozenRawValue | None, value_after: FrozenRawValue | None)\`.
  - \`EffectTrialProjector.project(transition: InteractionTransition, grounding: GroundingUpdate, before: PerceptualObservation, after: PerceptualObservation) -> tuple[FeatureTrial,...]\`.

Only align objects through evidence-backed public source identities / argument-to-entity matches. Use \`arg[0]:<feature_id>\`, \`global:<feature_id>\` or an explicitly ambiguous role; never include raw UUID values in the canonical feature signature. Compare observed features including no-change cases; unknown matching produces \`changed=None\`, not negative evidence. Context signatures come from observed role-relative condition features and do not contain timestamp/observation ID strings as identity shortcuts.

- [ ] **Step 1 — RED:** Add \`test_argument_binding_is_role_relative_across_uuid_renaming\`, \`test_missing_post_feature_is_not_no_change\`, \`test_unchanged_observed_feature_produces_explicit_negative_trial\`, \`test_ambiguous_identity_does_not_fabricate_match\`, \`test_schema_name_has_no_domain_semantics\`.
- [ ] **Step 2 — verify RED:** \`pytest -q tests/test_effect_attribution_projection.py\`; expected failure is missing generic projection behavior.
- [ ] **Step 3 — GREEN:** Implement immutable validated trial types and the complete feature-pair projection with stable train-independent role normalization.
- [ ] **Step 4 — verify GREEN:** \`pytest -q tests/test_effect_attribution_projection.py tests/test_world_model_grounding.py\`; map legacy test filename if necessary and do not silently skip.
- [ ] **Step 5 — commit:** \`git add src/ecsa/world_model/effect_attribution.py tests/test_effect_attribution_projection.py && git commit -m "feat: project comparable role-relative feature trials"\`.

## Task 2: Competing explanations with state-matched controls

**Files:**
- Modify: \`src/ecsa/world_model/effect_attribution.py\`
- Test: \`tests/test_effect_attribution_hypotheses.py\`

**Interfaces:**
- \`EffectExplanation(StrEnum)\`: \`BACKGROUND\`, \`STATE_DEPENDENT\`, \`ACTION_DEPENDENT\`, \`INTERACTION\`, \`UNRESOLVED\`.
- \`EffectEvidenceStatus(StrEnum)\`: \`PROPOSED\`, \`OBSERVATIONALLY_SUPPORTED\`, \`INTERVENTIONALLY_SUPPORTED\`, \`CONTRADICTED\`, \`INSUFFICIENT\`.
- \`CohortRef(cohort_id: str, episode_id: str, protocol_id: str, collection_mode: CollectionMode, pre_registered_prediction_id: str | None, state_match_group: str | None, assigned_action: GroundAction | None)\`.
- \`EffectClaim(effect_key: str, action_schema_id: str, explanation: EffectExplanation, status: EffectEvidenceStatus, support_ids: tuple[str,...], contradiction_ids: tuple[str,...], assumptions: tuple[str,...], context_signatures: tuple[tuple[str,...],...])\`.
- \`EffectAttributionLedger.observe(trials: tuple[FeatureTrial,...], *, cohort: CohortRef) -> None\`.
- \`EffectAttributionLedger.hypotheses(effect_key: str, *, action_schema_id: str) -> tuple[EffectClaim,...]\`.

For passive data use bounded counters conditional on coarse observed context and action. If support is not shared across compared actions/context, abstain. Produce a small competing set of explanations rather than forcing an exclusive hard label. On state-matched cross-action data distinguish high action-specific delta probability from high across-all-actions change probability. Report uncertainty, propensity overlap limits, and absence of controls; use no source-specific field filters. Retain explicitly separate observational and controlled evidence.

- [ ] **Step 1 — RED:** Add \`test_cross_action_counter_prefers_background\`, \`test_state_confounding_blocks_action_claim\`, \`test_matched_actions_support_action_dependent_proposal\`, \`test_action_state_interaction_distinguished\`, \`test_no_overlap_yields_unresolved\`, \`test_finite_memory_and_duplicate_trials_rejected\`. Assert no status above \`OBSERVATIONALLY_SUPPORTED\` from passive transitions.
- [ ] **Step 2 — verify RED:** \`pytest -q tests/test_effect_attribution_hypotheses.py\`.
- [ ] **Step 3 — GREEN:** Implement the minimal Beta-Bernoulli evidence tables indexed by effect/action/context with smoothing and explicit competing descriptive hypotheses; no causal hard claim and no automatic world-model admission.
- [ ] **Step 4 — verify GREEN:** \`pytest -q tests/test_effect_attribution_hypotheses.py tests/test_effect_attribution_projection.py\`.
- [ ] **Step 5 — commit:** \`git add src/ecsa/world_model/effect_attribution.py tests/test_effect_attribution_hypotheses.py && git commit -m "feat: infer competing background and action-effect hypotheses"\`.

## Task 3: Prospective discrimination and independent intervention promotion

**Files:**
- Modify: \`src/ecsa/world_model/effect_attribution.py\`
- Create: \`src/ecsa/world_model/attribution_experiments.py\`
- Test: \`tests/test_effect_attribution_interventions.py\`

**Interfaces:**
- \`AttributionExperimentSelector.select(claims: tuple[EffectClaim,...], candidate_actions: tuple[GroundAction,...], *, context_signature: tuple[str,...]) -> AttributionExperiment\`.
- \`AttributionExperiment(action: GroundAction, information_gain_bits: float, hypothesis_ids: tuple[str,...], pre_registered_predictions: tuple[PredictiveDistribution,...])\`.
- \`EffectAttributionLedger.confirm_intervention(effect_key: str, *, candidate: GroundAction, evidence: tuple[FeatureTrial,...], cohort: CohortRef) -> EffectClaim\`.

Use \`ScienceKernel.score_experiment\` / \`select_experiment\` on normalized discrete outcome distributions ("changed" / "unchanged" / "unobserved"). The intervention collector is an externally provided environment port, not a hidden simulator reset imported into the core. To mark \`INTERVENTIONALLY_SUPPORTED\`, require prespecified alternatives, disjoint proposal vs confirmation transition/episode IDs, matched initial public contexts, executed control and treatment arms with known assignment, and a prospective effect contrast on fresh evidence. Record context-limited result and do not claim identification when randomization/control cannot be established.

- [ ] **Step 1 — RED:** Add \`test_eig_selects_discriminating_public_action\`, \`test_offline_only_evidence_cannot_gain_interventional_status\`, \`test_confirmation_requires_disjoint_trial_and_episode_ids\`, \`test_matched_independent_intervention_promotes_only_tested_scope\`, \`test_failed_matched_intervention_contradicts_claim\`, \`test_nonresettable_world_abstains_instead_of_simulating_controls\`.
- [ ] **Step 2 — verify RED:** \`pytest -q tests/test_effect_attribution_interventions.py\`.
- [ ] **Step 3 — GREEN:** Implement typed experiment rankings and confirmation gates, and record exact action counts for treatment/control acquisition. Do not accept a bare \`controlled=True\` flag as proof: validate the cohort metadata and paired action evidence.
- [ ] **Step 4 — verify GREEN:** \`pytest -q tests/test_effect_attribution_interventions.py tests/test_effect_attribution_hypotheses.py\`.
- [ ] **Step 5 — commit:** \`git add src/ecsa/world_model/effect_attribution.py src/ecsa/world_model/attribution_experiments.py tests/test_effect_attribution_interventions.py && git commit -m "feat: qualify causal hypotheses with independent interventions"\`.

## Task 4: Integrate attribution with the established world-model kernel

**Files:**
- Modify: \`src/ecsa/world_model/kernel.py\` (\`WorldModelUpdate\`, \`observe_grounding\` compatible optional hook).
- Modify: \`src/ecsa/autonomy/scientist.py\` (optional \`effect_attribution: EffectAttributionLedger | None\`; record grounded trials after observing the action).
- Test: \`tests/test_scientist_effect_attribution.py\`

**Interfaces:** Attribution candidates attach to the same existing evidence ledger and \`WorldModelUpdate\` via a non-authoritative summary \`attribution_claim_ids: tuple[str,...]\`. No alternative contract truth store and no implicit \`HypothesisStatus.ADMITTED\` on observational evidence. \`AutonomousScientist\` routes prospective experiments via existing \`ContractExperimentCoordinator\` only if the request can be grounded from public \`RawActionSchema\`.

- [ ] **Step 1 — RED:** Add \`test_only_real_observed_transitions_reach_attribution\`, \`test_observational_effect_does_not_admit_causal_contract\`, \`test_claim_evidence_linked_to_original_grounding\`, \`test_default_kernel_behavior_unchanged_without_attribution\`, \`test_action_budget_counts_controls\`.
- [ ] **Step 2 — verify RED:** \`pytest -q tests/test_scientist_effect_attribution.py\`.
- [ ] **Step 3 — GREEN:** Wire optional attribution acquisition into existing grounding and scientist update without changing base environment behavior; preserve immutable contract versions and learner failure reports.
- [ ] **Step 4 — verify GREEN:** \`pytest -q tests/test_scientist_effect_attribution.py tests/test_effect_attribution_interventions.py\` and \`pytest -q\`; document all skips and failures.
- [ ] **Step 5 — commit:** \`git add src/ecsa/world_model/kernel.py src/ecsa/autonomy/scientist.py tests/test_scientist_effect_attribution.py && git commit -m "feat: connect causal attribution to world-model evidence"\`.

## PR and review gate

- Open **implementation PR B** on PR A's tested head; keep design PR/main intact.
- Require a source-blind, multi-action nuisance fixture, an explicit state-confounding fixture, and a controlled intervention fixture with disjoint cohorts.
- Test with two unrelated RawEnvironmentPort fixtures. No benchmark field or action-name hardcode.
- Report separately observational predictive accuracy, interventional status, matched cohort counts, active-action cost, and unsupported/abstaining claims.
- An interventionally supported claim is valid *only* for its tested context; do not rebrand as a universal causal law.
