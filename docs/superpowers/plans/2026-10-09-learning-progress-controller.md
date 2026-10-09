# Learning Progress Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Prevent the ECSA scientist from wasting interaction budgets on epistemically sterile action cycles while preserving valid repeated trials and domain independence.

**Architecture:** Add bounded progress and cycle evidence to the existing world-model layer. Give the current ContractExperimentCoordinator a strictly opt-in progress-aware candidate scoring hook; have the current AutonomousScientist record action outcomes and independently assessed hypothesis revisions. Retain the original ScienceKernel and all other default behavior as an ablation.

**Tech Stack:** Python >=3.12, dataclasses, existing ecsa.world_model + ecsa.science, pytest >=8. No new required dependency.

**Spec:** \`docs/superpowers/specs/2026-10-09-universal-causal-experiment-loop-design.md\`, sections 1–4, 7–10.

## Global Constraints

- This is core code only: no import of DiscoveryWorld, benchmark action names, or benchmark data.
- RawEnvironmentPort, PerceptionFrontend, InteractionGrounder, WorldModelAcquisitionKernel and ScienceKernel remain authoritative.
- No oracle/hidden labels; visible success is not the same thing as scientific information.
- Previously observed time or image changes are not assigned scientific meaning by hard-coded key names.
- Preserve existing policy with explicit \`use_progress_scoring=False\` ablation.
- Every task follows RED → verify failure → GREEN → verify pass → independently check diff → commit.
- Never claim generalization from one task or seed; deterministic two-domain acceptance is necessary but not sufficient.

## Review Focus

1. One legal action, always uninformative: still return that action or a typed, explicit exhausted-candidates result instead of throwing unexpectedly (Task 2: \`test_only_feasible_action_never_disappears\`).
2. Repeated experiment with independent positive evidence: do not suppress required replication (Task 2: \`test_independent_replications_retain_priority\`).
3. Unique state IDs or advancing counters with no validated new knowledge: do not reset stagnation (Task 1: \`test_unique_observation_identifiers_not_credit\`).
4. Repeated evidence IDs: reject duplicate history rather than inflate value (Task 1: \`test_duplicate_transition_provenance_rejected\`).
5. No finite measured predictive improvement yet: yield provisional uncertainty/abstention, not invented benefit (Task 1: \`test_unknown_progress_is_not_improvement\`).

---

## Task 1: Bounded, typed learning-progress evidence

**Files:**
- Create: \`src/ecsa/world_model/learning_progress.py\`
- Test: \`tests/test_learning_progress.py\`

**Interfaces:**
- Consumes: \`GroundAction\`, \`PerceptualObservation\`, immutable provenance strings and independently evaluated hypothesis updates.
- Produces: \`ProgressEvidence\`, \`ProgressAssessment\`, \`CycleEvidence\`, \`LearningProgressLedger.observe\`, \`LearningProgressLedger.assess\`, \`LearningProgressLedger.detect_cycle\`.

Contract to implement:

\`ProgressEvidence(transition_id: str, action: GroundAction, context_signature: tuple[str, ...], before_signature: tuple[str, ...], after_signature: tuple[str, ...], predictive_gain: float | None, uncertainty_reduction: float, confirmed_hypothesis_ids: tuple[str, ...], contradicted_hypothesis_ids: tuple[str, ...], action_cost: float, independent_trial_group: str | None)\`.

\`LearningProgressLedger(max_recent: int = 256, stagnation_horizon: int = 8)\`: immutable snapshot of recorded evidence for callers, bounded retention, cumulative counters and a duplicate-ID guard. \`assess(action: GroundAction, *, context_signature: tuple[str, ...]) -> ProgressAssessment\` returns a finite estimated benefit, diminishing-returns penalty, independent-replication credit and uncertainty; \`detect_cycle() -> CycleEvidence\` reports an evidence-scoped candidate cycle, not equality of hidden world state.

- [ ] **Step 1 — RED:** Write \`test_duplicate_transition_provenance_rejected\`, \`test_unique_observation_identifiers_not_credit\`, \`test_unknown_progress_is_not_improvement\`, \`test_progress_memory_is_bounded\`, \`test_cycle_detection_uses_context_not_private_ids\`. Make explicit assertions that \`predictive_gain=None\` is **unknown**, and changes in unique observation IDs alone cannot count as learning.
- [ ] **Step 2 — verify RED:** \`pytest -q tests/test_learning_progress.py\`. Expect only missing-feature/import or stated failing assertions, not typos or environmental errors.
- [ ] **Step 3 — GREEN:** Implement the typed ledger, finite input validators, per-schema/context decayed gain with a global fallback for changing partial observations, saturation of stagnation penalty, and clear separation of independent corroboration vs raw transition count. Do not infer progress from raw feature-delta count.
- [ ] **Step 4 — verify GREEN:** \`pytest -q tests/test_learning_progress.py\`. Expect all new tests green.
- [ ] **Step 5 — commit:** \`git add src/ecsa/world_model/learning_progress.py tests/test_learning_progress.py && git commit -m "feat: track universal learning progress and stagnation"\`.

## Task 2: Progress-aware candidate selection

**Files:**
- Create: \`src/ecsa/world_model/progress_selection.py\`
- Test: \`tests/test_progress_selection.py\`

**Interfaces:**
- Consumes: \`LearningProgressLedger.assess\`, \`ContractExperiment\`, existing EIG scores and action costs.
- Produces: \`SelectionBreakdown(experiment_id, information_gain_bits, uncertainty, progress_estimate, stagnation_penalty, cost, total_score)\` and \`ProgressAwareSelection.select(experiments: tuple[ContractExperiment,...], *, eig: dict[str,float], context_signature: tuple[str,...]) -> tuple[ContractExperiment,SelectionBreakdown]\`. Preserve the \`ScienceKernel\` EIG order when a candidate has independently established positive information value; bounded penalty is a tie-breaker or calibrated regularizer, never an unconditional ban.

- [ ] **Step 1 — RED:** Write \`test_unproductive_repeat_eventually_loses_to_alternative\`, \`test_independent_replications_retain_priority\`, \`test_only_feasible_action_never_disappears\`, \`test_selection_exposes_complete_score_breakdown\`, \`test_no_zero_division_with_unknown_gain\`. Test both syntactically distinct action names and argument permutations without special-case branches.
- [ ] **Step 2 — verify RED:** \`pytest -q tests/test_progress_selection.py\`; expected failure is the missing generic selector.
- [ ] **Step 3 — GREEN:** Implement deterministic finite scoring, a transparent bounded stagnation adjustment with no permanent blacklisting, stable ties by experiment ID, and explicit non-causal metadata for cycle evidence. Do not recompute a separate surrogate EIG in the selection layer.
- [ ] **Step 4 — verify GREEN:** \`pytest -q tests/test_progress_selection.py tests/test_learning_progress.py\`.
- [ ] **Step 5 — commit:** \`git add src/ecsa/world_model/progress_selection.py tests/test_progress_selection.py && git commit -m "feat: select experiments using observed learning progress"\`.

## Task 3: Wire into existing coordinator and scientist with ablation

**Files:**
- Modify: \`src/ecsa/world_model/experiments.py\` (\`ContractExperimentCoordinator.__init__\`, \`select_bootstrap\`, \`select_applicability\`, \`select_active\`).
- Modify: \`src/ecsa/autonomy/scientist.py\` (\`__init__\`, \`choose_experiment\`, \`observe_transition\`).
- Test: \`tests/test_scientist_learning_progress.py\`

**Interfaces:**
- Extend coordinator init with \`progress: LearningProgressLedger | None = None\` and \`use_progress_scoring: bool = False\`.
- Coordinator exposes \`last_selection_breakdown: SelectionBreakdown | None\`, never overwrites existing \`last_selection_mode\`.
- Scientist constructs progress from prequential action-success uncertainty and explicit \`WorldModelUpdate\` evidence statuses; **added or updated contract IDs are not automatically treated as validated science**. Provisional epistemic updates must be labeled separately from confirmed ones.

- [ ] **Step 1 — RED:** Write \`test_default_coordinator_ranking_unchanged\`, \`test_progress_enabled_breaks_repeated_unproductive_actions\`, \`test_world_model_proposals_not_counted_as_confirmed\`, \`test_failed_actions_count_cost_without_scientific_credit\`, \`test_existing_eig_decision_survives_progress_weight\`. Regression test must first demonstrate failure with enabled progress (not a vacuous run of the old code).
- [ ] **Step 2 — verify RED:** \`pytest -q tests/test_scientist_learning_progress.py\`.
- [ ] **Step 3 — GREEN:** Route candidate selection through the opt-in progress-aware path at the existing coordinator's final choice boundary and update progress after actual observed transitions. Use declared \`RawActionOutcome.success\` and before/after perception only; avoid calculating prospective scores with post-action observations.
- [ ] **Step 4 — verify GREEN:** \`pytest -q tests/test_scientist_learning_progress.py tests/test_world_model_experiments.py tests/test_autonomous_scientist.py\`; if these legacy filenames differ, identify actual existing test paths before executing, never silently skip.
- [ ] **Step 5 — commit:** \`git add src/ecsa/world_model/experiments.py src/ecsa/autonomy/scientist.py tests/test_scientist_learning_progress.py && git commit -m "feat: connect learning progress to autonomous scientist"\`.

## Task 4: Domain-independence acceptance and matched policy ablation

**Files:**
- Create: \`tests/test_generic_learning_progress_arena.py\`
- Modify: \`docs/experiments/2026-10-09-generic-learning-progress.md\` (create if missing)

**Interfaces:** Two tiny in-memory \`RawEnvironmentPort\` fixtures, with unrelated object names and action symbols, both using actual \`AutonomousScientist\`, \`ContractExperimentCoordinator\`, \`WorldModelAcquisitionKernel\` and a simple \`PerceptionFrontend\`. Compare fixed budget and fixed observable action options with \`use_progress_scoring=True/False\`.

- [ ] **Step 1 — RED:** Define tests \`test_repetition_reduces_without_domain_cues\`, \`test_all_actions_uninformative_stays_bounded\`, \`test_no_unbounded_contract_generation_under_cycles\`, \`test_distinct_domain_symbols_preserve_decisions\`, \`test_action_budget_identical_across_ablations\`. The first must fail on the pre-feature baseline: otherwise strengthen fixture.
- [ ] **Step 2 — verify RED:** \`pytest -q tests/test_generic_learning_progress_arena.py\`.
- [ ] **Step 3 — GREEN:** Use integration interfaces established in Tasks 1–3; create two controlled fixture worlds; expose improvement and negative cases without touching benchmark adapters. Document both settings and limitations.
- [ ] **Step 4 — verify GREEN:** \`pytest -q tests/test_generic_learning_progress_arena.py tests/test_learning_progress.py tests/test_progress_selection.py tests/test_scientist_learning_progress.py\`; run repository test suite \`pytest -q\` and explicitly report failures/skips, including optional DiscoveryWorld dependencies.
- [ ] **Step 5 — commit:** \`git add tests/test_generic_learning_progress_arena.py docs/experiments/2026-10-09-generic-learning-progress.md && git commit -m "test: verify domain-neutral progress and bounded exploration"\`.

## PR and review gate

- Open **implementation PR A** stacked on the approved design branch; keep main unchanged.
- Reviewer checks RED→GREEN evidence, candidate ranking in all three existing modes, and no domain strings in new core modules.
- Report episode-level budgets and negative trials, not just average score. If progress tracking cannot establish genuine improvement, retain the result as a negative and do not relabel it as solved.
- After PR A passes focused and full CI, proceed to the separate effect-attribution plan. Do not claim the whole three-part architecture is implemented by this first PR.
