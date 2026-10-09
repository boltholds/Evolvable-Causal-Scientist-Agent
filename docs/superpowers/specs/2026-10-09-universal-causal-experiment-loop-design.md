# Universal Active Causal Learning and Evaluation — Design

**Date:** 2026-10-09  
**Status:** Design for review; implementation is NOT authorized by this document alone.  
**Base:** ECSA world-model kernel, branch feat/discoveryworld-self-induced-interfaces-20261009.

## 1. Research objective and observed failure

Build a domain-neutral research loop that (1) detects when experiments stop producing knowledge, (2) distinguishes environmental background dynamics from action-contingent observations and designs interventions to test causal alternatives, and (3) evaluates different representation and prediction backends against the same frozen, public-world targets.

The immediate trigger was the independent DiscoveryWorld study: a repeated public feed action accounted for 825/1000 heldout interactions, while step counters and changing image payloads were frequently learned as action effects. Action-JEPA showed effective rank near 1, and the claimed common public-delta target differed between the public_hash and induced_interface encodings. **Those are diagnostics, not training labels and not hard-coded rules for this design.**

ECSA must learn *semantic* interfaces from experience. The transport layer still exposes legal action schemas and observable wire fields. We never infer unsupported API syntax or declare an unknown action valid merely because a model proposed it.

Success means an agent can handle a completely different raw-world implementation without changing these new core modules: no DiscoveryWorld imports, no action-specific rankings or names, no sensor-field blacklist, and no oracle labels.

## 2. Alternatives

**A. Benchmark-specific guards and field filters.** Quick initial metrics, but hard-codes alleged scientific importance and fails the transfer objective. Rejected.

**B. Monolithic neural latent world model controlling all experiments, perception, and inference.** Potentially expressive but lacks inspectable epistemic state, matched interventions, and independent measurement of evidence quality. Rejected as the first integration step.

**C. Typed, evidence-centric domain-neutral extensions to existing ECSA ports.** Chosen. Preserve RawEnvironmentPort, PerceptionFrontend, InteractionGrounder, WorldModelAcquisitionKernel, ContractExperimentCoordinator, ScienceKernel and the existing action-schema hypothesis machinery. Keep neural, symbolic and statistical predictors replaceable.

## 3. Architectural flow and boundaries

    RawEnvironmentPort
      -> RawObservation + public RawActionSchema[]
      -> PerceptionFrontend
      -> GroundAction (selected through existing coordinator)
      -> RawActionOutcome + next RawObservation
      -> InteractionGrounder
      -> WorldModelAcquisitionKernel
           |-> EffectAttributionLedger / competing explanations
           |-> LearningProgressLedger / experiment utility
           |-> Scientific memory / versioned proposed contracts
      -> ScienceKernel + ContractExperimentCoordinator
      -> next public GroundAction

A separate **SharedOutcomeFrame** consumes the same public event ledger for train/validation/test scoring. It never changes the agent's choices during the evaluation episode and never reads evaluator scorecards. The common scoring frame is *not* a model-specific embedding.

The benchmark adapter exclusively handles loading an environment, converting its public wire data, recording episodes and reading evaluator-only metrics after completion. It must not encode substantive scientific roles.

Existing Event/WorldModel structures remain authoritative. New types are adapters/projections over InteractionTransition, GroundingUpdate, FeatureDelta and typed hypothesis updates, not a parallel discovery-world data model.

## 4. Learning Progress and nonproductive cycles

### 4.1 Core API

New domain-neutral module: src/ecsa/world_model/learning_progress.py.

- LearningProgressSignal: immutable evidence references, action signature, context fingerprint, information-gain estimate, update in calibrated predictive quality, accepted/contradicted *hypothesis revision*, intervention-confirmation count and executed action cost.
- LearningProgressLedger.observe(signal): appends a bounded history and updates per-context/per-action progress.
- LearningProgressLedger.evaluate(candidate): provides observed novelty, estimated future information yield, cost, uncertainty and a bounded stagnation penalty.
- LearningProgressLedger.detect_cycle(): reports recurring *equivalent* public states and action sequences, with explicit partial-observability uncertainty.
- ProgressAwareSelection: a tie-breaker/regularizer over existing experiment candidates, never a replacement for the ScienceKernel EIG computation.

**Knowledge gain is NOT** (a) successful API response alone, (b) existence of another unique evidence ID, (c) an advancing clock, (d) a visual frame difference, or (e) repetition of the same unsupported hypothesis.

Count knowledge gain from a prospective prediction improvement on held-back observations; reduction in uncertainty over competing hypotheses; a newly *independently supported* rule, or a confirmed falsification. The update must carry provenance and must not use the same observation both to propose and confirm a rule.

When a candidate is repeatedly uninformative in equivalent public contexts, its expected benefit falls and other feasible candidates become preferable. The penalty saturates; repeated matched trials needed for hypothesis confirmation are NOT blocked. A single-action environment must still make progress or report that no discriminating action is available. No spontaneous creation of unlimited hypothetical graph nodes merely because an agent cycles.

The coordinator must preserve a public, testable breakdown of why an experiment was selected: information gain, uncertainty, expected progress, repetition/stagnation and cost. Continuous EIG remains the first-class scientific signal.

### 4.2 Integration

Minimal changes to src/ecsa/world_model/experiments.py and src/ecsa/autonomy/scientist.py: after observing a transition, hand generic evidence quality and any new tested contract updates to the progress ledger; before selecting the next experiment, include progress in candidate comparison. Legacy behavior can be selected through an explicit ablation flag and tested side by side. Do not hard-code special responses to feed/poll actions.

## 5. Action effect versus background dynamics

### 5.1 Competing hypotheses

New domain-neutral module: src/ecsa/world_model/effect_attribution.py.

For each *role-normalized public feature change*, construct explicit rival explanations:

- BACKGROUND: this feature changes under multiple actions or outside the putative intervention.
- STATE_DEPENDENT: differences are explained by observed pre-action context.
- ACTION_DEPENDENT: a candidate action-specific effect after state matching.
- INTERACTION: the candidate effect depends jointly on action and observed context.
- UNRESOLVED: insufficient comparable evidence or confounding.

Action symbols and argument roles are discovered from public schemas and matches of arguments with observed entities. Public identity may align entities *within evidence*, but must not become a semantic feature or allow cross-seed UUID memorization. Anonymous/unmatched objects remain unknown, not silently merged.

### 5.2 Evidence quality states

AttributionClaim records: role-relative feature key, candidate action/schema, compared contexts, observed change statistics, origin of all evidence, proposed prediction, confidence bounds and status:

- PROPOSED: generated from public transitions.
- OBSERVATIONALLY_SUPPORTED: matches heldout public observations, including matched-action controls.
- INTERVENTIONALLY_SUPPORTED: survives **prospective**, independent, controlled or randomized intervention where supported by the environment.
- CONTRADICTED / INSUFFICIENT: conflicting or inadequately comparable evidence.

Even INTERVENTIONALLY_SUPPORTED is bounded by the tested intervention/context. Do not use a boolean called "causal_proven". A simple observational difference before/after, including one failed API action, is not a causal control and cannot promote a claim to the interventional state.

EffectAttributionLedger takes grounded transitions plus optional intervention-cohort provenance. It must estimate the background change rate conditional on observed context and compare alternative actions in overlapping contexts. If there are no matched controls, emit UNRESOLVED. If a public feature changes under essentially all actions, prefer BACKGROUND over a spurious action effect; this is learned from data, never from sensor field names.

Use independent cohorts for (i) candidate discovery, (ii) contrast selection, (iii) prospective confirmation. A ScienceKernel experiment may be chosen to distinguish competing explanations. Active controls consume the same real environment action budget as experimental actions.

Reporting must distinguish predicting observed changes from identifying intervention effects. Inability to reset or randomize an arbitrary real environment is exposed as an evidence limit rather than faked with retrospective data.

## 6. Unified prediction and evaluation

New core module: src/ecsa/world_model/shared_evaluation.py.

SharedOutcomeFrame.fit(train_public_transitions) freezes the **same** training-derived role-relative public target vocabulary and per-feature scaling for all compared predictors. It cannot inspect validation/test samples when discovering targets or normalizers. Its structure contains observed numeric changes, categorical/binary changes, outcomes of attempted actions and explicit missing/unseen-feature masks. No target coordinate comes from a particular encoder's latent vectors.

PredictiveBackendPort predicts distributions or point estimates for these canonical targets from the current public observation and action. Adapters for raw Ridge, Action-JEPA, KAN, symbolic rules and persistence must all map to *this fixed target frame*. A JEPA latent readout may be learned on training data only; it must not fit on validation/test.

Score a single frozen heldout cohort with:

- continuous target error using train-derived scaling and a shared target mask;
- categorical/change detection, coverage, macro-by-feature and macro-by-action performance;
- outcome success Brier score when outcomes are actually visible;
- prospective hypothesis calibration and interventional-confirmation rate, reported separately;
- action/observation budgets, action distribution and effective-rank diagnostics;
- unseen features, feature drift and abstentions reported rather than dropped without accounting.

Equal-size denominators, masks and target scales across backends are preconditions for a claim of neural-versus-symbolic superiority. Latent MSE and internal representation rank are supplemental diagnostics, never cross-model scientific performance scores.

Train/test independence is keyed to complete environment seeds/episodes, not paired cold-vs-reuse clones. Tune only on training and designated validation seeds. Keep test seed inaccessible to feature discovery, experiment policy tuning and model selection.

## 7. Contracts remain learned, versioned and falsifiable

Existing entity types, argument roles, predicates, action schema hypotheses, affordances and contract canonicalization remain the source of semantic contracts. The core may propose new types/features/effects and fork an immutable contract revision following repeated unexpected predictions. A failed prediction generates a candidate refinement, not automatic promotion or silent overwrite.

Every feature/role/predicate has:
- provenance IDs with cohort/source;
- its observed state and attempted action scopes;
- explicit assumptions, missingness and certainty;
- evidence supporting and contradicting it;
- status and version/fingerprint.

Invariants: raw protocol syntax is known from the adapter; scientific meaning is **not**. No domain vocabulary in the core. Unknown source identity, unknown outcome or unknown effect are typed epistemic states, not fabricated defaults. A schema learned from one object must be tested on unseen objects before any lifted transfer claim.

## 8. Implementation slices and acceptance gates

Three sequential, independently testable implementation PRs, stacked over the current experimental branch (without merging into main unless explicitly requested).

**Slice 1: Progress-aware autonomous experiment coordination.** Add core progress types/tracker; connect the scientist and coordinator without changing their public environment contract; test eventual escape from an unproductive action loop, preservation of necessary independent repetitions, all-actions-uninformative abstention, finite memory and no generated new graph nodes. Compare opt-in and opt-out policies on matched observations.

**Slice 2: Observational and interventional background attribution.** Add typed rival effect hypotheses and evidence cohorts; connect to existing grounding/contract proposal pipeline. Test generic increasing counter under unrelated actions, state-confounded change, matched controls, mixed action/context effects, unobservable/unstable object identity, disjoint confirmation evidence and inability to claim causality from offline logs.

**Slice 3: Frozen common target frame and cross-model scoring.** Implement train-only frame, common masks and model adapter protocol. Test JEPA vs RawRidge vs persistence with exactly equal physical target coordinates and denominators, categorical/numeric/missing samples, unseen heldout features and action macro aggregation. Demonstrate the same core classes with two unrelated fake RawEnvironmentPort fixtures that contain no shared domain nouns.

For the final benchmark validation, use existing DiscoveryWorld Reactor Lab as **one** external fixture, including seeds 0–2/3/4 and matched action budgets. Additionally run an unrelated synthetic or alternate-domain environment to show the core is domain-independent. Report official environment success only from post-run evaluator results; it does not flow into the learner.

## 9. Explicit non-goals and limitations

- No manual rule identifying any benchmark action, instrument, scene object or field as scientifically important or unimportant.
- No model scaling or reinforcement-learning redesign in this architectural slice.
- No automatic declaration of a full causal graph or unique mechanism from noncommutativity or observational prediction.
- No proof of causal identification under unmeasured confounding or non-resettable environments.
- No requirement for a large new external library; use existing ECSA primitives and standard scientific dependencies.
- Do not remove prior JEPA/DiscoveryWorld baselines or rewrite the world-model acquisition kernel in a new framework.

## 10. Ready-for-implementation checklist

Before implementation planning, review that the chosen boundaries represent the intended system. Once approved, write a separate TDD plan in docs/superpowers/plans, execute native RED→GREEN slices with checked commits, run relevant focused and full test suites, and report all failures and benchmark negative results as observed. A passing integration smoke without task completion does not count as research success.
