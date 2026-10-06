# Integration Plan v1

## Objective

Build the smallest end-to-end system that proves the architecture can coordinate existing causal/scientific components before adding more engines.

The implementation strategy is **adapter-first**: external systems remain independent and are connected through typed contracts. No external implementation is re-created unless a verified integration gap requires it.

## Phase 0 — Frozen contracts

Define the minimum shared types:

- `TheoryRef`
- `ExperimentSpec`
- `Observation`
- `ProspectivePrediction`
- `EvidenceRecord`
- `TheoryPosterior`
- `PopulationAnomaly`
- `TheorySpaceExpansionRequest`

Keep these independent of PiEvo, BCS, DreamCoder, JAX, PDDL, and Java/LearnLib types.

Acceptance:

- deterministic serialization;
- versioned IDs;
- one evidence event can be traced back to the exact theory/model artifact and prediction made before observation.

## Phase 1 — PiEvo scientific kernel extraction

Reuse/adapt the PiEvo algorithms for:

- priors and posterior beliefs over theories;
- entropy;
- hypothetical posterior;
- BALD/information gain;
- information-directed selection;
- coherent prior mass for a newly admitted theory.

Do not bring the AutoGen P-H-E agent shell into the first runtime.

Required semantic changes:

- zero-valued outcomes remain valid evidence;
- computational failure is not likelihood zero;
- all-current-theories-fail becomes `PopulationAnomaly`, not a uniform reset;
- distinguish MAP/leader failure from population-wide failure.

Acceptance:

- synthetic likelihood table reproduces expected Bayesian posterior;
- a new theory can be admitted with explicit small prior mass;
- posterior updates are replay-deterministic from the evidence ledger.

## Phase 2 — Existing BCS inference integration

Connect the current Behavioral-Causal-State inference API rather than introducing a new predictor algorithm.

For BCS-supported theory artifacts:

- translate `ExperimentSpec` into `Predictive` or `Interventional` queries;
- call `bcs.inference.evaluate()`;
- retain `ExactDistribution`, `InferenceUndefined`, and `InferenceIncomplete` distinctions;
- obtain the likelihood of the subsequently observed outcome from the prospective distribution.

Counterfactual queries remain available for explanation/evaluation but are not automatically treated as executable experiments.

Acceptance scenario:

Two BCS models make different prospective predictions for the same intervention. The scientific kernel selects the discriminating experiment, the environment returns a valid outcome (including zero), and posterior mass moves toward the model that predicted it.

## Phase 3 — Population-anomaly gate

Implement the explicit difference between:

- leader anomaly: current MAP model fails but the active population contains a good explanation;
- population anomaly: the active population collectively assigns insufficient predictive support.

Population anomaly emits a typed repair request containing the contradicting evidence and relevant theory versions.

Acceptance:

- leader failure only reweights existing theories;
- population failure requests model-space expansion;
- no automatic uniform reset.

## Phase 4 — Repair/proposal engines

Connect specialized engines behind proposal interfaces. They are complementary rather than mutually exclusive.

### TTT / mut-learn

**Status: LearnLib TTT backend integrated for the frozen BCS deterministic fixtures.**

Use for resettable deterministic finite-state regions:

- behavioral state discovery;
- memory requirements;
- counterexamples.

The current adapter reuses the existing BCS TTT bridge at the pinned BCS
revision, executes the real Java/LearnLib learner, and returns a
content-addressed `TheoryProposal` in the `STATE_MEMORY` repair family.

**State projection is now integrated.** A minimal learned Mealy machine can be
projected into a residual-state feature. Admission is gated by an aliasing
witness: the same current causal-state key must correspond to two different
minimal TTT residual states. A qualified state-augmented theory receives an
explicit small prior mass while preserving the relative mass of existing
theories. For interventions present in the frozen TTT alphabet, the projected
theory emits `PredictiveDistribution` values in the same experiment space as
BCS theories and can therefore participate in information-directed selection.

The current end-to-end test uses a deliberately coarse synthetic causal-state
key to validate the mechanism. Demonstrating genuine missing state in a real
BCS representation remains an experimental acceptance criterion rather than
an architectural assumption.

### MMAE / IMM

Use when the repair question is selection among a known local family of modes/models.

### BOCPD

**Status: integrated with a pinned external BOCPD implementation.**

Use when evidence suggests a temporal regime shift.

The current backend uses `fiannai/bocd` at a pinned revision and feeds it a
cross-experiment signal derived from the evidence ledger:

`1 - P(observed outcome | current theory population)`.

This avoids treating a change in intervention itself as a time-series regime
change. Detection uses a sharp reset in the MAP run length (the recommended
signal for constant hazard in the upstream library), plus a minimum
post-change run length before emitting a `REGIME_CHANGE` proposal. A lone
terminal anomaly therefore remains available to TTT, causal-structure, fault,
or program-repair engines without being prematurely classified as a temporal
regime change.

The resulting diagnostic artifact records the discrepancy signal, experiment
IDs, run-length trace, detector parameters, reset step, and pinned upstream
revision.

### Fault Diagnosis Toolbox

Use when sensor/actuator/component fault explanations are plausible.

### CausalExploration

Use for structural causal alternatives and discriminating interventions.

### VACERL

Use for significant-event and candidate-variable proposals from richer trajectories.

### DreamCoder

Use when an executable mechanism/program must be synthesized.

Each engine returns proposals with provenance; it does not directly overwrite the current theory population.

Acceptance:

A population anomaly can produce multiple typed repair candidates from different engines, all of which enter the same scientific comparison process.

## Phase 5 — Mechanism abstraction and transfer

### Stitch

Run over validated/surviving programmatic mechanisms to propose common abstractions.

### VARIO

Evaluate whether a proposed abstraction should be:

- shared unchanged;
- shared with context-specific parameters;
- split into context-specific mechanisms.

A compressed abstraction remains provisional until transfer/intervention evidence supports it.

Acceptance:

The same mechanism discovered in multiple contexts is reused with fewer new experiments than learning from scratch, while a deliberately changed context triggers correct specialization rather than false reuse.

## Phase 6 — Goal-directed planning

Planning is layered onto the scientific state rather than replacing it.

### BAMCP / pymdp

Use when uncertainty over surviving theories changes action value and epistemic actions matter.

### Fast Downward / TheoryCoder-style PDDL projection

Use for grounded long-horizon symbolic plans.

Decision-relevant disagreement between surviving theories can request another experiment instead of blindly committing to one plan.

Acceptance:

The agent reaches a goal with fewer interactions than a random/exploration-only baseline while still resolving uncertainty when competing theories imply different consequential actions.

## Phase 7 — DreamerV3 shadow model

Run Dreamer as an independent learned dynamics backend.

Initial permissions:

- receive the same observation/action stream;
- predict/imagine future trajectories;
- report disagreement/uncertainty.

It does not directly mutate the causal theory archive.

Use latent disagreement to propose missing-state or memory hypotheses.

Acceptance:

A partially observed fixture causes Dreamer to separate histories that the current symbolic state aliases; the system proposes and experimentally tests a state/memory repair.

## Phase 8 — Full scientific-agent arena

Compare:

- BCS-only fixed model class;
- PiEvo-style scientific kernel without repair engines;
- TTT-only;
- DreamCoder-only mechanism proposal;
- no mechanism reuse;
- no active experiment selection;
- full ECSA stack.

Primary metrics:

- goal success vs interaction budget;
- prospective predictive log loss;
- interventional accuracy;
- counterfactual accuracy where ground truth is available;
- time to detect model/regime misspecification;
- false mechanism split / false merge;
- transfer sample efficiency;
- description length / library compression;
- compute cost.

## First vertical slice

Do **not** begin with all integrations.

The first executable milestone is:

```text
two BCS theories
      |
PiEvo-derived posterior + IDS
      |
automatically selected intervention
      |
environment observation
      |
existing BCS evaluate() distributions
      |
Bayesian reweighting
      |
next experiment/decision
```

Then inject one observation that both theories fail to explain and require a `PopulationAnomaly`.

Only after that milestone passes should TTT/DreamCoder/Stitch and diagnostic engines be connected.

## External-code policy

- Prefer dependencies, subprocess workers, or adapters over vendoring.
- Preserve upstream licenses and attribution.
- Do not copy code from repositories without a compatible license.
- Keep external runtime types outside the ECSA core domain model.
