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

**Status: integrated for the first explicit Boolean mechanism domain.**

Use when an executable mechanism/program must be synthesized.

The current backend pins `ellisk42/ec` at a fixed revision and reuses
DreamCoder's own type system, `Primitive`, `Task`, `Grammar`, and
`Grammar.enumeration()`. It does not install DreamCoder's historical full
requirements stack. An isolated worker loads only the core submodules from the
pinned source tree, bypassing the legacy package initializer that eagerly
imports unrelated recognition, regex, tower, logo, and other domain modules.

The first domain adapter accepts typed Boolean input/output examples plus an
explicit primitive set. A synthesis run emits a content-addressed
`PROGRAM_MECHANISM` proposal containing the exact program, log prior / MDL,
DSL, examples, enumeration count, search budget, and upstream revision.

The acceptance test synthesizes XOR behavior from the restricted DSL
`{not, and, or}`, so the target operation is not supplied as a primitive.
A negative-control DSL containing only `and` yields no proposal rather than
inventing unsupported behavior.

Synthesized programs are proposals, not automatically accepted causal laws.

**Program projection and admission are now integrated for the Boolean domain.**
A `DreamCoderProgramPredictionAdapter` evaluates the stored program through
the same pinned DreamCoder `Program.parse/evaluate` runtime and maps explicit
Boolean `ExperimentSpec` interventions to `PredictiveDistribution`.

A `ProgramMechanismProjector` requires at least one held-out prospective
observation whose experiment ID is absent from the synthesis examples. If the
program gives the observed held-out outcome probability 1 under the current
deterministic Boolean adapter, it becomes a content-addressed
`program-mechanism` `TheoryRef`. It can then be admitted with explicit small
prior mass and participate in the same information-directed experiment
selection as BCS theories. A held-out mismatch produces
`RejectedProgramMechanism` and no posterior admission.

The current vertical slice synthesizes `Y = M and not C` from three Boolean
examples and reserves the fourth truth-table case for prospective validation.
After admission, IDS prefers an intervention where the qualified program and a
zero-output BCS theory disagree.

Each engine returns proposals with provenance; it does not directly overwrite the current theory population.

Acceptance:

A population anomaly can produce multiple typed repair candidates from different engines, all of which enter the same scientific comparison process.

## Phase 5 — Mechanism abstraction and transfer

### Stitch

**Status: integrated with the pinned MIT Rust Stitch core.**

Run over validated/surviving programmatic mechanisms to propose common abstractions.

The current backend pins `mlb2251/stitch` at revision
`350804b7b35807c78bd21c313785ae5152ae2985` and builds/runs the upstream
`compress` binary. It consumes only `QualifiedProgramMechanism` values:
programs that have already passed the DreamCoder held-out prospective
validation gate. Raw synthesis outputs are rejected by the adapter boundary.

The upstream `programs-list` format is used directly, so DreamCoder lambda
program strings are not translated into a new internal language. For each
learned abstraction ECSA stores a content-addressed artifact with:

- exact upstream revision and Stitch arguments;
- source theory IDs and source DreamCoder program artifact IDs;
- original and rewritten programs;
- learned abstraction body and arity;
- utility, number of uses, final cost, and compression ratios.

A single qualified mechanism returns no abstraction proposal: the current
ECSA contract reserves Stitch for cross-mechanism reuse. The positive
acceptance test runs the real Rust compressor over three qualified mechanism
programs sharing repeated structure and requires positive utility,
multi-use support, and corpus compression.

A Stitch abstraction is a `StitchAbstractionProposal` in the
`ABSTRACTION_TRANSFER` family, not a causal `TheoryRef`. Compression proves
repeated program structure, not causal invariance. VARIO or equivalent
cross-context validation must decide whether the abstraction is shared,
parameter-specialized, or split before it becomes a reusable mechanism in the
library.

### VARIO

**Status: integrated with the official authors' R source archive.**

Evaluate whether a proposed abstraction should be:

- shared unchanged;
- shared with context-specific parameters;
- split into context-specific mechanisms.

The adapter uses the official August 2022 VARIO source archive from the
authors' project page and pins the exact archive bytes by SHA-256
`67df25a256622f86fe7c7b469e2928ddcd2252680b18502699786041ca554652`.
The archive is not copied into ECSA. It is downloaded into a runtime cache,
checksum-verified, and executed through R.

The current transfer adapter targets VARIO's `Vario_Pi_search` API for one
known mechanism input and target across multiple numeric contexts. Each
context is supplied as `VarioContextEvidence` and converted to the official
`x1, y` data convention. The first integration uses degree-1 regression;
this is a bounded acceptance surface, not a claim that arbitrary Stitch
lambda programs have been reduced to linear regressions.

For changing mechanisms the adapter consumes VARIO's MDL-ranked partition.
For the fully invariant case it respects VARIO's own
`invariant_conservative` result and `Pi_ranking_correctone` rather than
overriding that correction with the raw MDL ranking.

ECSA then projects the resulting VARIO partition into its own transfer
vocabulary:

- one context group → `SHARED`;
- more than one group with at least one multi-context group →
  `CONTEXT_SPECIALIZED`;
- every context in a singleton group → `SPLIT`.

This three-way vocabulary is an ECSA integration contract, not terminology
attributed to VARIO itself.

The content-addressed transfer artifact records the Stitch abstraction ID,
context IDs and evidence hashes, discovered VARIO partition and score, source
URL/checksum, and the resulting ECSA scope. The official source archive
contains no `LICENSE` file, so the artifact explicitly records unknown
source licensing and ECSA does not vendor or relicense that source.

Acceptance tests exercise the real R implementation on three cases:
fully invariant contexts, a two-context shared mechanism plus one changed
context, and three distinct context mechanisms.

A compressed abstraction remains provisional until this transfer evidence
supports its scope.

### Mechanism Library admission and applicability

**Status: core contract and first MLMD/SQLite repository implementation integrated.**

The authoritative schema and repository boundary are defined in
[MECHANISM_LIBRARY.md](MECHANISM_LIBRARY.md).

ECSA owns only:

- the typed mechanism schema;
- epistemic status;
- scope/assumptions;
- admission semantics;
- applicability semantics;
- scientific relations;
- projections into execution backends.

ECSA does **not** implement a second general-purpose artifact store, lineage
database, or model registry.

The persistence boundary is:

```python
class MechanismRepository(Protocol):
    def admit(...): ...
    def get(...): ...
    def find_applicable(...): ...
    def find_transfer_candidates(...): ...
    def lineage(...): ...
```

The first implementation is `MLMDMechanismRepository`, using ML Metadata
1.21 for artifacts, executions, events, contexts, and lineage while keeping
MLMD protobuf/runtime types outside the ECSA core domain. The current
acceptance backend uses SQLite.

Initial mechanism kinds:

- `PROGRAM`;
- `CAUSAL_MODEL`;
- `STATE_FEATURE`;
- `SYMBOLIC_RULE`.

Initial epistemic statuses:

- `CANDIDATE`;
- `QUALIFIED`;
- `ADMITTED`;
- `DEPRECATED`.

Initial transfer statuses:

- `SHARED`;
- `CONTEXT_SPECIALIZED`;
- `SPLIT`.

Admission is scientific rather than merely persistent. A representation is
not admitted because it exists in DreamCoder, Stitch, TTT, or BCS.

Minimum admission rules include:

- synthesis/training evidence must remain distinct from prospective validation;
- DreamCoder programs require prospective qualification;
- Stitch compression does not establish causal validity or transfer;
- global/cross-context reuse requires explicit transfer evidence;
- TTT state features require an aliasing witness;
- contradicting evidence is retained and may narrow scope, create a
  specialization/new version, supersede, or deprecate a mechanism.

Applicability queries are conservative:

- only `ADMITTED` mechanisms are returned by ordinary reuse queries;
- all explicit scope and assumption constraints must match;
- `SPLIT` mechanisms are never returned as globally shared;
- deprecated mechanisms remain available for lineage but not normal runtime
  reuse;
- unknown transfer into a new context triggers validation rather than
  optimistic reuse.

Runtime backends consume filtered projections. DreamCoder Grammar, a BCS model
set, or planner operators are therefore **views of applicable mechanisms**,
not the long-term source of truth.

**Runtime projection status: first executable layer integrated.**

- `MechanismRuntimeProjector` exposes typed artifact-only views for applicable
  `CAUSAL_MODEL`, `STATE_FEATURE`, and `SYMBOLIC_RULE` mechanisms.
  Concrete BCS/state/planner hydration remains owned by those backends rather
  than being fabricated in the mechanism library.
- `DreamCoderGrammarProjector` resolves applicable `PROGRAM` mechanisms,
  including qualified `program-mechanism` artifacts, back to the exact
  source DreamCoder program.
- `DreamCoderRepairEngine` accepts those projected programs as a runtime
  library, and the isolated worker adds them to the upstream Grammar as real
  `Invented` productions.
- An end-to-end reuse test learns and qualifies a mechanism in one context,
  admits it with shared scope covering a second context, and requires warm
  DreamCoder synthesis in the second context to enumerate strictly fewer
  programs than cold-start synthesis.

This verifies computational reuse. The stronger transfer goal—fewer
environment interventions/experiments in a new context—remains an arena
experiment rather than being inferred from synthesis speed.

Current repository acceptance:

1. admitted mechanisms and exact representation/evidence/provenance inputs are
   persisted as MLMD artifact/execution/event lineage;
2. stable identity/version lookup is implemented;
3. applicability is filtered conservatively by explicit context, regime,
   domain, task, and required assumptions;
4. `SUPERSEDES` relations hide superseded mechanisms from ordinary reuse
   without deleting their historical versions;
5. deprecation creates a new immutable `DEPRECATED` version and preserves the
   prior admission lineage;
6. implicit global scope is rejected;
7. the core remains isolated from MLMD protobuf/runtime types;
8. applicable program mechanisms can be projected into a real DreamCoder
   Grammar without making that Grammar authoritative;
9. typed artifact views for causal models, state features, and symbolic rules
   preserve mechanism kind boundaries.

Backend-specific BCS model hydration and symbolic planner-operator hydration
remain future adapter work.

### DiscoveryWorld Reactor Lab transfer arena

**Status: executable harness integrated; scientific transfer result not yet claimed.**

The first external environment is the Apache-2.0 DiscoveryWorld benchmark,
pinned to `allenai/discoveryworld` revision
`fd591323920be0d3786ef350955de1945aa571e5`.

The first arena is fixed to:

- scenario: `Reactor Lab`;
- difficulty: `Normal`;
- official seeds: `0,1,2,3,4`;
- one agent;
- maximum model-backed budget: 1000 environment interactions.

`DiscoveryWorldEnvironmentAdapter` exposes only ordinary agent observations,
action descriptions, teleport locations, and action results. Each
`performAgentAction()` is followed by exactly one `tick()`, including failed
actions. `getTaskScorecard()`, critical questions/hypotheses, hidden
`scoringInfo`, object internals, and full world history remain evaluator-only.

The Reactor Lab sidecar converts public instrument-result text into typed
measurements, freezes linear frequency hypotheses before their predicted
outcomes, and validates them only from later public reactor state. Successful
validation produces a context-scoped `SYMBOLIC_RULE` mechanism; transfer
hypotheses preserve `SPECIALIZES` lineage to the source mechanism.

Cross-context lookup is deliberately separated:

- `find_applicable(context)` requires the current seed/context to be in scope;
- `find_transfer_candidates(context)` ignores only context membership while
  requiring compatible regime/domain/task/assumptions;
- transfer candidates are guidance for target validation, not executable facts;
- failed source-backed prospective validation is counted as a false transfer.

The arena implements progressive online transfer in seed order 0→4:

```text
cold:
  each seed -> fresh MechanismRepository

reuse:
  seed 0 -> persistent repository -> seed 1 -> ... -> seed 4
```

This avoids future-seed leakage. Both arms instantiate fresh copies of the same
policy factory from the same JSON config and therefore share a deterministic
policy-config hash; the intended independent variable is accumulated mechanism
memory.

Per-run logs separate policy-facing and oracle information and include actions,
observations, scientific events, mechanism events, metrics, and final
scorecards. Metrics distinguish measurements, candidate retrieval/testing,
accepted/rejected transfer, and false transfer. Transfer gain uses
`1 - reuse/cold` and remains undefined when the cold denominator is zero.

CI runs a real deterministic no-LLM smoke for both arms across all five
official seeds with one interaction per episode. That smoke validates the
pinned environment, progressive orchestration, filesystem artifacts, and
oracle firewall. It does **not** establish that reuse reduces environment
sample complexity.

The scientific acceptance experiment is still:

- run the same model-backed policy in cold and reuse arms at the full
  1000-interaction limit;
- compare raw per-seed results for seeds 1-4;
- report success/procedural score, environment steps, scientific measurements,
  accepted/rejected transfer, and false-transfer cost;
- claim transfer benefit only if the measured environment interaction burden
  improves without an unacceptable increase in false transfer.

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
