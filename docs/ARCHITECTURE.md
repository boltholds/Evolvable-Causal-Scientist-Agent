# Architecture v1

## 1. Goal

The Evolvable Causal Scientist Agent (ECSA) is intended to operate in an initially unknown environment, pursue a supplied goal, perform interventions, maintain competing explanations, revise its theory space when evidence demands it, and reuse validated mechanisms across tasks and contexts.

The architecture deliberately composes existing research systems. A component is retained when it solves a distinct subproblem well; overlap is handled by assigning ownership rather than deleting methods.

## 2. Ownership boundaries

### Scientific coordination — PiEvo-derived kernel

The scientific kernel owns:

- the active set of world theories;
- priors and posterior beliefs over those theories;
- posterior entropy / epistemic uncertainty;
- information-directed experiment selection;
- anomaly detection;
- requests to expand or repair the current theory space.

PiEvo's terminology is adapted:

- PiEvo **Principle** ≈ ECSA **Theory / WorldHypothesis**;
- PiEvo **Hypothesis** ≈ ECSA **experiment or intervention candidate**;
- PiEvo **Experiment** ≈ ECSA **execution + observation**.

The original PiEvo GP/embedding predictor is not the causal authority in ECSA. For BCS-supported models, causal likelihoods come from BCS.

Two PiEvo semantics must be changed:

1. A valid outcome value of zero is evidence and must not be skipped.
2. If every current theory assigns negligible support to an observation, the system must emit a theory-class misspecification / population-anomaly event rather than resetting beliefs to uniform.

The architecture also distinguishes:

- **leader anomaly** — the MAP theory failed but another active theory explains the observation;
- **population anomaly** — the current theory population as a whole cannot adequately explain the observation.

Only the second necessarily requests theory-space expansion.

### Causal semantics — Behavioral-Causal-State

BCS owns causal computation for model classes it supports.

Existing operations are reused:

- `Predictive`
- `Interventional`
- `Counterfactual`
- `bcs.inference.evaluate()`
- `ExactDistribution.probability()`

No new causal-prediction algorithm named `BCSTheoryPredictor` is required. If an adapter class with that name exists later, it is only an interface bridge around existing BCS inference.

BCS additionally preserves distinctions such as:

- successful exact distribution;
- undefined inference;
- incomplete computation / exhausted budget;
- identification and witness information where available.

A failed computation is not equivalent to likelihood zero.

### Evidence

Scientific evidence is broader than `bcs.inference.History`.

`History` remains the factual trajectory supplied to one causal query. The scientific runtime needs an experiment ledger containing, at minimum:

- experiment identity and version;
- theory versions evaluated before observation;
- prospective predictions;
- executed intervention/action;
- observed outcome;
- context and provenance;
- computation status.

Predictions are fixed before the outcome is observed.

## 3. Specialized engines

### Robot Scientist

Robot Scientist is retained as the reference pattern for the complete scientific process:

`theory → prediction → discriminating experiment → observation → revision`.

It is not discarded in favor of PiEvo. PiEvo supplies a modern reusable coordination kernel for parts of this process; Robot Scientist remains the high-level scientific workflow model.

### LearnLib TTT + mut-learn

Used where the environment admits resettable finite-state interaction.

Responsibilities:

- active automata learning;
- membership/equivalence-style testing;
- shortest/candidate counterexamples;
- behavioral state discovery;
- evidence that additional memory/state may be required.

TTT is a specialized theory proposer / counterexample engine, not the global orchestrator.

### DreamCoder

Used to synthesize compact executable candidate mechanisms from transition/effect examples.

DreamCoder proposes programs; BCS/scientific evidence determines whether a proposal should be admitted as a causal theory.

### Stitch

Used to discover recurring abstractions across already-discovered programs.

A Stitch abstraction is a **candidate reusable mechanism**, not automatically a causal law. Admission requires validation under intervention and transfer.

### VARIO

Used to decide whether a mechanism should remain shared across contexts or specialize.

The target distinction is:

- same mechanism and parameters;
- same mechanism form with context-specific parameters;
- genuinely different mechanisms.

VARIO-style invariant/changing-mechanism evidence contributes to mechanism scope and transfer decisions.

### BOCPD

Used for online change-point detection.

It helps distinguish:

- a theory that was always wrong;
- a previously valid mechanism whose regime changed at a particular time.

### MMAE / IMM

Used for online weighting/filtering among a predefined family of dynamical models or modes.

This is complementary to the expanding PiEvo theory space: MMAE/IMM is valuable when the candidate model set is already known locally.

### Fault Diagnosis Toolbox

Used to generate and test structured fault/contradiction explanations.

A prediction failure may indicate a sensor/actuator fault or violated diagnostic assumption rather than a new world mechanism.

### VACERL

Used as a source of significant events, candidate causal variables, and hierarchical structure from richer observation streams.

It does not replace causal validation.

### CausalExploration

Used for active refinement of causal structure and experiment proposals aimed at distinguishing candidate causal graphs/mechanisms.

### BAMCP / pymdp

Used for action selection when model uncertainty itself matters to control.

They consume beliefs/models exposed by the scientific layer rather than owning the long-term theory archive.

### Fast Downward / TheoryCoder-style planning

Used when a sufficiently grounded symbolic theory can be projected into a planning representation.

Responsibilities:

- long-horizon symbolic goal planning;
- PDDL-level abstractions/operators;
- hierarchical planning/refinement where appropriate.

A plan may be generated separately under multiple surviving theories. Decision-relevant disagreement can return control to scientific experiment selection.

### DreamerV3

Used as a learned world-model backend for high-dimensional or expensive environments.

Initial role:

- shadow prediction;
- imagined trajectory generation;
- evidence that the symbolic state may be missing latent state/history.

Dreamer disagreement with symbolic models is evidence for model repair, not causal proof.

## 4. Mechanism library

The mechanism library is the long-term scientific memory for reusable mechanisms.

ECSA owns the **mechanism schema, admission rules, applicability semantics, and typed scientific relations**. It does not own a second artifact database or generic lineage engine. Persistence/versioned metadata/lineage are delegated through a `MechanismRepository` boundary; the first planned backend is ML Metadata (MLMD).

The authoritative contract is [MECHANISM_LIBRARY.md](MECHANISM_LIBRARY.md).

A mechanism record carries, at minimum:

- stable identity and immutable version;
- mechanism kind;
- epistemic status;
- executable/declarative representation artifact reference;
- explicit scope/context/regime constraints;
- transfer status;
- assumptions;
- parameters/specializations;
- supporting and contradicting evidence;
- provenance;
- typed relations such as `ABSTRACTED_FROM`, `SPECIALIZES`, `SUPERSEDES`, and `COMPOSED_OF`.

Initial epistemic statuses are:

- `CANDIDATE`;
- `QUALIFIED`;
- `ADMITTED`;
- `DEPRECATED`.

Initial transfer semantics are:

- `SHARED`;
- `CONTEXT_SPECIALIZED`;
- `SPLIT`.

Compression, synthesis, or repeated occurrence is not sufficient for admission.

Examples:

- a DreamCoder program must pass prospective validation before qualification/admission;
- a Stitch abstraction must be supported by qualified source mechanisms and transfer evidence before being treated as reusable across contexts;
- a TTT state feature requires a state-aliasing witness;
- a causal mechanism preserves the inference/identification semantics of its causal backend.

Applicability is explicit. An admitted mechanism is returned to a runtime consumer only if its scope and assumptions match the current context/regime. Unknown applicability requests more evidence rather than silently assuming transfer.

Individual engines receive **runtime projections** of applicable mechanisms. For example, DreamCoder Grammar is a projection of applicable program mechanisms, not the source of truth for long-term scientific memory.

The ECSA core depends on a small repository protocol:

```python
class MechanismRepository(Protocol):
    def admit(...): ...
    def get(...): ...
    def find_applicable(...): ...
    def lineage(...): ...
```

MLMD-specific protobuf types, IDs, and storage configuration must remain behind the concrete repository adapter.

## 5. Runtime flow

Normal cycle:

```text
Goal / current context
        |
Scientific kernel
  posterior over theories
        |
candidate experiments/actions
        |
specialized proposal engines
        |
information-directed selection
        |
Environment
        |
Observation
        |
BCS / model-specific likelihoods
        |
posterior update
        |
continue / plan / detect anomaly
```

Repair cycle:

```text
Population anomaly
        |
diagnostic classification
        |
+-- TTT / memory-state repair
+-- MMAE/IMM / parameter-mode alternatives
+-- BOCPD / regime change
+-- Fault Diagnosis / measurement or component fault
+-- CausalExploration / structural repair
+-- DreamCoder / new executable mechanism
+-- Stitch / reusable abstraction proposal
+-- VARIO / shared-vs-specialized scope
        |
candidate repaired/new theories
        |
small prior mass + explicit provenance
        |
scientific kernel
```

Planning cycle:

```text
Goal
  |
surviving theories
  |
BAMCP/pymdp if model uncertainty is decision-relevant
  |
Fast Downward / symbolic planner for grounded long-horizon structure
  |
Dreamer rollout when approximate latent dynamics are useful
  |
action or experiment
```

## 6. Non-goals

Architecture v1 explicitly avoids:

- implementing a second copy of BCS causal inference;
- replacing every specialized method with one universal planner;
- accepting LLM-generated explanations without empirical checks;
- treating program compression as proof of causality;
- silently resetting the theory population when all theories fail;
- forcing TTT, DreamCoder, BCS, PDDL, and Dreamer to share one internal state representation.

Their common interface is the prospective experiment/prediction/observation boundary.

## 7. Research question

The primary integration question is:

> Can an expanding Bayesian theory space, explicit causal inference, specialized model-repair engines, and a reusable mechanism library jointly improve causal accuracy, transfer, and goal-directed sample efficiency compared with their individual components?

That question must be answered experimentally rather than assumed from architecture alone.
