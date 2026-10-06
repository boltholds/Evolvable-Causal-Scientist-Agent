# Mechanism Library Contract v1

## Purpose

The ECSA mechanism library is the long-term scientific memory for reusable mechanisms.

It is intentionally **not** a second artifact database, model registry, lineage engine, DreamCoder grammar, or causal inference engine. ECSA owns the semantic contract for what a mechanism means, when it may be admitted, and where it is applicable. Persistence, versioned metadata storage, and workflow lineage are delegated to ML Metadata (MLMD) through a repository adapter.

The first persistence implementation is now executable:

```text
MechanismRepository
        |
        +-- MLMDMechanismRepository
              |
              +-- MLMD 1.21
              +-- SQLite reference backend
```

`MLMDMechanismRepository` persists immutable mechanism versions, admission/deprecation executions, exact external input artifacts, scope contexts, and typed mechanism-relation lineage without exposing MLMD protobuf types through the core repository protocol.

The ECSA core must depend on the `MechanismRepository` protocol rather than on MLMD runtime types.

## 1. Mechanism record

A stored mechanism is immutable with respect to identity/version. New evidence may create a new version, specialization, superseding mechanism, or status transition record; historical evidence and lineage are not erased.

Conceptual schema:

```text
MechanismRecord
├── mechanism_id
├── version
├── kind
├── epistemic_status
├── representation_artifact
├── scope
├── transfer
├── parameters / specializations
├── assumptions
├── supporting_evidence[]
├── contradicting_evidence[]
├── provenance
└── relations[]
```

### Identity and version

- `mechanism_id` is the stable identity of the scientific mechanism.
- `version` identifies an immutable revision of that mechanism record.
- A changed representation, scope, assumptions, or parameterization creates a new version or specialization rather than mutating scientific history in place.
- Content-addressed upstream artifacts remain referenced by their original artifact IDs.

### Mechanism kind

Initial kinds:

- `PROGRAM` — executable program mechanism, such as a qualified DreamCoder program or validated Stitch-derived abstraction;
- `CAUSAL_MODEL` — causal mechanism/model fragment with an external causal inference adapter, such as a BCS-supported artifact;
- `STATE_FEATURE` — reusable state/memory augmentation, such as a qualified TTT residual-state feature;
- `SYMBOLIC_RULE` — declarative symbolic mechanism/operator suitable for symbolic reasoning or planning.

The list may grow, but arbitrary backend-specific classes must not leak into the core contract.

### Epistemic status

Initial statuses:

- `CANDIDATE` — proposed but not yet prospectively validated;
- `QUALIFIED` — passed the backend-specific qualification gate;
- `ADMITTED` — accepted into the reusable mechanism library for at least one explicit scope;
- `DEPRECATED` — retained for lineage/history but excluded from ordinary applicability queries.

Deprecation does not delete the mechanism or its evidence.

## 2. Representation

`representation_artifact` is a reference to the executable or declarative artifact, not the artifact bytes themselves.

Examples:

- DreamCoder program artifact;
- Stitch abstraction artifact;
- BCS model artifact;
- TTT state-projection artifact;
- symbolic rule/PDDL fragment.

The mechanism library does not impose one internal representation on all engines.

Runtime consumers use representation-specific adapters:

```text
MechanismRecord
      |
      +-- DreamCoder projection -> Grammar primitive/invention
      +-- BCS projection        -> causal theory/model
      +-- TTT projection        -> state feature / behavioral predictor
      +-- Planner projection    -> symbolic operator/rule
```

## 3. Scope and applicability

A mechanism is never globally applicable by default.

A scope may constrain:

- contexts;
- regimes;
- environmental/task domains;
- required observables or variables;
- assumptions;
- representation/runtime capabilities.

The transfer decision is stored separately from the representation.

Initial transfer statuses:

- `SHARED` — current evidence supports the same mechanism across the validated contexts;
- `CONTEXT_SPECIALIZED` — a shared form exists but applicability/parameters split into context groups;
- `SPLIT` — current evidence supports distinct mechanisms rather than one reusable cross-context mechanism.

These are ECSA semantics. For VARIO-backed transfer validation they are projected from the discovered VARIO context partition.

### Applicability rule

`find_applicable(context)` may return an admitted mechanism only when all required scope constraints and assumptions are satisfied.

A mechanism with `SPLIT` transfer status cannot be returned as a global shared mechanism. Its specialized child records may be returned for their matching contexts.

Unknown context membership is not evidence of applicability. The caller may instead request a transfer/validation experiment.

## 4. Evidence

Scientific evidence is append-only in meaning.

A mechanism record may reference:

- supporting prospective evidence;
- contradicting prospective evidence;
- synthesis/training evidence;
- transfer/invariance evidence;
- anomaly/repair evidence.

Training/synthesis evidence must remain distinguishable from prospective validation evidence.

Contradicting evidence does not silently delete or overwrite supporting evidence. It may cause:

- posterior re-evaluation;
- scope narrowing;
- specialization;
- a new version;
- supersession;
- deprecation.

## 5. Provenance

A mechanism must be traceable to the exact artifacts/executions that produced and validated it.

Relevant provenance may include:

- BCS inference/model artifacts;
- TTT learned machine and state-projection artifacts;
- DreamCoder synthesis artifact;
- prospective program validation;
- Stitch abstraction artifact;
- VARIO transfer artifact;
- future fault/causal-structure/planning engines.

The repository must be able to answer:

- Which observations and interventions support this mechanism?
- Which synthesis/repair execution created it?
- Which qualified mechanisms were abstracted into it?
- Which transfer decision established its scope?
- Which later mechanism specializes or supersedes it?

## 6. Relations

Initial typed relations:

- `ABSTRACTED_FROM` — reusable abstraction learned from mechanisms;
- `SPECIALIZES` — narrower-scope/context-specific mechanism derived from a parent;
- `SUPERSEDES` — newer mechanism/version replaces an earlier one for an overlapping scope;
- `COMPOSED_OF` — mechanism is explicitly composed from other mechanisms.

Relations are scientific lineage, not only storage references.

## 7. Admission semantics

Admission is a scientific operation, not a persistence operation.

### Program mechanism

Minimum admission path:

```text
DreamCoder proposal
    |
prospective held-out validation
    |
QualifiedProgramMechanism
    |
(optional) Stitch abstraction
    |
(if cross-context reuse claimed) VARIO transfer validation
    |
Mechanism admission
```

A program that only fits synthesis/training examples is not admissible.

### Stitch abstraction

Compression alone is insufficient.

A Stitch abstraction may become a reusable mechanism only after:

1. its source mechanisms are already qualified;
2. the abstraction has multi-use/reuse evidence;
3. any claimed cross-context scope is supported by transfer validation;
4. the runtime representation has an executable/predictive adapter for the intended consumers.

### State feature

A TTT-derived state feature may be admitted only after the state-aliasing qualification gate establishes that the previous state representation collapsed behaviorally distinct residual states.

### Causal mechanism

A causal model/mechanism must preserve the identification/computation semantics of its causal backend. A computation failure is not evidence against the mechanism.

## 8. Admission decision

The library admission operation should conceptually consume:

```text
Qualified representation
        +
prospective validation
        +
scope / transfer evidence
        +
assumptions
        +
provenance
        |
        v
MechanismRecord(status=ADMITTED)
```

A proposed reusable mechanism with insufficient transfer evidence remains `QUALIFIED`, not `ADMITTED` for a global scope.

## 9. Repository boundary

The ECSA-owned persistence interface is deliberately small:

```python
class MechanismRepository(Protocol):
    def admit(...): ...
    def get(...): ...
    def find_applicable(...): ...
    def lineage(...): ...
```

Concrete persistence/backend concerns are outside the scientific core.

The first implementation is `MLMDMechanismRepository`, backed by MLMD 1.21 with SQLite in the current acceptance suite.

## 10. MLMD mapping

MLMD is used as persistence and lineage infrastructure, not as the scientific schema itself.

Planned mapping:

### Artifact types

- `Mechanism`
- `MechanismRepresentation`
- `Evidence`
- `DreamCoderProgram`
- `TTTStateFeature`
- `StitchAbstraction`
- `VarioTransferDecision`

### Execution types

- `MechanismSynthesis`
- `ProspectiveValidation`
- `StateQualification`
- `AbstractionLearning`
- `TransferValidation`
- `MechanismAdmission`
- `MechanismDeprecation`

### Context types

- `Environment`
- `Domain`
- `Task`
- `Regime`
- `ScientificRun`

MLMD Events record input/output artifact lineage. MLMD Contexts group executions and artifacts by environmental/scientific context.

The ECSA contract must remain usable with another metadata backend later; MLMD-specific IDs and protobuf classes must stay behind the adapter.

## 11. Runtime projections

**Status: first executable projection layer integrated.**

The mechanism library is the source of reusable scientific knowledge. Individual engines receive filtered projections.

```text
MechanismRepository
        |
current context / regime / assumptions
        |
find_applicable(...)
        |
        +-- DreamCoder grammar projection
        +-- BCS causal-model artifact view
        +-- symbolic-rule artifact view
        +-- state-feature artifact view
```

`MechanismRuntimeProjector` now provides typed artifact views for admitted
`CAUSAL_MODEL`, `STATE_FEATURE`, and `SYMBOLIC_RULE` mechanisms. These
views deliberately stop at artifact references until a backend-specific loader
is available; the projection layer does not invent a BCS model loader or PDDL
compiler.

`DreamCoderGrammarProjector` is executable. It:

1. queries only mechanisms applicable to the current context;
2. selects admitted `PROGRAM` mechanisms;
3. resolves a direct `dreamcoder-program` representation or a qualified
   `program-mechanism` projection back to its exact source DreamCoder
   artifact;
4. loads the stored program body;
5. passes those programs into DreamCoder as reusable library productions.

The DreamCoder worker wraps each reused program in the upstream
`Invented` representation and adds it to the real `Grammar`. Thus the
repository remains the source of truth while the Grammar is an execution-time
projection.

The acceptance scenario synthesizes and prospectively qualifies a Boolean
program in one context, admits it with `SHARED` scope covering a second
context, then projects it into DreamCoder in that second context. Warm
synthesis must enumerate strictly fewer programs than cold-start synthesis on
the same target.

This currently demonstrates **computational reuse** of admitted mechanisms. It
does not yet establish that reuse reduces the number of interventions required
in a new environment; transfer sample efficiency remains an arena-level
acceptance criterion.

## 12. Non-goals

Mechanism Library v1 does not:

- implement its own artifact database;
- implement its own generic lineage graph engine;
- replace DreamCoder Grammar;
- replace Stitch abstraction learning;
- replace VARIO transfer validation;
- replace BCS causal inference;
- treat compression as proof of causality;
- treat transfer to an unseen context as established without evidence;
- erase deprecated/contradicted mechanisms or their supporting history;
- require all mechanism kinds to share one executable representation.

## 13. First implementation status and acceptance criteria

**Status: implemented and covered by the integration suite.**

The first executable repository implementation can:

1. admit a qualified mechanism with explicit scope, assumptions, evidence, and provenance;
2. persist the record and its lineage through MLMD;
3. retrieve a mechanism by stable identity/version;
4. query only mechanisms applicable to a supplied context/regime/assumption set;
5. trace an admitted mechanism back through DreamCoder/Stitch/VARIO or TTT/BCS evidence where applicable;
6. create specialization/supersession relations without rewriting prior history;
7. exclude deprecated mechanisms from ordinary applicability queries while keeping them queryable for lineage;
8. project applicable program mechanisms into a DreamCoder grammar without making that grammar the source of truth.
