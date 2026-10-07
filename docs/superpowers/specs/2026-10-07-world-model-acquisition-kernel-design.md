# World Model Acquisition Kernel Design

Date: 2026-10-07  
Status: proposed architecture for Arena B / autonomous world-contract discovery

## 1. Purpose

ECSA is intended to enter an initially unknown environment and discover enough
of that environment's operative structure to pursue a goal without requiring a
human-authored semantic adapter for every benchmark or world.

The current DiscoveryWorld autonomous prototype still violates that objective.
Its adapter assigns meanings such as observation, acquisition, binary probe,
placement, or state change to named environment actions. The policy therefore
receives a partially interpreted world rather than learning the world's
interaction contract itself.

This design introduces a new ECSA subsystem:

`WorldModelAcquisitionKernel`

Its responsibility is to turn raw interaction traces into explicit,
uncertain, testable hypotheses about:

- persistent entities and their identities;
- latent entity types;
- observable features and predicates;
- relations between entities;
- action schemas and argument roles;
- action preconditions;
- action effects;
- affordances;
- numeric state variables;
- reusable structural world contracts.

The scientific kernel may then reason causally over the induced language rather
than over benchmark-specific concepts supplied by an adapter.

The intended autonomous loop is:

```text
unknown environment
        |
raw observation + raw action schemas
        |
interaction transition history
        |
WorldModelAcquisitionKernel
        |
candidate world contracts
        |
ECSA scientific kernel
        |
experiment selection
        |
ground action
        |
unknown environment
```

No LLM is required by this architecture.

## 2. Relationship to the existing DiscoveryWorld arenas

The existing
`2026-10-06-discoveryworld-reactor-transfer-arena-design.md` remains useful
as **Arena A: structured transfer**.

Arena A answers:

> Given benchmark-specific public evidence already converted into structured
> scientific records, can ECSA qualify, store, retrieve, validate, and reuse
> mechanisms across contexts?

Arena B introduced by this design answers the stronger question:

> Given only the public interaction surface of an unfamiliar world, can ECSA
> induce the world contract required to discover and test scientific
> mechanisms without benchmark-specific action semantics?

Arena A remains a regression baseline and ablation. Arena B must not depend on
Arena A's scenario-specific scientific sidecar.

## 3. Architectural principles

### 3.1 The environment adapter is transport, not interpretation

A raw environment adapter may expose information the environment itself makes
public, including stable object identifiers, names, descriptions, action
symbols, action argument names, text, numbers, success/failure results, and
terminal state.

It must not assign semantic roles such as:

- measurement;
- instrument;
- sample;
- controller;
- container;
- acquire;
- observe;
- probe;
- place;
- activate;
- causal variable.

An upstream action named `USE` and an action named `ZQ4` must be treated
identically if their observed transition structure is isomorphic.

### 3.2 Names are evidence, not ontology

Public strings may be retained as observations because they can carry useful
information. They are never authoritative type declarations or action roles.

For example, an entity named `thermometer` is not typed as a measurement
device merely because of its name. Its role must be supported by observed
interaction behavior.

### 3.3 Stable upstream IDs are observations, not a core dependency

When a world exposes stable object IDs, the structured perception frontend may
use them as identity evidence.

The downstream contracts must not require such IDs. A perceptual frontend using
object-like slots must be able to emit the same downstream entity-observation
contract without any upstream UUID.

### 3.4 World contracts are hypotheses

Induced predicates, types, affordances, action schemas, preconditions, and
effects are uncertain scientific objects.

Each world-contract hypothesis must retain:

- supporting evidence;
- contradicting evidence;
- source contexts;
- confidence or ranking information;
- unresolved alternatives;
- experiments capable of distinguishing alternatives.

A learned contract is never made true merely because one backend emitted it.

### 3.5 Backend algorithms are replaceable

MAcq, LOCM/LOCM2-style learners, a future trace-only lifted STRIPS learner,
symbolic program synthesis, and learned perception are backends behind ECSA
contracts.

Their internal representations must not leak into the scientific kernel.

## 4. Layered architecture

```text
                     UNKNOWN WORLD
                          |
                          v
                  RawEnvironmentPort
                          |
            +-------------+-------------+
            |                           |
     structured/API world        perceptual world
            |                           |
            |                    PerceptionFrontend
            |                           |
            |                 SlotAttentionFrontend
            |                    (planned backend)
            |                           |
            +-------------+-------------+
                          |
                          v
                Interaction Grounding
                          |
             entity identity hypotheses
             latent type hypotheses
             feature candidates
             transition deltas
             argument correspondences
                          |
                          v
             WorldModelAcquisitionKernel
                          |
          +---------------+----------------+
          |               |                |
     MAcq adapter      LOCM-style      trace-only lifted
                                      learner (future)
          |               |                |
          +---------------+----------------+
                          |
                          v
               WorldContractHypotheses
                          |
        predicates / relations / types / fluents
        action schemas / preconditions / effects
        affordances / numeric observables
                          |
                          v
                 ECSA Science Kernel
                          |
        BCS / TTT / BOCPD / repair / synthesis
        causal hypotheses / mechanism memory
        information-directed experiment selection
                          |
                          v
                  ContractExperiment
                          |
                          +-----------> UNKNOWN WORLD
```

## 5. Core raw interaction contracts

These contracts belong to ECSA core, not to a benchmark package.

### 5.1 Raw observation

Conceptually:

```python
RawObservation(
    observation_id,
    timestamp_or_step,
    payload,
)
```

The payload is an immutable, lossless representation of agent-visible
environment data.

Core acquisition algorithms must not depend directly on benchmark-specific
payload fields.

### 5.2 Raw action schema

```python
RawActionSchema(
    schema_id,
    parameter_names,
    public_metadata,
)
```

This says only that an environment exposes an action symbol with a particular
argument surface.

It does not say what the action means.

### 5.3 Ground action

```python
GroundAction(
    schema_id,
    arguments,
)
```

Arguments may refer to entity observations, strings, numbers, coordinates, or
other public values supported by the environment.

### 5.4 Raw outcome

```python
RawActionOutcome(
    success,
    payload,
)
```

A failed interaction is evidence. Failure must not be silently discarded.

### 5.5 Interaction transition

```python
InteractionTransition(
    before,
    action,
    outcome,
    after,
)
```

This is the foundational observation consumed by world-model acquisition.

It must remain valid whether the action is called `PICKUP`, `USE`, `A17`,
or any other symbol.

## 6. Perception and grounding boundary

The acquisition kernel operates on entity and feature observations, but it
must not assume how those entities were obtained.

Define:

```python
class PerceptionFrontend(Protocol):
    def perceive(
        self,
        raw_observation: RawObservation,
    ) -> PerceptualObservation: ...
```

`PerceptualObservation` contains candidate entities, candidate features, and
provenance back to the raw observation.

### 6.1 StructuredObservationFrontend — v1

Used when a world already provides agent-visible structured objects.

It may preserve:

- public object ID;
- public name;
- public description;
- public position;
- public membership in visible/inventory-like sets;
- arbitrary public scalar/text attributes.

It must not assign semantic types.

### 6.2 SlotAttentionFrontend — planned perception backend

Slot Attention is explicitly reserved as the classical object-centric
perception option for environments in which ECSA receives low-level visual
features instead of public object IDs.

The planned backend consumes perceptual feature maps and produces exchangeable
object-like slots:

```python
PerceptualSlot(
    local_slot_id,
    embedding,
    spatial_support,
    confidence,
    provenance,
)
```

Important invariant:

> A slot is not yet an entity.

Slot Attention proposes object-like representations. A separate grounding and
tracking process must decide whether slots across time correspond to the same
persistent entity, different entities, or changing parts of a larger entity.

The architecture must therefore keep these responsibilities distinct:

```text
visual features
      |
Slot Attention
      |
object-like slots
      |
EntityIdentityTracker
      |
EntityObservation
```

The first implementation does not train or require Slot Attention. The
interface is specified now so that ECSA's downstream world-contract
representation cannot accidentally depend on UUIDs.

The Slot Attention implementation should live behind an optional dependency
extra and must not add PyTorch or vision dependencies to the ECSA core install.

### 6.3 Entity identity hypotheses

Grounding maintains hypotheses such as:

```text
slot/object observation O_t and O_t+1 are the same entity
slot/object observation O_t and O_t+1 are different entities
O_t is a component of entity E
```

Stable upstream IDs may provide strong evidence for identity, but the contract
must support inferred identity when those IDs do not exist.

## 7. Interaction grounding

`InteractionGrounder` converts transitions into representation-neutral
evidence.

It owns:

- observation-delta extraction;
- candidate entity correspondences;
- stable versus changing feature detection;
- numeric/text/categorical feature candidates;
- action argument participation;
- successful and failed argument combinations;
- candidate state predicates;
- candidate relational predicates.

It must not label an action as an instrument action, movement action, pickup,
placement, dialog, or control operation.

### 7.1 Feature induction

Raw numbers and text-derived values begin as anonymous observable features:

```text
feature F17(entity E4) = 39.36
feature F29(entity E9) = 3880.56
```

Their semantic meaning is optional and secondary.

Scientific reasoning may use them before a human-readable ontology name has
been learned.

### 7.2 Argument-role induction

For an action `A(x, y)`, the kernel may maintain competing hypotheses:

```text
H1: x determines whether the action succeeds
H2: y determines whether the action succeeds
H3: the ordered pair (x, y) determines success
H4: the pair is order-invariant
H5: one argument selects the affected entity
H6: both arguments participate symmetrically
```

Observed transitions and targeted experiments distinguish these hypotheses.

No argument position is globally assumed to be the subject, tool, target, or
container.

## 8. World contract representation

A `WorldContractHypothesis` is a structured candidate language for the
environment.

It may contain:

### 8.1 Entity type hypotheses

Latent type assignments inferred from shared affordances, transition roles,
feature structure, or learned perceptual representations.

### 8.2 Predicate hypotheses

Boolean state features inferred from recurring transition patterns.

Examples are represented anonymously until semantics are learned:

```text
P3(E)
R8(agent, E)
R11(E1, E2)
```

### 8.3 Numeric fluent hypotheses

Numeric variables associated with entities, relations, or global state.

### 8.4 Action schema hypotheses

Conceptually:

```python
ActionSchemaHypothesis(
    schema_id,
    parameter_roles,
    precondition_hypotheses,
    effect_hypotheses,
    numeric_effect_hypotheses,
    support,
    contradictions,
)
```

### 8.5 Affordance hypotheses

An affordance is induced from evidence that some action schema is available,
successful, informative, or state-changing for a class of arguments.

Affordances are learned properties of the world contract, not adapter metadata.

## 9. WorldModelAcquisitionKernel

The kernel owns the population of competing world contracts.

Conceptually:

```python
class WorldModelAcquisitionKernel:
    def observe(
        self,
        transition: InteractionTransition,
    ) -> WorldModelUpdate: ...

    def propose_experiments(
        self,
        goal_context,
        budget,
    ) -> tuple[ContractExperiment, ...]: ...

    def contract_hypotheses(
        self,
    ) -> tuple[WorldContractHypothesis, ...]: ...
```

A `WorldModelUpdate` may include:

- new entity identity candidates;
- new latent types;
- new predicates;
- new action schema hypotheses;
- invalidated hypotheses;
- uncertainty changes;
- requests for additional experiments;
- proposals from backend action-model learners.

## 10. Action-model learner protocol

Planning-model libraries are backends behind a narrow protocol:

```python
class ActionModelLearner(Protocol):
    def update(
        self,
        traces: tuple[AcquisitionTrace, ...],
        current_contracts: tuple[WorldContractHypothesis, ...],
    ) -> ActionModelProposal: ...
```

### 10.1 MAcq backend — first external backend

MAcq is used as an optional planning-model acquisition backend where the
available trace representation is compatible with its extractors.

ECSA remains responsible for:

- trace construction;
- uncertainty and provenance;
- deciding when a MAcq proposal is admissible;
- translating proposals into ECSA world-contract hypotheses;
- testing them prospectively.

No MAcq class is allowed to appear in public ECSA core contracts.

### 10.2 LOCM / LOCM2-style backend

A LOCM-style learner is important because it can infer object-centered state
machines from action sequences without requiring full state observations.

LOCM2-style multiple state machines per object should be representable by the
ECSA learner adapter.

The initial implementation may use MAcq's existing extraction support where
appropriate or implement a dedicated adapter if needed. The architectural
contract does not depend on which library provides the algorithm.

### 10.3 Trace-only lifted predicate learner — planned

The design reserves a backend for the 2025 trace-only lifted STRIPS approach
that jointly discovers hidden predicates and action patterns from action
traces.

This is important because it reduces another human-authored assumption:
predicates and their arities need not be supplied in advance.

The first implementation only defines the backend boundary and fixtures needed
to add this learner later.

### 10.4 Noisy/probabilistic action-model learning — future option

World-contract evidence may be partial or noisy. The learner protocol must
therefore allow probabilistic proposals and posterior confidence rather than
requiring deterministic traces.

## 11. Numeric and continuous branch

Planning-style symbolic acquisition is insufficient for scientific worlds.

The kernel must preserve numeric observations independently of predicate
learning.

Conceptually:

```text
interaction grounding
        |
        +---- discrete predicate candidates
        |
        +---- numeric feature candidates
        |
        +---- text/categorical feature candidates
```

The scientific kernel may then use:

- symbolic regression;
- DreamCoder/program synthesis;
- BCS;
- probabilistic model comparison;
- other repair engines;

to discover relationships among numeric features and interventions.

The world-model kernel identifies the variables and action structure. It does
not replace ECSA's causal-scientific reasoning.

## 12. Active world-contract experimentation

Passive trace learning is not enough.

The acquisition kernel must expose uncertainty that the scientific coordinator
can use to choose experiments.

Example:

```text
C1: action A's first argument controls success
C2: action A's second argument controls success
C3: success depends on the pair relation
```

A useful experiment is an action/context choice whose possible outcomes most
strongly distinguish C1, C2, and C3.

Selection may use existing ECSA machinery for information-directed
experimentation. The contract-acquisition layer therefore becomes another
source of hypotheses and candidate experiments rather than a separate scripted
exploration policy.

This replaces hand-authored heuristics such as:

- try every action once;
- try every pair of UUIDs;
- visit every named location;
- treat specific actions as probes.

Those may remain debugging baselines only.

### 12.1 Future Bayesian OED backend — Pyro

The default scheduler uses ECSA's existing `ScienceKernel` for exact
information gain over discrete competing world contracts and a lightweight
structural-uncertainty prior while the contract population is not yet
informative.

Reserve Pyro OED as an optional future backend for the stage where ECSA has a
probabilistic generative world model of the form:

```text
p(observation | latent mechanism, experiment design)
```

A future `PyroOEDBackend` may estimate expected information gain for
continuous, noisy, or high-dimensional experiment designs. It must remain
behind the world-model experiment-selection boundary and must not replace the
core `ScienceKernel` contract.

Pyro/PyTorch are not dependencies of World Model Acquisition v1 and must not be
added to the default ECSA installation. If implemented, they belong to a
separate optional dependency extra.

## 13. Integration with existing ECSA subsystems

### 13.1 TTT

TTT may distinguish latent behavioral states and identify traces that current
state abstractions fail to separate.

### 13.2 BCS

BCS operates after or alongside variable induction to form predictive and
interventional causal states.

### 13.3 BOCPD

BOCPD detects changes in regime that may invalidate previously learned action
models or predicates.

### 13.4 MMAE/IMM and model populations

Competing dynamics or world-contract hypotheses may coexist instead of being
collapsed prematurely.

### 13.5 DreamCoder and program synthesis

Program synthesis operates on induced variables and actions rather than on
benchmark-specific fields.

### 13.6 MechanismRepository

Validated causal mechanisms remain distinct from world-contract hypotheses.

A world contract describes how an environment can be represented and
interacted with. A mechanism describes a validated relationship within that
representation.

Both may be transferred, but with separate provenance and validation.

## 14. World-contract transfer

A later environment or context may retrieve a prior world contract only as a
candidate.

Transfer may preserve structural abstractions such as:

```text
binary action exposes stable scalar feature of one participant
state-changing action has reversible numeric parameter
persistent relation to agent enables cross-location manipulation
```

It must not assume that source action names, entity labels, IDs, or numeric
feature identities are unchanged.

Canonicalization must support structural comparison under renaming.

## 15. Structural canonicalization

The system must be able to compare contracts across arbitrary renaming.

Acceptance fixture:

```text
World A
  PICKUP(x)
  USE(x, y)

World B
  A17(p)
  ZQ4(p, q)
```

If the observed transition structures are isomorphic, the induced canonical
contracts should be structurally equivalent even though all symbols differ.

This is a stronger test than checking that the code contains no benchmark
keywords.

## 16. DiscoveryWorld Arena B

Arena B uses a `DiscoveryWorldRawAdapter` whose autonomous responsibilities
are limited to:

- return agent-visible raw observation;
- return raw public action schemas;
- execute a ground raw action;
- return public success/failure/outcome;
- report terminal state;
- count steps.

The autonomous path must not depend on:

- `_ROLE_MAP`;
- `ActionRole.OBSERVE_UNARY`;
- `ActionRole.ACQUIRE`;
- `ActionRole.PROBE_BINARY`;
- `ActionRole.PLACE`;
- `MeasurementKind`;
- `ReactorMeasurement`;
- `ReactorMechanismHypothesis`;
- `ReactorLabScientificSidecar`.

Those may remain available to Arena A.

The autonomous loop becomes conceptually:

```python
while not environment.done:
    before = environment.observe_raw()
    actions = environment.list_raw_actions()

    experiment = ecsa.choose_experiment(
        raw_observation=before,
        raw_actions=actions,
        world_contract_hypotheses=world_model.contract_hypotheses(),
        theory_hypotheses=science.theories(),
        goal=goal,
    )

    outcome = environment.execute(experiment.action)
    after = environment.observe_raw()

    transition = InteractionTransition(
        before=before,
        action=experiment.action,
        outcome=outcome,
        after=after,
    )

    world_model.observe(transition)
    science.observe(transition, world_model.contract_hypotheses())
```

## 17. Migration from the current autonomous prototype

The existing `src/ecsa/autonomy.py` is treated as an experimental prototype.

The migration should progressively split it into focused modules:

```text
src/ecsa/world_model/
    contracts.py
    grounding.py
    kernel.py
    experiments.py

    learners/
        base.py
        macq.py
        locm.py
        trace_only.py        # boundary first, backend later

    perception/
        base.py
        structured.py
        slot_attention.py    # interface/stub first

src/ecsa/autonomy/
    scientist.py
```

The current DiscoveryWorld autonomous policy remains only until Arena B can run
through induced contracts.

No big-bang deletion is required.

## 18. Dependency policy

Core ECSA world-model contracts have no heavy runtime dependencies.

Optional extras may include:

```text
world-model-planning   # MAcq and planning dependencies
perception-slot        # future Slot Attention / PyTorch stack
```

Exact package pins belong in the implementation plan after compatibility is
verified.

The design does not require an LLM dependency.

## 19. Testing strategy

### 19.1 Contract tests

Verify immutable construction and serialization of:

- raw observations;
- raw action schemas;
- ground actions;
- outcomes;
- transitions;
- entity observations;
- world-contract hypotheses;
- action-model proposals.

### 19.2 Renaming invariance

Two isomorphic synthetic worlds with unrelated action/entity names must induce
equivalent canonical contracts.

### 19.3 Argument-order tests

Binary-action evidence must not assume that argument 0 or argument 1 has a
predefined semantic role.

### 19.4 Failure-as-evidence tests

Failed actions must update precondition/affordance hypotheses.

### 19.5 Partial-observation tests

The kernel must preserve alternative contracts when observations do not
uniquely identify the true model.

### 19.6 Backend conformance

MAcq and LOCM-style adapters must translate into the same ECSA proposal
contracts and must not leak backend-specific types.

### 19.7 Perception frontend equivalence

A structured frontend and a synthetic slot-based frontend describing the same
toy world must produce compatible downstream entity-observation structures.

This test exists before the real Slot Attention model is implemented.

### 19.8 DiscoveryWorld autonomous boundary

The Arena B production path must fail a static test if it references
benchmark-semantic concepts such as:

```text
Reactor
Crystal
Densitometer
MeasurementKind
PROBE_BINARY
ACQUIRE
OBSERVE_UNARY
PLACE
_ROLE_MAP
```

The test should focus on the autonomous production path, not Arena A.

## 20. Acceptance criteria for World Model Acquisition v1

Version 1 is complete when all of the following are true:

1. ECSA can record raw interaction transitions without benchmark semantics.
2. A structured observation frontend produces entity candidates without
   assigning domain roles.
3. Stable public IDs are optional in downstream contracts.
4. The kernel induces at least latent entity-type and action-schema
   hypotheses from generic traces.
5. Failed and successful interactions both affect contract hypotheses.
6. A learner protocol supports at least one real action-model acquisition
   backend.
7. MAcq is integrated behind that protocol without leaking MAcq types.
8. A LOCM/LOCM2-style learning path is available through the same boundary.
9. Numeric observable features survive independently of symbolic predicates.
10. Contract uncertainty can generate distinguishing experiment candidates.
11. ECSA's experiment selector can choose among contract-learning and
    scientific experiments through one coordination boundary.
12. DiscoveryWorld Arena B no longer requires `_ROLE_MAP` or semantic action
    categories.
13. Arena B does not consume the Reactor Lab scientific sidecar.
14. Contract canonicalization is invariant to arbitrary symbol renaming on
    synthetic isomorphic worlds.
15. The perception boundary includes a Slot Attention backend contract.
16. Slot Attention is not required for structured-world v1 execution.
17. No PyTorch/vision dependency is introduced into the default ECSA install.
18. Arena A continues to pass as the structured-transfer regression baseline.

## 21. Non-goals for v1

Version 1 does not require:

- end-to-end learning directly from pixels;
- training a Slot Attention model;
- solving entity tracking from arbitrary video;
- natural-language ontology naming;
- a universal PDDL exporter;
- replacing all existing ECSA theory/repair engines;
- implementing every published action-model learner;
- claiming that a learned contract is causally correct without prospective
  validation;
- removing Arena A;
- using an LLM to infer action semantics.

## 22. Planned follow-up: Slot Attention perception

After World Model Acquisition v1 is stable on structured public observations,
the next perception milestone may implement `SlotAttentionFrontend`.

That milestone should test:

1. feature-map to slot extraction;
2. slot permutation invariance;
3. temporal entity correspondence across frames;
4. object appearance/disappearance;
5. split/merge ambiguity represented as competing identity hypotheses;
6. equivalence of downstream world-contract learning between structured
   entity observations and slot-derived entity observations on matched toy
   environments.

Slot Attention remains one backend. The architecture must also permit future
object detectors, segmentation models, or multimodal learned frontends.

## 23. References and implementation prior art

- MAcq / Model Acquisition Toolkit:
  https://github.com/AI-Planning/macq
- LOCM2, Cresswell and Gregory, ICAPS 2011:
  https://ojs.aaai.org/index.php/ICAPS/article/view/13476
- Learning Lifted STRIPS Models from Action Traces Alone,
  Gösgens, Jansen, and Geffner, ICAPS 2025:
  https://ojs.aaai.org/index.php/ICAPS/article/view/36117
- Object-Centric Learning with Slot Attention,
  Locatello et al., 2020:
  https://arxiv.org/abs/2006.15055

These references are implementation prior art. Their native representations do
not define ECSA's public contracts.
