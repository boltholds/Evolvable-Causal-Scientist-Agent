# World Model Acquisition Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmark-neutral world-model acquisition layer that learns entity structure, action schemas, predicates, numeric observables, and affordances from raw interaction traces, then migrate DiscoveryWorld Arena B to that induced contract without semantic action-role mappings.

**Architecture:** Add a new `ecsa.world_model` package with immutable raw interaction contracts, perception/grounding, competing world-contract hypotheses, canonicalization, learner adapters, contract memory, and active contract experiments. MAcq 0.3.11 supplies the first external LOCM backend; LOCM2 is implemented behind the same learner protocol because MAcq does not provide LOCM2. DiscoveryWorld becomes a thin raw transport adapter; Arena A remains unchanged as the structured-transfer baseline.

**Tech Stack:** Python 3.12, stdlib dataclasses/enums/sqlite3/hashlib, pytest, existing ECSA `ScienceKernel`, optional `macq==0.3.11`, pinned DiscoveryWorld dependency, no LLM requirement, no PyTorch dependency in the default install.

**Spec:** `docs/superpowers/specs/2026-10-07-world-model-acquisition-kernel-design.md`

## Global Constraints

- Python remains `>=3.12`.
- Core `ecsa.world_model` contracts must have no heavy runtime dependencies.
- `macq==0.3.11` is optional under a `world-model-planning` extra; importing core world-model modules must not import MAcq.
- Slot Attention is a planned perception backend boundary only in v1; do not add PyTorch or vision dependencies to the default install.
- No LLM is required by the world-model acquisition path.
- Raw environment adapters expose public transport data only; they must not assign semantic roles such as measurement, instrument, acquire, observe, probe, place, controller, or container.
- Public names/descriptions are evidence, not authoritative ontology declarations.
- Stable upstream object IDs are optional downstream; identity must remain representable as a hypothesis.
- Successful and failed interactions are both evidence.
- Action argument positions have no predefined semantic role.
- World-contract hypotheses remain distinct from validated causal mechanisms in `MechanismRepository`.
- DiscoveryWorld Arena A and its Reactor-specific sidecar remain a regression/ablation path.
- DiscoveryWorld Arena B must not depend on `_ROLE_MAP`, `ActionRole.OBSERVE_UNARY`, `ActionRole.ACQUIRE`, `ActionRole.PROBE_BINARY`, `ActionRole.PLACE`, `MeasurementKind`, `ReactorMeasurement`, `ReactorMechanismHypothesis`, or `ReactorLabScientificSidecar`.

## Review Focus

1. **No stable upstream object ID:** two observations with ambiguous visual/entity correspondence must preserve multiple identity hypotheses instead of silently merging entities. Covered in Task 2.
2. **Action failure with no state delta:** failure must update affordance/precondition evidence while producing no positive effect claim. Covered in Task 3.
3. **Argument-order ambiguity:** a binary action whose observed behavior is symmetric or auto-swapped must not hard-code argument 0/1 roles. Covered in Tasks 3 and 4.
4. **Optional backend unavailable/incompatible:** absence of MAcq or a trace that LOCM cannot consume must return a typed learner failure without breaking the kernel. Covered in Task 7.
5. **Perception source changes:** structured entity observations and synthetic slot-derived observations of the same toy world must feed compatible downstream grounding/contracts. Covered in Task 2.

---

### Task 1: Core raw-interaction contracts

**Files:**
- Create: `src/ecsa/world_model/__init__.py`
- Create: `src/ecsa/world_model/contracts.py`
- Test: `tests/test_world_model_contracts.py`

**Interfaces:**
- Consumes: no new project interfaces.
- Produces:
  - `FrozenRawValue`
  - `freeze_raw_value(value: object) -> FrozenRawValue`
  - `RawObservation`
  - `RawActionParameter`
  - `RawActionSchema`
  - `GroundAction`
  - `RawActionOutcome`
  - `InteractionTransition`
  - `RawEnvironmentPort`

- [ ] **Step 1: Write failing immutability and validation tests**

Add tests asserting:

```python
def test_interaction_transition_is_immutable_and_lossless(): ...
def test_failed_action_is_a_valid_raw_outcome(): ...
def test_ground_action_rejects_wrong_arity(): ...
def test_raw_action_schema_carries_public_candidates_without_semantic_roles(): ...
```

The action-schema fixture must use meaningless symbols such as `A17` and
`arg0`; no test should require semantic action names.

- [ ] **Step 2: Run the contract tests and verify RED**

Run:

```bash
python -m pytest tests/test_world_model_contracts.py -q
```

Expected: collection/import failure because `ecsa.world_model.contracts` does
not exist.

- [ ] **Step 3: Implement the raw contracts**

Implement these exact public shapes in `contracts.py`:

```python
@dataclass(frozen=True)
class FrozenRawValue:
    value: RawScalar | tuple["FrozenRawValue", ...] | tuple[tuple[str, "FrozenRawValue"], ...]

def freeze_raw_value(value: object) -> FrozenRawValue: ...

@dataclass(frozen=True)
class RawObservation:
    observation_id: str
    step: int
    payload: FrozenRawValue

@dataclass(frozen=True)
class RawActionParameter:
    name: str
    public_candidates: tuple[FrozenRawValue, ...] = ()

@dataclass(frozen=True)
class RawActionSchema:
    schema_id: str
    parameters: tuple[RawActionParameter, ...]
    public_metadata: FrozenRawValue

@dataclass(frozen=True)
class GroundAction:
    schema_id: str
    arguments: tuple[FrozenRawValue, ...]

@dataclass(frozen=True)
class RawActionOutcome:
    success: bool
    payload: FrozenRawValue

@dataclass(frozen=True)
class InteractionTransition:
    transition_id: str
    before: RawObservation
    action: GroundAction
    outcome: RawActionOutcome
    after: RawObservation
```

Define `RawEnvironmentPort(Protocol)` with:

```python
def observe_raw(self) -> RawObservation: ...
def list_raw_actions(self) -> tuple[RawActionSchema, ...]: ...
def execute_raw_action(self, action: GroundAction) -> RawActionOutcome: ...
@property
def done(self) -> bool: ...
@property
def steps(self) -> int: ...
```

`GroundAction` arity validation is performed against a supplied
`RawActionSchema` helper, not by embedding semantic parameter types.

- [ ] **Step 4: Run Task 1 tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_world_model_contracts.py -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ecsa/world_model tests/test_world_model_contracts.py
git commit -m "feat: add raw world interaction contracts"
```

---

### Task 2: Perception frontend boundary and entity identity hypotheses

**Files:**
- Create: `src/ecsa/world_model/perception/__init__.py`
- Create: `src/ecsa/world_model/perception/base.py`
- Create: `src/ecsa/world_model/perception/structured.py`
- Create: `src/ecsa/world_model/perception/slot_attention.py`
- Test: `tests/test_world_model_perception.py`

**Interfaces:**
- Consumes: `RawObservation`, `FrozenRawValue`.
- Produces:
  - `ObservedFeature`
  - `EntityObservation`
  - `PerceptualObservation`
  - `PerceptualSlot`
  - `EntityIdentityRelation`
  - `EntityIdentityHypothesis`
  - `PerceptionFrontend`
  - `StructuredEntityDecoder`
  - `StructuredObservationFrontend`
  - `SlotEncoder`
  - `SlotAttentionFrontend`

- [ ] **Step 1: Write failing frontend-equivalence and ambiguous-identity tests**

Add:

```python
def test_structured_frontend_preserves_public_id_as_evidence_not_required_identity(): ...
def test_slot_frontend_accepts_permuted_slots(): ...
def test_structured_and_slot_frontends_emit_compatible_entity_observations(): ...
def test_missing_stable_id_preserves_competing_identity_hypotheses(): ...
```

The slot test uses a fake `SlotEncoder`; it must not import torch.

- [ ] **Step 2: Run Task 2 tests and verify RED**

Run:

```bash
python -m pytest tests/test_world_model_perception.py -q
```

Expected: missing perception modules/types.

- [ ] **Step 3: Implement the perception contracts and lightweight frontends**

In `base.py` define:

```python
@dataclass(frozen=True)
class ObservedFeature:
    feature_id: str
    value: FrozenRawValue
    provenance_id: str

@dataclass(frozen=True)
class EntityObservation:
    local_ref: str
    source_identity: str | None
    features: tuple[ObservedFeature, ...]
    provenance_id: str

@dataclass(frozen=True)
class PerceptualObservation:
    observation_id: str
    entities: tuple[EntityObservation, ...]
    global_features: tuple[ObservedFeature, ...]

class PerceptionFrontend(Protocol):
    def perceive(self, raw_observation: RawObservation) -> PerceptualObservation: ...
```

Add `EntityIdentityRelation` values `SAME`, `DIFFERENT`, `COMPONENT`,
and an immutable `EntityIdentityHypothesis` with evidence IDs and confidence.

`StructuredObservationFrontend` accepts a `StructuredEntityDecoder`
protocol supplied by the environment integration; it does not know
DiscoveryWorld field names.

`SlotAttentionFrontend` accepts an injected:

```python
class SlotEncoder(Protocol):
    def encode(self, raw_observation: RawObservation) -> tuple[PerceptualSlot, ...]: ...
```

It converts slots to `EntityObservation` candidates without claiming temporal
identity. No real neural model is implemented.

- [ ] **Step 4: Run Task 2 tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_world_model_perception.py -q
```

Expected: all Task 2 tests pass with no torch dependency installed.

- [ ] **Step 5: Commit**

```bash
git add src/ecsa/world_model/perception tests/test_world_model_perception.py
git commit -m "feat: add perception and entity identity boundary"
```

---

### Task 3: Interaction grounding and anonymous feature induction

**Files:**
- Create: `src/ecsa/world_model/grounding.py`
- Test: `tests/test_world_model_grounding.py`

**Interfaces:**
- Consumes: `InteractionTransition`, before/after `PerceptualObservation`.
- Produces:
  - `FeatureDelta`
  - `ActionParticipationEvidence`
  - `ActionOutcomeEvidence`
  - `GroundedTransition`
  - `GroundingUpdate`
  - `InteractionGrounder.observe(...) -> GroundingUpdate`

- [ ] **Step 1: Write failing grounding tests**

Add:

```python
def test_grounder_emits_anonymous_numeric_feature_delta(): ...
def test_failed_action_updates_outcome_evidence_without_positive_effect(): ...
def test_binary_action_preserves_both_argument_role_hypotheses(): ...
def test_unchanged_identifier_like_numbers_are_not_promoted_as_effects(): ...
```

The argument-order test must run the same transition with arguments reversed and
assert that both participants remain represented rather than choosing a
predefined target position.

- [ ] **Step 2: Run Task 3 tests and verify RED**

Run:

```bash
python -m pytest tests/test_world_model_grounding.py -q
```

Expected: missing grounding implementation.

- [ ] **Step 3: Implement `InteractionGrounder`**

Use exact public method:

```python
class InteractionGrounder:
    def observe(
        self,
        transition: InteractionTransition,
        before: PerceptualObservation,
        after: PerceptualObservation,
    ) -> GroundingUpdate: ...
```

Requirements:

- compare observed features by provenance/identity hypotheses;
- preserve anonymous numeric/text/categorical features;
- record action-argument participation independently of argument order;
- retain failed combinations as negative affordance/precondition evidence;
- do not emit domain-semantic labels.

- [ ] **Step 4: Run Task 3 tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_world_model_grounding.py -q
```

Expected: all Task 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ecsa/world_model/grounding.py tests/test_world_model_grounding.py
git commit -m "feat: ground raw interaction transitions"
```

---

### Task 4: World-contract hypothesis model and renaming-invariant canonicalization

**Files:**
- Create: `src/ecsa/world_model/hypotheses.py`
- Create: `src/ecsa/world_model/canonical.py`
- Test: `tests/test_world_contract_hypotheses.py`
- Test: `tests/test_world_contract_canonicalization.py`

**Interfaces:**
- Consumes: grounding evidence IDs and anonymous feature/entity references.
- Produces:
  - `HypothesisStatus`
  - `EntityTypeHypothesis`
  - `PredicateHypothesis`
  - `NumericFluentHypothesis`
  - `ArgumentRoleHypothesis`
  - `ActionSchemaHypothesis`
  - `AffordanceHypothesis`
  - `WorldContractHypothesis`
  - `CanonicalWorldContract`
  - `canonicalize_world_contract(contract: WorldContractHypothesis) -> CanonicalWorldContract`

- [ ] **Step 1: Write failing typed-hypothesis and isomorphism tests**

Add tests for validation/provenance plus:

```python
def test_isomorphic_worlds_with_renamed_symbols_have_same_canonical_contract(): ...
def test_non_isomorphic_action_structure_changes_canonical_contract(): ...
def test_argument_swap_is_not_collapsed_when_evidence_is_directional(): ...
def test_argument_swap_is_collapsed_when_contract_marks_relation_symmetric(): ...
```

World A uses names `PICKUP`, `USE`; World B uses unrelated symbols
`A17`, `ZQ4`.

- [ ] **Step 2: Run Task 4 tests and verify RED**

Run:

```bash
python -m pytest   tests/test_world_contract_hypotheses.py   tests/test_world_contract_canonicalization.py -q
```

Expected: missing hypothesis/canonical modules.

- [ ] **Step 3: Implement the immutable hypothesis graph**

Every hypothesis stores:

- stable hypothesis ID;
- supporting evidence IDs;
- contradicting evidence IDs;
- confidence in `[0, 1]`;
- explicit status;
- structural references to other hypothesis IDs.

No backend-specific MAcq/LOCM types are allowed.

- [ ] **Step 4: Implement structural canonicalization**

Use iterative structural signatures/hashing over type/predicate/action nodes,
parameter positions, precondition/effect edges, numeric-fluent edges, and
symmetry flags. Source symbol names and source entity IDs do not contribute to
the final canonical fingerprint.

Public API:

```python
@dataclass(frozen=True)
class CanonicalWorldContract:
    fingerprint: str
    node_signatures: tuple[str, ...]

def canonicalize_world_contract(
    contract: WorldContractHypothesis,
) -> CanonicalWorldContract: ...
```

- [ ] **Step 5: Run Task 4 tests and verify GREEN**

Run the two Task 4 test files. Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/ecsa/world_model/hypotheses.py src/ecsa/world_model/canonical.py   tests/test_world_contract_hypotheses.py tests/test_world_contract_canonicalization.py
git commit -m "feat: add canonical world contract hypotheses"
```

---

### Task 5: WorldModelAcquisitionKernel and persistent contract memory

**Files:**
- Create: `src/ecsa/world_model/memory.py`
- Create: `src/ecsa/world_model/kernel.py`
- Test: `tests/test_world_model_kernel.py`
- Test: `tests/test_world_contract_memory.py`

**Interfaces:**
- Consumes: `GroundingUpdate`, `WorldContractHypothesis`, learner proposals from Task 7.
- Produces:
  - `WorldModelUpdate`
  - `WorldContractRef`
  - `WorldContractCandidate`
  - `WorldContractStore`
  - `SQLiteWorldContractStore`
  - `WorldModelAcquisitionKernel.observe_grounding(...)`
  - `WorldModelAcquisitionKernel.ingest_proposals(...)`
  - `WorldModelAcquisitionKernel.contract_hypotheses()`

- [ ] **Step 1: Write failing population/update/memory tests**

Add:

```python
def test_kernel_preserves_competing_contracts_under_ambiguous_evidence(): ...
def test_contradicting_evidence_lowers_or_rejects_contract_without_mutating_history(): ...
def test_sqlite_store_round_trips_canonical_contract_and_provenance(): ...
def test_transfer_candidate_matches_structure_without_matching_source_names(): ...
```

- [ ] **Step 2: Run Task 5 tests and verify RED**

Run:

```bash
python -m pytest tests/test_world_model_kernel.py tests/test_world_contract_memory.py -q
```

Expected: missing kernel/memory modules.

- [ ] **Step 3: Implement `WorldContractStore` and SQLite persistence**

Protocol:

```python
class WorldContractStore(Protocol):
    def admit(self, contract: WorldContractHypothesis) -> WorldContractRef: ...
    def get(self, ref: WorldContractRef) -> WorldContractHypothesis: ...
    def find_transfer_candidates(
        self,
        canonical_context: CanonicalWorldContract,
    ) -> tuple[WorldContractCandidate, ...]: ...
```

Use stdlib `sqlite3`; do not reuse `MechanismRepository`.

- [ ] **Step 4: Implement kernel population management**

Public constructor:

```python
class WorldModelAcquisitionKernel:
    def __init__(
        self,
        *,
        store: WorldContractStore | None = None,
        max_contracts: int = 64,
    ) -> None: ...
```

The kernel records grounding evidence, keeps alternatives, ingests proposals,
and exposes immutable snapshots. It does not choose benchmark actions directly.

- [ ] **Step 5: Run Task 5 tests and verify GREEN**

Run both Task 5 test files. Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/ecsa/world_model/memory.py src/ecsa/world_model/kernel.py   tests/test_world_model_kernel.py tests/test_world_contract_memory.py
git commit -m "feat: add world model kernel and contract memory"
```

---

### Task 6: Contract experiments and ScienceKernel bridge

**Files:**
- Create: `src/ecsa/world_model/experiments.py`
- Modify: `src/ecsa/science.py`
- Test: `tests/test_world_model_experiments.py`

**Interfaces:**
- Consumes: contract population, raw action schemas, perceptual entities, existing `ScienceKernel`.
- Produces:
  - `ExperimentBudget`
  - `ContractExperiment`
  - `ContractOutcomePrediction`
  - `ContractExperimentCoordinator.propose(...)`
  - `ContractExperimentCoordinator.select(...)`

- [ ] **Step 1: Write failing discriminating-experiment tests**

Add:

```python
def test_contract_experiment_distinguishes_argument_role_hypotheses(): ...
def test_failed_action_outcome_can_discriminate_precondition_hypotheses(): ...
def test_contract_and_scientific_candidates_use_same_information_gain_selector(): ...
def test_budget_limits_ground_action_enumeration_without_semantic_action_roles(): ...
```

- [ ] **Step 2: Run Task 6 tests and verify RED**

Run:

```bash
python -m pytest tests/test_world_model_experiments.py -q
```

Expected: missing experiment coordinator.

- [ ] **Step 3: Implement generic ground-action candidate enumeration**

Enumerate candidates from:

- raw action schemas;
- public parameter candidates when provided;
- current entity observations for otherwise untyped argument slots;
- bounded Cartesian products controlled by `ExperimentBudget`.

No action-name special cases are allowed.

- [ ] **Step 4: Bridge contract hypotheses to existing Bayesian experiment scoring**

Add only the minimal reusable hook to `ScienceKernel` if required; prefer
adapting contract predictions to existing `TheoryPosterior`,
`PredictiveDistribution`, and `select_experiment()` rather than creating a
second information-gain implementation.

- [ ] **Step 5: Run Task 6 tests and verify GREEN**

Run Task 6 tests plus:

```bash
python -m pytest tests/test_science.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/ecsa/world_model/experiments.py src/ecsa/science.py   tests/test_world_model_experiments.py
git commit -m "feat: select active world contract experiments"
```

---

### Task 7: ActionModelLearner protocol and MAcq LOCM backend

**Files:**
- Create: `src/ecsa/world_model/learners/__init__.py`
- Create: `src/ecsa/world_model/learners/base.py`
- Create: `src/ecsa/world_model/learners/macq.py`
- Create: `src/ecsa/world_model/learners/trace_only.py`
- Modify: `pyproject.toml`
- Test: `tests/test_world_model_learners.py`
- Test: `tests/test_world_model_macq.py`

**Interfaces:**
- Consumes: grounded action sequences and contract snapshots.
- Produces:
  - `AcquisitionTrace`
  - `ActionModelProposal`
  - `LearnerFailure`
  - `ActionModelLearner`
  - `MacqLocmLearner`
  - trace-only backend boundary for future ICAPS 2025 implementation.

- [ ] **Step 1: Write failing learner-conformance and missing-backend tests**

Add:

```python
def test_action_model_learner_protocol_returns_ecsa_proposals_only(): ...
def test_missing_macq_returns_typed_learner_failure(): ...
def test_macq_locm_proposal_contains_no_macq_types(): ...
def test_incompatible_partial_trace_returns_typed_failure_not_kernel_exception(): ...
```

- [ ] **Step 2: Run protocol tests without installing MAcq and verify RED**

Run:

```bash
python -m pytest tests/test_world_model_learners.py -q
```

Expected: missing learner package.

- [ ] **Step 3: Implement backend-neutral learner contracts**

Exact protocol:

```python
class ActionModelLearner(Protocol):
    learner_id: str

    def update(
        self,
        traces: tuple[AcquisitionTrace, ...],
        current_contracts: tuple[WorldContractHypothesis, ...],
    ) -> ActionModelProposal | LearnerFailure: ...
```

`ActionModelProposal` contains only ECSA hypothesis values/evidence IDs.

- [ ] **Step 4: Add optional MAcq dependency**

Add:

```toml
world-model-planning = ["macq==0.3.11"]
```

Do not add it to default dependencies.

- [ ] **Step 5: Implement `MacqLocmLearner`**

Translate ECSA action-only traces into MAcq `ActionObservation` /
`ObservedTraceList`, invoke MAcq's LOCM extractor, and translate the returned
model into ECSA `ActionModelProposal`.

Do not expose MAcq `Model`, `LearnedLiftedAction`, or fluent classes beyond
`macq.py`.

- [ ] **Step 6: Run MAcq conformance tests with the optional extra**

Run:

```bash
python -m pip install -e '.[test,world-model-planning]'
python -m pytest tests/test_world_model_learners.py tests/test_world_model_macq.py -q
```

Expected: all pass under Python 3.12.

- [ ] **Step 7: Implement the trace-only learner boundary**

`trace_only.py` defines the adapter/protocol surface and explicit
`LearnerFailure(reason="backend-not-implemented")`; it does not implement the
2025 algorithm in v1.

- [ ] **Step 8: Commit**

```bash
git add src/ecsa/world_model/learners pyproject.toml   tests/test_world_model_learners.py tests/test_world_model_macq.py
git commit -m "feat: add action model learner and MAcq LOCM backend"
```

---

### Task 8: LOCM2-style multi-state-machine backend

**Files:**
- Create: `src/ecsa/world_model/learners/locm2.py`
- Test: `tests/test_world_model_locm2.py`

**Interfaces:**
- Consumes: `AcquisitionTrace`.
- Produces: `Locm2Learner(ActionModelLearner)` returning the same
  `ActionModelProposal | LearnerFailure` contract as Task 7.

- [ ] **Step 1: Write a failing domain fixture that LOCM's single-FSM representation cannot express**

Use a small synthetic driver/resource-style action trace where one object sort
has two independent aspects of state. Assert:

```python
def test_locm2_produces_multiple_state_machines_for_one_latent_sort(): ...
def test_locm2_output_translates_to_standard_action_model_proposal(): ...
def test_locm2_is_invariant_to_action_and_object_renaming(): ...
```

- [ ] **Step 2: Run Task 8 tests and verify RED**

Run:

```bash
python -m pytest tests/test_world_model_locm2.py -q
```

Expected: missing LOCM2 learner.

- [ ] **Step 3: Implement LOCM2 transition-centered machine synthesis**

Implement the Cresswell/Gregory LOCM2 representation behind private types in
`locm2.py`:

- infer sorts from action-parameter co-occurrence;
- construct transition sets per sort;
- partition transitions into multiple compatible state machines;
- derive parameter bindings/relations;
- translate machines into ECSA predicate/action-schema hypotheses.

Only `ActionModelProposal` crosses the module boundary.

- [ ] **Step 4: Verify Task 8 GREEN and compare LOCM/LOCM2 behavior**

Run:

```bash
python -m pytest tests/test_world_model_locm2.py tests/test_world_model_macq.py -q
```

Expected: LOCM2 fixture passes; MAcq LOCM regression remains green.

- [ ] **Step 5: Commit**

```bash
git add src/ecsa/world_model/learners/locm2.py tests/test_world_model_locm2.py
git commit -m "feat: add LOCM2 world contract learner"
```

---

### Task 9: Split the autonomy prototype around the new world-model kernel

**Files:**
- Delete after migration: `src/ecsa/autonomy.py`
- Create: `src/ecsa/autonomy/__init__.py`
- Create: `src/ecsa/autonomy/scientist.py`
- Modify imports in: `tests/test_discoveryworld_autonomous_policy.py`
- Test: `tests/test_autonomous_scientist.py`

**Interfaces:**
- Consumes:
  - `WorldModelAcquisitionKernel`
  - `ContractExperimentCoordinator`
  - raw/perceptual observation
  - goal text/value supplied by environment integration.
- Produces:
  - `AutonomousScientist.observe_transition(...)`
  - `AutonomousScientist.choose_experiment(...)`

- [ ] **Step 1: Write failing tests for the new autonomy API**

Add:

```python
def test_autonomous_scientist_chooses_from_contract_experiments_not_action_roles(): ...
def test_autonomous_scientist_updates_world_model_after_every_outcome(): ...
def test_autonomous_scientist_has_no_benchmark_semantic_action_categories(): ...
```

- [ ] **Step 2: Run Task 9 tests and verify RED**

Run:

```bash
python -m pytest tests/test_autonomous_scientist.py -q
```

Expected: package/API missing.

- [ ] **Step 3: Implement the thin coordinator**

Exact constructor:

```python
class AutonomousScientist:
    def __init__(
        self,
        *,
        world_model: WorldModelAcquisitionKernel,
        experiments: ContractExperimentCoordinator,
        perception: PerceptionFrontend,
    ) -> None: ...

    def choose_experiment(
        self,
        observation: RawObservation,
        actions: tuple[RawActionSchema, ...],
        goal: FrozenRawValue,
        budget: ExperimentBudget,
    ) -> ContractExperiment: ...

    def observe_transition(
        self,
        transition: InteractionTransition,
    ) -> WorldModelUpdate: ...
```

No `ActionRole` enum or semantic action role mapping is retained.

- [ ] **Step 4: Migrate imports and remove the monolithic module**

Move only still-general functionality into the package. Benchmark-specific
logic is not migrated.

- [ ] **Step 5: Run Task 9 tests plus current core tests**

Run:

```bash
python -m pytest tests/test_autonomous_scientist.py tests/test_science.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/ecsa/autonomy tests/test_autonomous_scientist.py   tests/test_discoveryworld_autonomous_policy.py
git commit -m "refactor: coordinate autonomy through world model contracts"
```

---

### Task 10: DiscoveryWorld raw adapter and Arena B migration

**Files:**
- Create: `src/ecsa/benchmarks/discoveryworld/raw_environment.py`
- Create: `src/ecsa/benchmarks/discoveryworld/arena_b.py`
- Create: `src/ecsa/benchmarks/discoveryworld/perception.py`
- Preserve: `src/ecsa/benchmarks/discoveryworld/environment.py`
- Preserve: `src/ecsa/benchmarks/discoveryworld/reactor_lab.py`
- Preserve: `src/ecsa/benchmarks/discoveryworld/arena.py`
- Retire from Arena B only: `src/ecsa/benchmarks/discoveryworld/policies/autonomous_scientist.py`
- Test: `tests/test_discoveryworld_raw_environment.py`
- Test: `tests/test_discoveryworld_arena_b.py`

**Interfaces:**
- Consumes: pinned DiscoveryWorld public API only.
- Produces:
  - `DiscoveryWorldRawEnvironment(RawEnvironmentPort)`
  - `DiscoveryWorldStructuredDecoder(StructuredEntityDecoder)`
  - Arena B runner using `AutonomousScientist`.

- [ ] **Step 1: Write failing raw-adapter firewall tests**

Add tests asserting:

```python
def test_raw_adapter_exposes_action_symbol_and_arguments_without_role_map(): ...
def test_dialog_option_is_exposed_as_raw_action_surface_without_dialog_semantics_in_core(): ...
def test_raw_adapter_records_failed_action_as_outcome(): ...
def test_raw_adapter_policy_surface_contains_no_scorecard_or_hidden_world_fields(): ...
```

- [ ] **Step 2: Write the static Arena B semantic-leak test**

The autonomous production path must fail if it contains/imports any of:

```text
_ROLE_MAP
PROBE_BINARY
OBSERVE_UNARY
ACQUIRE
PLACE
MeasurementKind
ReactorMeasurement
ReactorMechanismHypothesis
ReactorLabScientificSidecar
```

The test may mention these strings; Arena B production modules may not.

- [ ] **Step 3: Run Task 10 tests and verify RED**

Run:

```bash
python -m pytest   tests/test_discoveryworld_raw_environment.py   tests/test_discoveryworld_arena_b.py -q
```

Expected: missing raw adapter/Arena B.

- [ ] **Step 4: Implement `DiscoveryWorldRawEnvironment`**

Responsibilities are limited to:

- obtain public observation;
- freeze it into `RawObservation`;
- expose public raw action schemas and public argument candidates;
- execute requested `GroundAction`;
- tick exactly once after every attempted action;
- expose public outcome, done, and step count.

Any normalization needed to serialize DiscoveryWorld's API shape is transport
logic only. Do not infer action purpose.

- [ ] **Step 5: Implement `DiscoveryWorldStructuredDecoder`**

Decode publicly exposed object records into neutral `EntityObservation`
features. UUID/name/description/location/inventory-like membership remain
provenanced public features; no domain role is assigned.

- [ ] **Step 6: Implement Arena B loop**

Use:

```python
before = environment.observe_raw()
experiment = scientist.choose_experiment(...)
outcome = environment.execute_raw_action(experiment.action)
after = environment.observe_raw()
scientist.observe_transition(
    InteractionTransition(...)
)
```

Arena B does not instantiate the Reactor sidecar.

- [ ] **Step 7: Run Task 10 tests and verify GREEN**

Run Task 10 tests. Expected: all pass on real pinned DiscoveryWorld seed 0.

- [ ] **Step 8: Re-run Arena A regression tests**

Run:

```bash
python -m pytest   tests/test_discoveryworld_environment.py   tests/test_discoveryworld_reactor_lab.py   tests/test_discoveryworld_transfer_candidates.py   tests/test_discoveryworld_arena.py -q
```

Expected: Arena A remains green.

- [ ] **Step 9: Commit**

```bash
git add src/ecsa/benchmarks/discoveryworld tests/test_discoveryworld_raw_environment.py   tests/test_discoveryworld_arena_b.py
git commit -m "feat: run DiscoveryWorld through induced world contracts"
```

---

### Task 11: End-to-end autonomous contract discovery, transfer, CI, and documentation

**Files:**
- Create: `tests/test_world_model_end_to_end.py`
- Modify: `.github/workflows/test.yml`
- Modify or create autonomous workflow file under: `.github/workflows/`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: complete v1 world-model stack.
- Produces: acceptance evidence for the approved spec.

- [ ] **Step 1: Write the two-world renaming acceptance test**

Create two synthetic raw environments with identical transition structure but
unrelated symbols and entity labels.

Assert:

```python
canonicalize_world_contract(contract_a).fingerprint == (
    canonicalize_world_contract(contract_b).fingerprint
)
```

Neither environment fixture uses DiscoveryWorld terms.

- [ ] **Step 2: Write the structured-vs-slot perception acceptance test**

Feed the same toy transitions through:

- `StructuredObservationFrontend`;
- `SlotAttentionFrontend` with a deterministic fake slot encoder.

Assert compatible canonical contracts.

- [ ] **Step 3: Write the persistent contract-transfer acceptance test**

Run context A, admit a canonical world contract to
`SQLiteWorldContractStore`, then run renamed context B.

Assert:

- source names/IDs are not reused as facts;
- structural transfer candidate is retrieved;
- candidate affects experiment priority only;
- target evidence is still required before target admission.

- [ ] **Step 4: Write the real DiscoveryWorld Arena B smoke**

Run pinned Reactor Lab / Normal / seed 0 with a small explicit step budget.

Acceptance is not task completion yet. Assert:

- raw transitions are recorded;
- world-contract hypotheses are produced;
- no Arena A semantic sidecar types enter the autonomous path;
- no LLM/API key is required.

- [ ] **Step 5: Run end-to-end tests and verify GREEN**

Run:

```bash
python -m pytest   tests/test_world_model_contracts.py   tests/test_world_model_perception.py   tests/test_world_model_grounding.py   tests/test_world_contract_hypotheses.py   tests/test_world_contract_canonicalization.py   tests/test_world_model_kernel.py   tests/test_world_contract_memory.py   tests/test_world_model_experiments.py   tests/test_world_model_learners.py   tests/test_world_model_macq.py   tests/test_world_model_locm2.py   tests/test_autonomous_scientist.py   tests/test_discoveryworld_raw_environment.py   tests/test_discoveryworld_arena_b.py   tests/test_world_model_end_to_end.py -q
```

Expected: all world-model/Arena B tests pass.

- [ ] **Step 6: Update CI**

Keep the full project suite. Add/retain a fast world-model/Arena B job that
installs:

```bash
pip install -e '.[test,mlmd,discoveryworld,world-model-planning]'
```

and runs the Task 11 test list without R/VARIO setup.

Use branch concurrency so superseded `main` runs are cancelled.

- [ ] **Step 7: Update architecture documentation**

Document:

```text
Arena A = structured transfer / regression baseline
Arena B = autonomous world-contract discovery
```

Document that Slot Attention is a future optional frontend and that LOCM2 is
an ECSA backend rather than a capability provided by MAcq.

- [ ] **Step 8: Run the complete repository verification**

Run:

```bash
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
python -m pytest -q
```

Expected: zero failed tests. If external VARIO infrastructure is unavailable,
report those exact failures separately; do not call the implementation complete
until the project CI run is green.

- [ ] **Step 9: Run the static autonomous-boundary check**

Run:

```bash
grep -RniE   '_ROLE_MAP|PROBE_BINARY|OBSERVE_UNARY|ACQUIRE|PLACE|MeasurementKind|ReactorMeasurement|ReactorMechanismHypothesis|ReactorLabScientificSidecar'   src/ecsa/world_model   src/ecsa/autonomy   src/ecsa/benchmarks/discoveryworld/raw_environment.py   src/ecsa/benchmarks/discoveryworld/arena_b.py
```

Expected: no output.

- [ ] **Step 10: Commit**

```bash
git add tests .github docs README.md
git commit -m "test: validate autonomous world contract discovery"
```

---

## Plan Self-Review

### Spec coverage

- Raw interaction contracts: Task 1.
- Structured perception and optional Slot Attention boundary: Task 2.
- Identity hypotheses and interaction grounding: Tasks 2-3.
- Anonymous numeric/text feature induction: Task 3.
- World-contract hypothesis representation: Task 4.
- Renaming invariance/canonicalization: Tasks 4 and 11.
- Contract population, evidence, persistence, transfer candidates: Task 5.
- Active contract experiments and ScienceKernel coordination: Task 6.
- MAcq backend: Task 7.
- LOCM path: Task 7.
- LOCM2 multi-state-machine path: Task 8.
- Trace-only ICAPS 2025 backend boundary: Task 7.
- Split of current autonomy prototype: Task 9.
- DiscoveryWorld raw transport and Arena B: Task 10.
- Arena A preservation: Tasks 10-11.
- Slot Attention future implementation boundary without PyTorch core dependency: Tasks 2 and 11.
- Cross-context structural transfer: Tasks 5 and 11.

No approved-spec requirement is intentionally left without a task.

### Type consistency

The dependency chain is one-way:

```text
world_model.contracts
    -> perception
    -> grounding
    -> hypotheses/canonical
    -> kernel/memory
    -> experiments
    -> learners
    -> autonomy.scientist
    -> DiscoveryWorld raw adapter / Arena B
```

Learners return only `ActionModelProposal | LearnerFailure`; backend-native
types never cross into the kernel.

### Scope/proportion

The plan implements World Model Acquisition v1 only. It does not implement a
real neural Slot Attention model or the ICAPS 2025 trace-only learner; it
implements their approved boundaries and acceptance fixtures. The only
additional research implementation in v1 is LOCM2 because the approved spec
requires a LOCM2-style path and MAcq currently provides LOCM, not LOCM2.
