# DiscoveryWorld Reactor Lab Transfer Arena Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a pinned DiscoveryWorld Reactor Lab / Normal arena that compares cold ECSA against progressive mechanism reuse across official seeds 0-4 without oracle leakage.

**Architecture:** Keep DiscoveryWorld behind an optional benchmark adapter and keep ECSA core independent of the environment. Reactor-specific public evidence is translated by a scientific sidecar into prospective mechanism records; cross-seed knowledge is surfaced first as transfer candidates, then becomes applicable only after target-context validation. The arena pairs cold and reuse runs under one provider-neutral policy protocol and writes evaluator-only scorecards separately from policy-facing data.

**Tech Stack:** Python 3.12, pytest, DiscoveryWorld pinned at fd591323920be0d3786ef350955de1945aa571e5, existing MLMD 1.21 / SQLite MechanismRepository, existing ECSA contracts.

**Spec:** docs/superpowers/specs/2026-10-06-discoveryworld-reactor-transfer-arena-design.md

## Global Constraints

- DiscoveryWorld source is https://github.com/allenai/discoveryworld at exact commit fd591323920be0d3786ef350955de1945aa571e5.
- Canonical benchmark is Reactor Lab / Normal / one agent / official seeds 0,1,2,3,4 / max 1000 steps.
- Cold and reuse arms use the same policy factory path, config hash, model settings, action set, seed, and step budget; only mechanism memory differs.
- Every performAgentAction() call is followed by exactly one tick(), even when DiscoveryWorld reports the action as failed or invalid.
- Policy/scientific runtime must never consume getTaskScorecard(), criticalQuestions, criticalHypotheses, hidden scoringInfo, hidden crystal/reactor attributes, or exported world history.
- Canonical context ID is discoveryworld:reactor-lab:normal:seed-<N>; domain_id=discoveryworld, task_id=reactor-lab, regime_id=normal.
- Unknown context membership never implies applicability. Cross-context knowledge enters through find_transfer_candidates() and requires target-context validation before reuse as a fact.
- The full model-backed benchmark is manual; CI uses real DiscoveryWorld plus deterministic no-LLM policies only.
- Benchmark-specific code must not become a dependency of ecsa.science, ecsa.mechanisms, or other core scientific modules.

## File Structure

- Modify pyproject.toml — optional commit-pinned DiscoveryWorld dependency.
- Modify .github/workflows/test.yml — install the discoveryworld extra and run real headless DiscoveryWorld tests.
- Modify src/ecsa/mechanisms.py — add the repository transfer-candidate contract only; no DiscoveryWorld types.
- Modify src/ecsa/adapters/mlmd.py — implement the generic transfer-candidate query.
- Create src/ecsa/benchmarks/__init__.py — benchmark package marker.
- Create src/ecsa/benchmarks/discoveryworld/__init__.py — DiscoveryWorld package marker.
- Create src/ecsa/benchmarks/discoveryworld/contracts.py — JSON/action/policy/arena/Reactor scientific data contracts.
- Create src/ecsa/benchmarks/discoveryworld/environment.py — strict DiscoveryWorldAPI wrapper and evaluator-only score access.
- Create src/ecsa/benchmarks/discoveryworld/reactor_lab.py — public measurement parsing, prospective hypotheses, validation, and content-addressed Reactor rule artifacts.
- Create src/ecsa/benchmarks/discoveryworld/runlog.py — JSON/JSONL run artifact writer with oracle separation.
- Create src/ecsa/benchmarks/discoveryworld/metrics.py — per-run counters and transfer-gain calculations.
- Create src/ecsa/benchmarks/discoveryworld/arena.py — episode loop, progressive cold/reuse orchestration, policy-factory loader, CLI.
- Create tests/test_discoveryworld_environment.py.
- Create tests/test_discoveryworld_reactor_lab.py.
- Create tests/test_discoveryworld_transfer_candidates.py.
- Create tests/test_discoveryworld_arena.py.
- Modify README.md and docs/INTEGRATION_PLAN.md only after the executable arena is green.

## Review Focus

1. Invalid/malformed DiscoveryWorld actions must still consume exactly one tick after performAgentAction(); pinned by test_failed_action_still_ticks_once in Task 1.
2. Oracle scorecard/world-state data must be inaccessible through policy-facing adapter methods; pinned by test_policy_surface_never_reads_scorecard and test_real_reactor_observation_has_no_oracle_keys in Task 1.
3. Headless CI must load all five official Reactor Lab Normal seeds without opening a display; pinned by test_real_reactor_lab_all_official_seeds_load in Task 1 and SDL dummy settings in CI.
4. Transfer lookup with missing domain/task/regime/assumptions must fail closed, and a candidate from seed 0 must remain absent from find_applicable() in seed 1; pinned in Task 3.
5. Transfer-gain metrics with a cold denominator of zero must return an explicit undefined value (None), never inf/NaN or a division error; pinned by test_transfer_gain_is_none_when_cold_count_is_zero in Task 6.

---

### Task 1: Pin DiscoveryWorld and build the environment/oracle firewall

**Files:**
- Modify: pyproject.toml
- Modify: .github/workflows/test.yml
- Create: src/ecsa/benchmarks/__init__.py
- Create: src/ecsa/benchmarks/discoveryworld/__init__.py
- Create: src/ecsa/benchmarks/discoveryworld/contracts.py
- Create: src/ecsa/benchmarks/discoveryworld/environment.py
- Test: tests/test_discoveryworld_environment.py

**Interfaces:**
- Consumes: official DiscoveryWorldAPI.
- Produces: DiscoveryWorldEpisodeConfig, JSONValue, ActionPacket, DiscoveryWorldEnvironmentAdapter, DiscoveryWorldEvaluation.

- [ ] **Step 1: Write the failing real-environment tests**

Add tests with these assertions:

~~~python
def test_real_reactor_lab_all_official_seeds_load():
    for seed in range(5):
        env = DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(seed)
        observation = env.observe()
        assert observation["ui"]["taskProgress"][0]["description"]
        assert env.steps == 0
        assert env.done is False

def test_action_call_ticks_exactly_once():
    env = DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(0)
    before = env.steps
    env.act({"action": "TELEPORT_TO_LOCATION", "arg1": "quantum reactor lab"})
    assert env.steps == before + 1

def test_failed_action_still_ticks_once():
    env = DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(0)
    before = env.steps
    result = env.act({"action": "THIS_ACTION_DOES_NOT_EXIST"})
    assert result.success is False
    assert env.steps == before + 1

def test_real_reactor_observation_has_no_oracle_keys():
    observation = DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(0).observe()
    encoded = json.dumps(observation).lower()
    assert "criticalquestions" not in encoded
    assert "criticalhypotheses" not in encoded
    assert "scorecard" not in encoded

def test_policy_surface_never_reads_scorecard(monkeypatch):
    env = DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(0)
    monkeypatch.setattr(env._api, "getTaskScorecard", lambda: (_ for _ in ()).throw(AssertionError("oracle read")))
    env.observe()
    env.available_actions()
    env.teleport_locations()
~~~

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest tests/test_discoveryworld_environment.py -q`

Expected: FAIL because the discoveryworld extra/package and ECSA benchmark adapter do not exist.

- [ ] **Step 3: Pin the optional benchmark dependency and headless CI**

Add optional dependency:

```toml
discoveryworld = ["discoveryworld @ git+https://github.com/allenai/discoveryworld.git@fd591323920be0d3786ef350955de1945aa571e5"]
```

Update CI install to include discoveryworld. Set `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy` for pytest.

- [ ] **Step 4: Implement benchmark contracts in contracts.py**

Define exact public types:

```python
type JSONScalar = None | bool | int | float | str
type JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
type ActionPacket = dict[str, JSONScalar]

@dataclass(frozen=True)
class DiscoveryWorldEpisodeConfig:
    scenario: str
    difficulty: str
    seed: int
    max_steps: int

@dataclass(frozen=True)
class DiscoveryWorldActionResult:
    success: bool
    errors: tuple[str, ...]

@dataclass(frozen=True)
class DiscoveryWorldEvaluation:
    completed_successfully: bool
    score_normalized: float
    steps: int
    scorecard: dict[str, JSONValue]
```

Validate seed 0-4 and positive max_steps in the Reactor convenience constructor rather than making the generic config reject future scenarios.

- [ ] **Step 5: Implement DiscoveryWorldEnvironmentAdapter**

Required signatures:

```python
class DiscoveryWorldEnvironmentAdapter:
    @classmethod
    def reactor_lab_normal(cls, seed: int, *, max_steps: int = 1000, thread_id: int = 0) -> "DiscoveryWorldEnvironmentAdapter": ...

    def observe(self) -> dict[str, JSONValue]: ...
    def available_actions(self) -> dict[str, JSONValue]: ...
    def teleport_locations(self) -> dict[str, JSONValue]: ...
    def act(self, action: ActionPacket) -> DiscoveryWorldActionResult: ...

    @property
    def done(self) -> bool: ...

    @property
    def steps(self) -> int: ...

    def evaluate_after_run(self) -> DiscoveryWorldEvaluation: ...
```

act() must call performAgentAction() and then tick() in a finally-style single path so success/failure cannot change tick count. evaluate_after_run() is the only adapter method allowed to call getTaskScorecard().

- [ ] **Step 6: Run the environment tests GREEN**

Run: `pytest tests/test_discoveryworld_environment.py -q`

Expected: all tests PASS with real DiscoveryWorld.

- [ ] **Step 7: Run the existing suite**

Run: `pytest -q`

Expected: existing tests plus DiscoveryWorld environment tests PASS.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .github/workflows/test.yml src/ecsa/benchmarks tests/test_discoveryworld_environment.py
git commit -m "feat: add pinned DiscoveryWorld environment adapter"
```

### Task 2: Parse public Reactor Lab evidence and qualify prospective rules

**Files:**
- Modify: src/ecsa/benchmarks/discoveryworld/contracts.py
- Create: src/ecsa/benchmarks/discoveryworld/reactor_lab.py
- Test: tests/test_discoveryworld_reactor_lab.py

**Interfaces:**
- Consumes: pre/post agent-visible observations and ActionPacket from Task 1.
- Produces: MeasurementKind, ReactorMeasurement, ReactorFrequencyPrediction, ReactorMechanismHypothesis, ReactorValidationEvent, ReactorLabScientificSidecar, ReactorRuleArtifactStore.

- [ ] **Step 1: Write failing measurement-parser tests using real upstream message forms**

Assert parsing for densitometer, thermometer, spectrometer channel values, microscope quantum gap, and radiation meter. Include a malformed/inconclusive message test that returns no measurement rather than inventing a value.

Required examples include:

```python
assert parse_public_measurement("The density ... approximately 12.340 grams per cubic centimeter.").values == (12.34,)
assert parse_public_measurement("The thermometer reports a temperature of 17.25 degrees Celsius.").values == (17.25,)
assert parse_public_measurement("- Channel 5: 19.20").channel_values[4] == 19.2
assert parse_public_measurement("The quantum gap ... 11.4 nm").values == (11.4,)
assert parse_public_measurement("The radiation meter reports a level of 8.22 micro Seiverts per hour.").values == (8.22,)
assert parse_public_measurement("The results are inconclusive.") is None
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_discoveryworld_reactor_lab.py -q`

Expected: FAIL because Reactor contracts/sidecar do not exist.

- [ ] **Step 3: Add the structured scientific contracts**

Define:

```python
class MeasurementKind(StrEnum):
    DENSITY = "density"
    TEMPERATURE = "temperature"
    QUANTUM_SIZE = "quantum_size"
    RADIATION = "radiation"
    SPECTRUM = "spectrum"

@dataclass(frozen=True)
class ReactorMeasurement: ...
@dataclass(frozen=True)
class ReactorFrequencyPrediction: ...
@dataclass(frozen=True)
class ReactorMechanismHypothesis: ...
@dataclass(frozen=True)
class ReactorValidationEvent: ...
```

ReactorMechanismHypothesis fields are hypothesis_id, measurement_kind, slope, offset, source_evidence_ids, predictions. ReactorFrequencyPrediction binds target_crystal_uuid, target_reactor_uuid, predicted_frequency, and frozen_step.

- [ ] **Step 4: Implement parse_public_measurement() and record_transition()**

Required signatures:

```python
def parse_public_measurement(message: str) -> ParsedMeasurement | None: ...

class ReactorLabScientificSidecar:
    def record_transition(
        self,
        *,
        step: int,
        context_id: str,
        pre_observation: dict[str, JSONValue],
        action: ActionPacket,
        action_result: DiscoveryWorldActionResult,
        post_observation: dict[str, JSONValue],
    ) -> tuple[ReactorMeasurement, ...]: ...
```

Only successful USE instrument-on-crystal actions may create ReactorMeasurement. Object names/UUIDs come from the public pre-observation; measured values come from post-observation ui.lastActionMessage.

- [ ] **Step 5: Write RED tests for prospective validation and oracle independence**

Tests must show:

1. freeze_hypothesis() stores predictions before validation;
2. a pre-observation showing an uncalibrated public reactor followed by a post-observation showing the same UUID named crystal reactor (activated) yields a successful ReactorValidationEvent;
3. no validation occurs if it was already activated before the prediction;
4. no hidden isActivated/resonanceFreq field is required by the API.

- [ ] **Step 6: Implement hypothesis freezing and validation**

Required methods:

```python
class ReactorLabScientificSidecar:
    def freeze_hypothesis(self, hypothesis: ReactorMechanismHypothesis) -> None: ...
    def observe_validation(
        self,
        *,
        step: int,
        pre_observation: dict[str, JSONValue],
        post_observation: dict[str, JSONValue],
    ) -> tuple[ReactorValidationEvent, ...]: ...
```

Detect activation exclusively by public object UUID/name transitions.

- [ ] **Step 7: Implement content-addressed ReactorRuleArtifactStore and MechanismRecord builder**

Signatures:

```python
class ReactorRuleArtifactStore:
    def write(self, hypothesis: ReactorMechanismHypothesis, validation: ReactorValidationEvent) -> str: ...
    def load(self, artifact_id: str) -> dict[str, JSONValue]: ...

def build_admitted_reactor_mechanism(
    *,
    hypothesis: ReactorMechanismHypothesis,
    validation: ReactorValidationEvent,
    context_id: str,
    artifact_store: ReactorRuleArtifactStore,
) -> MechanismRecord: ...
```

Output kind=SYMBOLIC_RULE, status=ADMITTED, scope context_ids=(context_id,), regime_ids=("normal",), domain_ids=("discoveryworld",), task_ids=("reactor-lab",), required assumptions include public-observation and linear-family identifiers.

- [ ] **Step 8: Run Reactor tests and full suite GREEN**

Run: `pytest tests/test_discoveryworld_reactor_lab.py -q && pytest -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/ecsa/benchmarks/discoveryworld/contracts.py src/ecsa/benchmarks/discoveryworld/reactor_lab.py tests/test_discoveryworld_reactor_lab.py
git commit -m "feat: add Reactor Lab scientific evidence sidecar"
```

### Task 3: Add conservative cross-context transfer-candidate retrieval

**Files:**
- Modify: src/ecsa/mechanisms.py
- Modify: src/ecsa/adapters/mlmd.py
- Create: tests/test_discoveryworld_transfer_candidates.py

**Interfaces:**
- Consumes: ApplicabilityContext and admitted MechanismRecord.
- Produces: MechanismRepository.find_transfer_candidates(context) -> tuple[MechanismRecord, ...].

- [ ] **Step 1: Write the failing transfer-candidate tests**

Use source scope context seed-0 and target query seed-1.

Assertions:

```python
target = ApplicabilityContext(
    context_id="discoveryworld:reactor-lab:normal:seed-1",
    regime_id="normal",
    domain_id="discoveryworld",
    task_id="reactor-lab",
    assumptions=("public-observation-only", "linear-family"),
)

assert repo.find_applicable(target) == ()
assert repo.find_transfer_candidates(target) == (source_mechanism,)
```

Also assert QUALIFIED/DEPRECATED records are excluded and missing regime/domain/task or empty assumptions raise ValueError.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_discoveryworld_transfer_candidates.py -q`

Expected: FAIL because the protocol/backend method does not exist.

- [ ] **Step 3: Extend MechanismRepository**

Add:

```python
def find_transfer_candidates(
    self,
    context: ApplicabilityContext,
) -> tuple[MechanismRecord, ...]: ...
```

Do not introduce DiscoveryWorld concepts into core.

- [ ] **Step 4: Implement MLMD transfer lookup**

Reuse latest-version, ADMITTED, supersession, and deprecation semantics from find_applicable(). Ignore context_ids only; match stored regime/domain/task constraints and required assumptions. Require query regime_id, domain_id, task_id, and at least one assumption.

- [ ] **Step 5: Run transfer tests and existing mechanism tests GREEN**

Run: `pytest tests/test_discoveryworld_transfer_candidates.py tests/test_mechanism_repository.py -q`

Expected: PASS.

- [ ] **Step 6: Run full suite**

Run: `pytest -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ecsa/mechanisms.py src/ecsa/adapters/mlmd.py tests/test_discoveryworld_transfer_candidates.py
git commit -m "feat: add conservative mechanism transfer candidates"
```

### Task 4: Build the policy boundary, episode loop, and oracle-separated run logs

**Files:**
- Modify: src/ecsa/benchmarks/discoveryworld/contracts.py
- Create: src/ecsa/benchmarks/discoveryworld/runlog.py
- Create: src/ecsa/benchmarks/discoveryworld/arena.py
- Test: tests/test_discoveryworld_arena.py

**Interfaces:**
- Consumes: environment adapter, MechanismRepository, ReactorLabScientificSidecar.
- Produces: DiscoveryWorldActionPolicy protocol, PolicyDecision, ScientificContext, ArenaEpisodeResult, run_episode().

- [ ] **Step 1: Write the failing policy/episode tests**

Define a deterministic test policy that returns one valid action and records the ScientificContext it receives.

Assertions:

- policy never receives DiscoveryWorldEvaluation/scorecard;
- each loop iteration calls observe -> decide -> act -> sidecar transition;
- loop stops at done or max_steps;
- evaluate_after_run() is called only after the policy loop;
- output directory contains run.json, actions.jsonl, observations.jsonl, scientific_events.jsonl, mechanism_events.jsonl, metrics.json, final_scorecard.json;
- final_scorecard.json is not read to construct later policy inputs.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_discoveryworld_arena.py::test_episode_loop_keeps_oracle_outside_policy_surface -q`

Expected: FAIL because policy/episode/logging APIs do not exist.

- [ ] **Step 3: Implement policy and episode contracts**

Add exact signatures:

```python
@dataclass(frozen=True)
class ScientificContext:
    context_id: str
    measurements: tuple[ReactorMeasurement, ...]
    transfer_candidates: tuple[MechanismRecord, ...]
    admitted_mechanisms: tuple[MechanismRecord, ...]

@dataclass(frozen=True)
class PolicyDecision:
    action: ActionPacket
    reasoning: str | None = None
    memory: str | None = None
    hypotheses: tuple[ReactorMechanismHypothesis, ...] = ()

@runtime_checkable
class DiscoveryWorldActionPolicy(Protocol):
    def decide(
        self,
        observation: dict[str, JSONValue],
        available_actions: dict[str, JSONValue],
        teleport_locations: dict[str, JSONValue],
        scientific_context: ScientificContext,
    ) -> PolicyDecision: ...

@dataclass(frozen=True)
class ArenaEpisodeResult: ...
```

- [ ] **Step 4: Implement ArenaRunWriter**

Create runlog.py with one writer responsible for the seven required files. final_scorecard.json is written only from DiscoveryWorldEvaluation after the loop; no reader API is provided to the policy path.

- [ ] **Step 5: Implement run_episode()**

Signature:

```python
def run_episode(
    *,
    config: DiscoveryWorldEpisodeConfig,
    policy: DiscoveryWorldActionPolicy,
    repository: MechanismRepository,
    output_dir: Path,
) -> ArenaEpisodeResult: ...
```

At each step:
1. observe;
2. build ScientificContext from find_applicable() and find_transfer_candidates();
3. call policy;
4. freeze any hypotheses before action;
5. act (which ticks exactly once);
6. observe post-action;
7. record public scientific evidence and validation;
8. admit only newly validated target-context mechanisms;
9. log event.

After loop, call evaluate_after_run() and write evaluator-only scorecard.

- [ ] **Step 6: Run targeted and full tests GREEN**

Run: `pytest tests/test_discoveryworld_arena.py -q && pytest -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ecsa/benchmarks/discoveryworld/contracts.py src/ecsa/benchmarks/discoveryworld/runlog.py src/ecsa/benchmarks/discoveryworld/arena.py tests/test_discoveryworld_arena.py
git commit -m "feat: add DiscoveryWorld episode harness"
```

### Task 5: Implement progressive cold-vs-reuse orchestration

**Files:**
- Modify: src/ecsa/benchmarks/discoveryworld/contracts.py
- Modify: src/ecsa/benchmarks/discoveryworld/arena.py
- Test: tests/test_discoveryworld_arena.py

**Interfaces:**
- Consumes: run_episode(), policy factory, repository factory.
- Produces: ArenaArm, PairedSeedResult, TransferArenaResult, run_progressive_transfer().

- [ ] **Step 1: Write failing progressive-transfer tests**

Use a deterministic no-LLM policy factory and a fake/light episode runner injected only at orchestration boundary to assert ordering/state isolation:

```python
result = run_progressive_transfer(..., seeds=(0,1,2,3,4))

assert [pair.seed for pair in result.pairs] == [0,1,2,3,4]
assert result.pairs[0].cold.starting_mechanism_count == 0
assert result.pairs[0].reuse.starting_mechanism_count == 0
assert all(pair.cold.starting_mechanism_count == 0 for pair in result.pairs)
assert result.pairs[1].reuse.starting_mechanism_count >= result.pairs[0].reuse.ending_mechanism_count
```

Also assert every pair uses the same policy config hash and no later seed appears in earlier reuse lineage.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_discoveryworld_arena.py::test_progressive_transfer_has_no_future_seed_leakage -q`

Expected: FAIL because orchestration API does not exist.

- [ ] **Step 3: Implement orchestration contracts**

Define:

```python
class ArenaArm(StrEnum):
    COLD = "cold"
    REUSE = "reuse"

@dataclass(frozen=True)
class PairedSeedResult: ...
@dataclass(frozen=True)
class TransferArenaResult: ...
```

- [ ] **Step 4: Implement run_progressive_transfer()**

Signature:

```python
def run_progressive_transfer(
    *,
    seeds: tuple[int, ...],
    policy_factory: PolicyFactory,
    policy_config: dict[str, JSONValue],
    output_dir: Path,
    max_steps: int = 1000,
) -> TransferArenaResult: ...
```

Use a new temporary/fresh SQLite repository for every cold episode and one persistent SQLite repository for the reuse arm. For each seed run cold and reuse with fresh policy instances created from identical config.

- [ ] **Step 5: Add one real DiscoveryWorld short-run pairing test**

Run both arms on Reactor Lab Normal seed 0 with max_steps=1 and deterministic policy; assert both real environments load, execute one action/tick, and produce paired result/log directories. This tests harness wiring without requiring task completion.

- [ ] **Step 6: Run arena tests and full suite GREEN**

Run: `pytest tests/test_discoveryworld_arena.py -q && pytest -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ecsa/benchmarks/discoveryworld/contracts.py src/ecsa/benchmarks/discoveryworld/arena.py tests/test_discoveryworld_arena.py
git commit -m "feat: add progressive DiscoveryWorld transfer arena"
```

### Task 6: Add transfer metrics and false-transfer accounting

**Files:**
- Create: src/ecsa/benchmarks/discoveryworld/metrics.py
- Modify: src/ecsa/benchmarks/discoveryworld/contracts.py
- Modify: src/ecsa/benchmarks/discoveryworld/arena.py
- Test: tests/test_discoveryworld_arena.py

**Interfaces:**
- Consumes: per-episode scientific/action events and paired cold/reuse results.
- Produces: ReactorRunMetrics, SeedTransferMetrics, TransferSummary, compute_transfer_gain().

- [ ] **Step 1: Write failing metrics tests**

Assertions cover:
- successful USE instrument-on-crystal counts as a measurement action;
- repeated same instrument/crystal pair increments action count but not distinct-measurement count;
- candidate retrieval/testing/rejection/acceptance are separate counters;
- false transfer increments only after a target-context scientific commitment fails prospective validation;
- cold denominator zero returns None.

Required helper:

```python
def compute_transfer_gain(*, cold: int, reuse: int) -> float | None:
    ...
```

Test:

```python
assert compute_transfer_gain(cold=10, reuse=4) == pytest.approx(0.6)
assert compute_transfer_gain(cold=0, reuse=0) is None
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_discoveryworld_arena.py -k "transfer_gain or false_transfer or measurement" -q`

Expected: FAIL because metrics module/counters do not exist.

- [ ] **Step 3: Implement metrics.py**

Define immutable metric records. StepTransferGain and MeasurementTransferGain are computed per seed; aggregate summary reports raw seeds 1-4 plus mean over defined gains only.

Never coerce undefined gain to zero.

- [ ] **Step 4: Wire metrics into run_episode() and run_progressive_transfer()**

metrics.json contains per-run counters. Arena summary contains per-seed cold/reuse official metrics and ECSA transfer metrics, then aggregate values.

- [ ] **Step 5: Run metrics and full tests GREEN**

Run: `pytest tests/test_discoveryworld_arena.py -q && pytest -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ecsa/benchmarks/discoveryworld/metrics.py src/ecsa/benchmarks/discoveryworld/contracts.py src/ecsa/benchmarks/discoveryworld/arena.py tests/test_discoveryworld_arena.py
git commit -m "feat: add DiscoveryWorld transfer metrics"
```

### Task 7: Add provider-neutral CLI, final docs, and full verification

**Files:**
- Modify: src/ecsa/benchmarks/discoveryworld/arena.py
- Modify: README.md
- Modify: docs/INTEGRATION_PLAN.md
- Test: tests/test_discoveryworld_arena.py

**Interfaces:**
- Consumes: policy factory path in module:factory form and JSON config.
- Produces: load_policy_factory(), CLI main(), documented benchmark command.

- [ ] **Step 1: Write failing policy-loader/CLI tests**

Create a test-only module factory and assert:

```python
factory = load_policy_factory("tests.discoveryworld_policy:create_policy")
policy = factory({"model": "deterministic-test"})
assert isinstance(policy, DiscoveryWorldActionPolicy)
```

Assert invalid import string, missing factory, non-callable factory, and factory returning a non-policy object fail with clear ValueError/TypeError.

Assert the config hash is deterministic under JSON key reordering.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_discoveryworld_arena.py -k "policy_factory or config_hash" -q`

Expected: FAIL because loader/CLI helpers do not exist.

- [ ] **Step 3: Implement load_policy_factory() and CLI main()**

Required command surface:

```text
python -m ecsa.benchmarks.discoveryworld.arena \
  --scenario "Reactor Lab" \
  --difficulty Normal \
  --seeds 0,1,2,3,4 \
  --arms cold,reuse \
  --max-steps 1000 \
  --output <dir> \
  --policy-factory package.module:create_policy \
  --policy-config policy.json
```

Reject any first-arena scenario/difficulty other than Reactor Lab/Normal rather than silently changing benchmark semantics.

- [ ] **Step 4: Update README and integration plan after code is green**

Document:
- DiscoveryWorld commit pin;
- optional install extra;
- manual benchmark command;
- oracle firewall;
- cold/reuse progressive protocol;
- distinction between CI smoke tests and full model-backed arena;
- the fact that computational DreamCoder reuse is no longer being used as a proxy for environment sample efficiency.

- [ ] **Step 5: Run the complete verification suite**

Run:

```bash
pytest -q
```

Expected: all existing ECSA + BCS + TTT + BOCPD + DreamCoder + Stitch + VARIO + MLMD + DiscoveryWorld tests PASS.

Then run the real benchmark smoke command with a deterministic test policy, Reactor Lab Normal, seeds 0-4, max_steps=1, both arms. Expected: ten paired episode directories and one summary artifact, with no policy/provider network calls.

- [ ] **Step 6: Inspect output firewall manually**

Open one generated run directory and verify:
- observations.jsonl contains only agent-visible observations;
- final_scorecard.json exists only as evaluator output;
- no criticalQuestions/criticalHypotheses/scoringInfo appears in policy/scientific event files;
- run.json records both ECSA and DiscoveryWorld revisions.

- [ ] **Step 7: Commit**

```bash
git add src/ecsa/benchmarks/discoveryworld README.md docs/INTEGRATION_PLAN.md tests/test_discoveryworld_arena.py
git commit -m "feat: finish DiscoveryWorld Reactor transfer arena"
```
