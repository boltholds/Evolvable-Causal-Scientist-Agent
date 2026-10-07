# DiscoveryWorld Reactor Lab Transfer Arena Design

> **Status:** Arena A / structured-transfer baseline. This design remains the
> regression and ablation path for mechanism-memory transfer. Autonomous
> world-contract discovery is specified separately in
> `2026-10-07-world-model-acquisition-kernel-design.md`.

Date: 2026-10-06

## 1. Purpose

The first external ECSA arena will use **DiscoveryWorld Reactor Lab / Normal**
instead of a synthetic environment.

The experiment asks one concrete question:

> Does a persistent ECSA mechanism library reduce the number of real
> environment interactions and scientific measurements needed to solve a new
> parametric Reactor Lab instance, compared with the same agent starting cold?

The benchmark must exercise the existing ECSA mechanism-memory path rather
than introduce a new toy simulator:

```text
DiscoveryWorld
      |
agent-visible observations/actions
      |
scientific sidecar
      |
MechanismRepository
      |
transfer candidate retrieval
      |
target-context validation
      |
runtime reuse
```

This first arena is about **transfer/sample efficiency**, not about proving
every ECSA subsystem at once.

## 2. Upstream benchmark

Use the official Apache-2.0 repository:

```text
https://github.com/allenai/discoveryworld
commit: fd591323920be0d3786ef350955de1945aa571e5
```

The dependency must be commit-pinned.

The canonical scenario is:

```text
scenarioName = "Reactor Lab"
difficulty   = "Normal"
seeds        = 0, 1, 2, 3, 4
num agents   = 1
max steps    = 1000
```

DiscoveryWorld's official baseline documentation uses a 1000-step cap for
Normal and Challenge tasks and identifies seeds 0-4 as the official benchmark
set.

### Why Reactor Lab

Reactor Lab Normal contains four quantum crystals. Reactors for crystals 1 and
2 are already correctly tuned; the agent must infer the frequencies for
crystals 3 and 4 using five scientific instruments and activate all reactors.

The upstream scenario generator makes the resonance frequency a linear
function of one measurement dimension:

```text
resonance_frequency = slope * measurement + offset
```

For Normal difficulty:

- the critical measurement dimension is selected by `randomSeed % 5`;
- the slope and offset are generated per seed;
- the available measurement families include density, temperature, quantum
  size, radiation, and a spectral channel.

Therefore seeds 0-4 deliberately vary the relevant measurement dimension while
preserving a higher-level scientific family. This is useful for ECSA: an exact
seed-specific mechanism should **not** be blindly reused in a new seed, but the
previous mechanism can serve as a transfer candidate that reduces subsequent
search.

## 3. Chosen architecture

### 3.1 DiscoveryWorldEnvironmentAdapter

Add a benchmark adapter around the official `DiscoveryWorldAPI`.

Responsibilities:

- instantiate `DiscoveryWorldAPI`;
- call `loadScenario("Reactor Lab", "Normal", seed, 1)`;
- expose the agent-visible observation from `getAgentObservation(0)`;
- expose `listKnownActions(limited=False)`;
- expose teleport locations;
- execute one JSON action with `performAgentAction(0, action)`;
- call exactly one `tick()` after every call to `performAgentAction()`, including actions that DiscoveryWorld reports as failed or invalid;
- expose terminal state via `areTasksComplete()`;
- expose step count via `getStepCounter()`.

The adapter must not read hidden object properties or scenario `scoringInfo`
during an agent run.

### 3.2 Information firewall

DiscoveryWorld exposes privileged evaluator information through
`getTaskScorecard()` and can export complete world history.

Those are **evaluator-only**.

The following may be supplied to the agent/policy:

- normal `getAgentObservation()` output;
- public action descriptions;
- teleport locations;
- result of its own previous action;
- ECSA scientific memory/proposals derived only from prior agent-visible
  evidence.

The following must never be supplied to the policy, scientific kernel, repair
engines, transfer admission, or mechanism retrieval during the run:

- `criticalQuestions`;
- `criticalHypotheses`;
- scorecard subgoals;
- hidden `scoringInfo`;
- crystal hidden attributes;
- hidden correct reactor frequencies;
- exported world history.

The scorecard and world history may be read **after the run** for evaluation
and debugging only.

## 4. Agent/controller boundary

Do not make the benchmark harness depend on one LLM provider.

Define a policy protocol conceptually equivalent to:

```python
class DiscoveryWorldActionPolicy(Protocol):
    def decide(
        self,
        observation,
        available_actions,
        teleport_locations,
        scientific_context,
    ) -> PolicyDecision: ...
```

`PolicyDecision` contains:

- one DiscoveryWorld JSON action;
- optional free-form reasoning/memory for logs;
- zero or more structured scientific hypotheses.

Cold and reuse arms must use the **same policy implementation, model,
temperature, prompt template, limits, and action budget**. The only allowed
difference is the mechanism memory visible through ECSA.

The arena package must not fork or copy the official HypothesizerAgent into the
scientific core. Its prompt/agent code is useful prior art, but the ECSA
environment and policy boundaries remain separate.

## 5. Reactor Lab scientific sidecar

The first arena needs a benchmark-specific sidecar that converts public
interaction evidence into structured scientific records without reading the
oracle state.

### 5.1 Measurement evidence

Record successful instrument-on-crystal actions.

The upstream tools produce public textual results such as:

- densitometer: density value;
- thermometer: temperature value;
- spectrometer: per-channel spectrum values;
- microscope: microscopic description containing quantum-gap information;
- radiation meter: radiation reading.

The sidecar stores:

```text
ReactorMeasurement
  step
  seed/context_id
  crystal public identity/UUID
  instrument public identity/UUID
  measurement kind
  parsed value(s)
  raw action-result message
```

Parsing is restricted to text the agent actually receives.

### 5.2 Structured mechanism hypotheses

A policy may propose a structured Reactor Lab mechanism:

```text
ReactorMechanismHypothesis
  functional_family = LINEAR
  measurement_kind
  slope
  offset
  source_evidence_ids
  prospective_predictions
```

The sidecar freezes prospective predictions before the relevant reactor
outcome is observed.

A hypothesis is not admitted because the policy stated it.

### 5.3 Qualification

A Reactor Lab mechanism may become a reusable ECSA mechanism only if its
prospective prediction is subsequently validated by agent-visible environment
behavior.

For the first implementation, successful activation of the target crystal
reactor at a previously predicted frequency is the decisive prospective
validation event. Validation is detected only from the next agent-visible
observation (for example, the public reactor name/state becoming
`crystal reactor (activated)`) after the action and tick. The sidecar must
not inspect the hidden `isActivated` or crystal `resonanceFreq` attributes.

The source run stores the resulting representation as an ECSA
`SYMBOLIC_RULE` mechanism artifact. The benchmark context ID is canonical:

```text
discoveryworld:reactor-lab:normal:seed-<N>
```

The mechanism record uses:

```text
domain_id = discoveryworld
task_id   = reactor-lab
regime_id = normal
context   = the exact seed context above
```

and carries:

- exact structured rule;
- source measurement evidence;
- frozen prospective prediction;
- validation action/result;
- explicit DiscoveryWorld context ID;
- benchmark/scenario provenance.

No oracle scorecard value participates in admission.

## 6. New transfer-candidate boundary

The current `MechanismRepository.find_applicable()` is intentionally
conservative: an unseen context is not applicable by default.

That means cross-seed transfer must **not** bypass applicability by pretending
that a seed-0 mechanism is already valid in seed 1.

Add a separate read-only transfer-candidate query:

```python
find_transfer_candidates(context)
```

Semantics:

- only `ADMITTED` non-deprecated mechanisms;
- domain/task/regime/assumptions must be explicitly present in the query and
  compatible with the stored scope;
- current context membership is deliberately ignored for candidate retrieval;
- results are labelled transfer candidates, never directly executable facts;
- candidates retain their validated source scope and provenance.

A transfer candidate can influence experiment priority or policy context, but
cannot be projected as an applicable runtime mechanism until target-context
validation succeeds.

After validation in the target seed, ECSA creates a new target-context
mechanism/specialization or other explicitly scoped record. It does not mutate
the source version in place.

This preserves the existing rule:

> Unknown transfer requests validation; it is not optimistic reuse.

## 7. Arena protocol

Use **progressive transfer** in the official seed order 0 → 4. This avoids
future-seed leakage.

### 7.1 Cold arm

For every seed:

- create a fresh empty MechanismRepository;
- run the same policy with no prior mechanism memory;
- record results;
- discard that repository before the next seed.

### 7.2 Reuse arm

Use one persistent MechanismRepository across seeds:

```text
seed 0 -> discover/validate/admit
            |
seed 1 <- transfer candidates from seed 0
        -> validate/admit target specialization
            |
seed 2 <- memory from earlier seeds
...
seed 4
```

Seed 0 is an anchor: cold and reuse both start without prior memory.

Primary transfer comparisons therefore use seeds 1-4.

No mechanism learned from a later seed may be visible to an earlier seed.

### 7.3 Determinism and pairing

For every seed, cold and reuse runs use identical:

- DiscoveryWorld commit;
- scenario/difficulty/seed;
- policy/model configuration;
- prompt template;
- maximum steps;
- model temperature/randomness configuration;
- action set.

Each result row must identify its paired cold/reuse run.

A later robustness study may repeat the experiment under several seed orders,
but that is not part of the first arena.

## 8. Metrics

### 8.1 Official DiscoveryWorld metrics

Collected after each run:

- `completedSuccessfully`;
- `scoreNormalized`;
- total environment steps.

The detailed scorecard remains evaluator-only.

### 8.2 ECSA transfer metrics

Record:

- number of scientific instrument `USE` actions on crystals;
- number of distinct crystal/instrument measurements;
- number of reactor-frequency adjustment/dialog actions;
- step at which the first qualified target-context mechanism is obtained;
- number of transfer candidates retrieved;
- number of transfer candidates tested;
- number accepted/rejected;
- number of new mechanisms discovered from scratch;
- number of mechanisms reused after target validation;
- false-transfer events;
- mechanism scope changes/specializations;
- model/policy token usage when available;
- wall time.

Primary measures:

```text
MeasurementTransferGain(seed)
  = 1 - reuse_measurement_actions / cold_measurement_actions

StepTransferGain(seed)
  = 1 - reuse_environment_steps / cold_environment_steps
```

Aggregate these over seeds 1-4.

A successful first result requires reporting the individual per-seed values in
addition to the mean. With only four transfer seeds, averages alone are
insufficient.

## 9. What counts as a false transfer

A false transfer occurs when prior mechanism memory causes the agent to commit
to a target-context scientific claim or reactor prediction that fails
prospective validation.

Merely retrieving a transfer candidate is not a false transfer.

The arena must separately count:

- candidate retrieval;
- candidate testing;
- candidate rejection;
- accepted transfer;
- false accepted transfer.

This is necessary because aggressive reuse can reduce measurements while
increasing incorrect commitments.

## 10. Logs and artifacts

Each run writes a self-contained result directory containing:

```text
run.json
actions.jsonl
observations.jsonl
scientific_events.jsonl
mechanism_events.jsonl
metrics.json
final_scorecard.json
```

`final_scorecard.json` is clearly marked evaluator-only.

Optional post-run diagnostics may call
`api.world.exportWorldHistoryJSON(...)`, but those files live under an
`oracle/` subdirectory and must never be consumed by the agent.

Each `run.json` includes:

- ECSA git revision;
- DiscoveryWorld git revision;
- scenario;
- difficulty;
- seed;
- arm;
- policy/model configuration hash;
- maximum steps;
- mechanism repository identity;
- timestamps.

## 11. Package/layout

Planned ECSA layout:

```text
src/ecsa/benchmarks/discoveryworld/
    environment.py
    contracts.py
    reactor_lab.py
    arena.py
    metrics.py

tests/
    test_discoveryworld_environment.py
    test_discoveryworld_reactor_lab.py
    test_discoveryworld_transfer_candidates.py
    test_discoveryworld_arena.py
```

The DiscoveryWorld dependency is optional and commit-pinned.

The benchmark adapter must not become a dependency of `ecsa.science`,
`ecsa.mechanisms`, or other core modules.

## 12. Testing strategy

### CI tests

CI does not run a paid/full LLM benchmark.

It does run real DiscoveryWorld:

1. pin/import the official repository;
2. load `Reactor Lab / Normal / seed 0`;
3. obtain a real observation;
4. verify every `performAgentAction()` call, successful or not, is followed by exactly one tick;
5. verify oracle information is absent from the policy-facing observation;
6. test parsing of real public instrument result messages;
7. test transfer-candidate retrieval does not make the candidate applicable;
8. test cold/reuse arena pairing with a deterministic test policy;
9. verify evaluator scorecard access occurs outside the policy-facing path.

### Manual benchmark command

The finished arena provides a command equivalent to:

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

The arena loads the policy through a Python import string in
`module:factory` form. The factory receives the parsed JSON config and
returns a `DiscoveryWorldActionPolicy`. The same factory path and the same
config-file content hash must be used in both arms. Provider/model-specific
code therefore stays outside the benchmark core.

## 13. Approaches considered

### A. Progressive transfer with ECSA sidecar — selected

Advantages:

- no future-seed leakage;
- directly tests persistent scientific memory;
- preserves DiscoveryWorld's native interaction loop;
- makes false transfer measurable;
- does not require changing the benchmark environment.

Cost: requires a Reactor Lab evidence/hypothesis adapter and a
transfer-candidate repository query.

### B. Leave-one-seed-out cross-validation

For each target seed, preload mechanisms learned on the other four seeds.

Advantage: all five seeds can be target seeds with a mature library.

Rejected for the first arena because it lets future benchmark seeds train the
memory and is less representative of an online scientific agent.

### C. Scripted scientific executor

Automate navigation, instrument use, and reactor controls, leaving only
mechanism inference to ECSA.

Advantage: cheaper and easier to reproduce.

Rejected as the primary arena because it removes too much of the agentic
environment. It may be useful later as a debugging/ablation mode.

## 14. Non-goals

The first DiscoveryWorld arena does not:

- modify DiscoveryWorld's world-generation logic;
- read hidden world state to help the agent;
- use scorecard `criticalQuestions` for mechanism admission;
- claim that a source-seed law is applicable in a new seed without validation;
- run all eight DiscoveryWorld science themes;
- test Challenge difficulty;
- replace DiscoveryWorld navigation/action mechanics with a synthetic API;
- require every existing ECSA repair engine to participate in every step;
- claim reduced environment sample complexity from the already-demonstrated
  DreamCoder enumeration speedup.

## 15. Acceptance criteria

The implementation is ready for the first real arena when:

1. the pinned DiscoveryWorld Reactor Lab Normal environment runs through the
   ECSA adapter for official seeds 0-4;
2. policy-facing data contain no oracle scorecard/world-state leakage;
3. every action is followed by exactly one environment tick;
4. Reactor Lab instrument observations can be recorded as scientific evidence;
5. structured hypotheses can freeze prospective reactor-frequency predictions;
6. successful target activation can qualify a mechanism without oracle data;
7. source-context mechanisms are persisted through the existing
   MechanismRepository;
8. unseen-context mechanisms are retrieved only as transfer candidates, not
   as applicable facts;
9. target validation creates explicitly scoped target-context knowledge;
10. the arena runs paired cold/reuse arms in progressive seed order;
11. output includes official scores plus measurement/step/transfer metrics;
12. a deterministic no-LLM test policy can execute the harness in CI;
13. the full paid/model-backed benchmark is runnable by one explicit command;
14. raw per-seed results are retained, not only aggregates.

The first scientific analysis will compare cold versus reuse on seeds 1-4 and
report both benefit and false-transfer cost.
