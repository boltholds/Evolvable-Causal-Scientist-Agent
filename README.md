# Evolvable Causal Scientist Agent

An experimental architecture for an agent that learns unknown environments through intervention, maintains and revises competing theories, discovers reusable mechanisms, and uses the resulting causal knowledge to act toward goals.

The project is built around **integration of existing research systems rather than reimplementing them**.

## Architecture at a glance

- **PiEvo-derived scientific kernel** — Bayesian beliefs over an expanding theory space, information-directed experiment selection, and anomaly-driven theory-space growth.
- **Behavioral-Causal-State (BCS)** — exact predictive, interventional, and counterfactual inference for supported causal models. The existing `bcs.inference.evaluate()` remains the causal computation path; there is no separate `BCSTheoryPredictor` algorithm.
- **Robot Scientist** — reference model for the overall hypothesis → experiment → evidence → revision scientific cycle.
- **LearnLib TTT + mut-learn** — active finite-state behavioral discovery and counterexample generation.
- **DreamCoder** — program synthesis for candidate mechanisms.
- **Stitch** — abstraction/library learning over discovered programs.
- **VARIO** — shared-vs-context-specific mechanism analysis and transfer boundaries.
- **BOCPD** — online regime/change-point detection.
- **MMAE / IMM** — online comparison of predefined dynamical/model hypotheses.
- **Fault Diagnosis Toolbox** — contradiction/fault-oriented diagnostic proposals.
- **VACERL + CausalExploration** — candidate causal structure, significant-event discovery, and active causal exploration.
- **BAMCP / pymdp** — planning and action selection under model uncertainty.
- **Fast Downward / TheoryCoder-style PDDL projection** — symbolic and hierarchical goal planning.
- **DreamerV3** — learned latent dynamics and imagined trajectories for environments where exact symbolic rollout is insufficient.
- **DiscoveryWorld** — pinned external scientific-agent environment for the first real cold-vs-reuse transfer arena.

These components are not being discarded when another component overlaps with them. The architecture assigns each method a specific responsibility and removes duplicated ownership.

## Core principle

The system separates three concerns:

1. **Scientific coordination** — which theories remain plausible, which experiment should be performed next, and when the current theory class is insufficient.
2. **Causal semantics** — what a concrete model predicts observationally, under intervention, and counterfactually.
3. **Specialized discovery and planning engines** — finite-state learning, program synthesis, abstraction learning, regime detection, diagnosis, causal-structure discovery, uncertainty-aware planning, and neural dynamics.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the architecture contract, [docs/INTEGRATION_PLAN.md](docs/INTEGRATION_PLAN.md) for the staged implementation plan, and [docs/MECHANISM_LIBRARY.md](docs/MECHANISM_LIBRARY.md) for the reusable-mechanism schema, admission, applicability, and persistence boundary.

## Current executable milestone

The executable path in `main` now includes:

- immutable theory / experiment / observation contracts;
- Bayesian posterior updates over competing theories;
- exact expected information gain for discrete experiment outcomes;
- automatic selection of the most discriminating experiment;
- explicit `PopulationAnomaly` when the whole current theory population assigns zero support to an observation;
- a thin adapter into the existing Behavioral-Causal-State `evaluate()` API;
- `TheorySpaceExpansionRequest` and multi-engine repair fan-out with provenance;
- a real **LearnLib TTT** repair backend that reuses the existing BCS bridge at the pinned BCS revision;
- content-addressed persistence of the accepted Mealy hypothesis;
- TTT residual-state projection into a typed state-augmentation candidate;
- an aliasing qualification gate: the same causal-state key must map to distinct minimal TTT residual states before the state feature is admitted;
- coherent admission of a qualified projected theory with explicit small prior mass;
- deterministic TTT predictions for supported frozen interventions in the same `ExperimentSpec` / `PredictiveDistribution` space used by BCS theories, allowing the projected theory to participate in IDS;
- a real **BOCPD** regime-change repair backend, pinned to `fiannai/bocd`, operating on population predictive discrepancy rather than raw observations;
- confirmation gating for regime changes: a run-length reset must be followed by a nontrivial new segment, so a single terminal anomaly does not automatically become a regime-change theory;
- content-addressed BOCPD diagnostic artifacts containing the signal, MAP run-length trace, reset step, and detector parameters;
- a real **DreamCoder** program-repair backend, pinned to `ellisk42/ec`, using DreamCoder's own `Grammar`, `Program`, `Task`, type system, and `Grammar.enumeration()` core;
- an isolated DreamCoder worker that bypasses only the legacy eager package initializer, avoiding its unrelated historical domain dependencies while preserving the original enumerator;
- a first explicit Boolean mechanism domain adapter with provenance-bearing examples and a bounded DSL;
- content-addressed DreamCoder program artifacts carrying the synthesized program, MDL/log prior, DSL, examples, enumeration count, and pinned source revision;
- a DreamCoder program prediction adapter that evaluates synthesized programs through the same pinned DreamCoder `Program.parse/evaluate` runtime and emits ordinary `PredictiveDistribution` values;
- a prospective held-out qualification gate: synthesis evidence and validation evidence must be disjoint before a program can become a `TheoryRef`;
- coherent small-prior admission of qualified program mechanisms into the same posterior as BCS/TTT-derived theories;
- information-directed selection over experiments where causal and synthesized-program theories disagree;
- a real **Stitch** abstraction-learning backend, pinned to the MIT Rust core at `mlb2251/stitch`;
- Stitch input is restricted to `QualifiedProgramMechanism` values that already passed prospective validation; raw DreamCoder synthesis outputs are not eligible for library learning;
- content-addressed Stitch abstraction artifacts carrying the learned body/arity, utility, compression ratio, number of uses, original programs, rewritten programs, and exact source mechanism provenance;
- a cross-mechanism reuse gate: one qualified mechanism alone yields no reusable abstraction proposal;
- an **official VARIO** transfer-validation backend using the authors' August 2022 R source archive, checksum-pinned at runtime rather than vendored;
- cross-context mechanism partitioning with the official `Vario_Pi_search` path and VARIO's own conservative invariance correction for the fully invariant case;
- an explicit ECSA transfer-scope projection from VARIO partitions: one context group → `SHARED`, multiple groups with reuse → `CONTEXT_SPECIALIZED`, all singleton groups → `SPLIT`;
- content-addressed VARIO transfer artifacts binding the Stitch abstraction, context evidence hashes, official source checksum, learned partition, MDL score, and ECSA scope;
- a fixed **Mechanism Library v1 architecture contract**: ECSA owns mechanism identity/version, kind, epistemic status, scope, assumptions, evidence, relations, admission, and applicability semantics, while persistence/lineage are delegated behind a `MechanismRepository` protocol;
- **ML Metadata (MLMD) 1.21** is the first persistence/lineage backend: `MLMDMechanismRepository` is implemented with SQLite acceptance coverage for admission, version lookup, applicability, exact input lineage, supersession, and immutable deprecation;
- executable **mechanism runtime projection** is integrated: admitted mechanisms are filtered by current context/regime/domain/task/assumptions before reaching execution backends;
- `DreamCoderGrammarProjector` resolves applicable qualified program mechanisms back to their exact DreamCoder program artifacts and injects them into the upstream Grammar as real `Invented` productions;
- typed runtime artifact views preserve separate `CAUSAL_MODEL`, `STATE_FEATURE`, and `SYMBOLIC_RULE` channels without pretending that BCS/planner artifact hydration is already implemented;
- an end-to-end reuse test requires warm DreamCoder synthesis in a second validated context to enumerate fewer programs than cold-start synthesis, establishing computational reuse without yet claiming reduced environmental experiment count;
- a pinned **DiscoveryWorld** backend at `allenai/discoveryworld@fd591323920be0d3786ef350955de1945aa571e5`, with a strict Reactor Lab / Normal environment adapter and one-action/one-tick semantics;
- an explicit DiscoveryWorld oracle firewall: normal observations/actions/teleports are policy-facing, while scorecards, critical questions/hypotheses, hidden world state, and exported history are evaluator-only;
- a Reactor Lab scientific sidecar that parses only public instrument-result text, freezes linear mechanism hypotheses prospectively, and qualifies symbolic mechanisms only from subsequent public reactor activation;
- conservative `find_transfer_candidates()`: only out-of-scope, non-`SPLIT`, admitted mechanisms with compatible regime/domain/task/assumptions can guide validation; mechanisms already applicable in the current context remain facts rather than transfer candidates;
- progressive cold-vs-reuse orchestration over official Reactor Lab Normal seeds 0→4, with cold repositories reset per seed and one persistent reuse repository;
- provider-neutral policy loading via `module:factory`, per-run JSON/JSONL logs, transfer/measurement counters, and deterministic configuration hashing; policies distinguish merely stated hypotheses from explicit `validation_hypothesis_ids`, so a false transfer is counted only after a source-backed hypothesis is deliberately committed to a prospective test and fails;
- a real no-LLM CI smoke runs both arms for all official seeds for one environment interaction each; this verifies benchmark wiring and information isolation, **not** transfer sample-efficiency. The 1000-step model-backed arena remains the experiment that must establish or reject that scientific claim;
- end-to-end tests using real BCS models, a real Java/LearnLib TTT learner, the external BOCPD implementation, the pinned DreamCoder core, the real Rust Stitch compressor, and the official VARIO R implementation.

The TTT integration does **not** vendor the BCS TTT implementation. The adapter checks out the pinned BCS source revision, invokes its existing bridge, and records the learned machine as an ECSA artifact.

A TTT machine is emitted as a `TheoryProposal` with `STATE_MEMORY` provenance. It is still **not auto-admitted merely because it was learned successfully**. The state projection must first produce an aliasing witness showing that the current causal-state key collapses behaviorally distinct minimal residual states. A qualified projection becomes a state-augmented `TheoryRef`, receives explicit small prior mass, and can make deterministic predictions for interventions represented in the frozen TTT alphabet.

The current qualification test deliberately uses a coarse synthetic causal-state key to verify this bridge. It does not yet claim that a real BCS state representation is insufficient on the frozen fixture; that scientific claim requires a separate arena experiment.

Run the full tests:

```bash
python -m pip install -e '.[test,bcs,bocpd,dreamcoder,mlmd,discoveryworld]'
# VARIO tests additionally require R and the R packages:
# rlist, combinat, dplyr, car
pytest -q
```

Run the minimal causal example:

```bash
python examples/first_vertical_slice.py
```

Run the DiscoveryWorld Reactor transfer arena with a configured policy:

```bash
python -m ecsa.benchmarks.discoveryworld.arena \
  --scenario "Reactor Lab" \
  --difficulty Normal \
  --seeds 0,1,2,3,4 \
  --arms cold,reuse \
  --max-steps 1000 \
  --output output/discoveryworld-reactor \
  --policy-factory package.module:create_policy \
  --policy-config policy.json
```

The policy factory receives the JSON config and must return a `DiscoveryWorldActionPolicy`. Cold and reuse arms use the same factory/config hash; only persistent mechanism memory differs.


TTT, BOCPD, and DreamCoder are now integrated as distinct repair paths: TTT proposes missing state/memory structure, BOCPD proposes temporal regime change, and DreamCoder proposes executable program mechanisms when the current mechanism form is insufficient. They can all be invoked from the same `TheorySpaceExpansionRequest` and return provenance-bearing proposals without overwriting the active theory population.

The current DreamCoder integration intentionally starts with one Boolean-domain adapter. A synthesized program is no longer admitted from training fit alone: it must make a correct prospective prediction on held-out evidence that was not used during synthesis. Only then is a content-addressed program-mechanism theory created and assigned explicit small prior mass. The adapter can subsequently compete in the same IDS experiment set as BCS theories.

This proves the repair → prediction → validation → admission loop for one domain, but it does not claim a universal DSL or automatic causal semantics for arbitrary synthesized programs.

Qualified program mechanisms can now enter Stitch. Stitch learns recurring lambda-calculus structure across those mechanisms and emits `StitchAbstractionProposal` values in the `ABSTRACTION_TRANSFER` family. These proposals are **candidate reusable mechanisms, not causal laws**: compression and multi-use support establish reusable structure.

VARIO now supplies the next gate. Given numeric evidence from multiple contexts, the adapter runs the official `Vario_Pi_search` implementation and records the discovered context partition. ECSA maps that partition to transfer scope (`SHARED`, `CONTEXT_SPECIALIZED`, or `SPLIT`). This scope vocabulary is ECSA's integration contract, not terminology claimed from the VARIO paper.

The official VARIO archive does not contain a license file. ECSA therefore does not vendor or redistribute its R source: the adapter downloads the authors' archive at runtime, verifies the frozen SHA-256, executes it externally, and records this licensing/provenance boundary in every transfer artifact.

The project still does **not** claim that the remaining external engines are integrated or interchangeable. Each one will be connected behind an explicit adapter and validated experimentally.

## License

MIT. External projects retain their own licenses; code must not be vendored or copied unless its license permits it.
