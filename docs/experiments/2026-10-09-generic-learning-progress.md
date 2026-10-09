# ECSA: domain-neutral learning-progress controller

This is a **generic policy experiment**, not an assertion that Reactor Lab has been solved.

## Causal/epistemic boundary

A successful public action, a changing sensor image, a new observation ID, or a new unqualified contract is not counted as a scientific discovery. A failed action can reduce prediction uncertainty about the action's applicability, but does not prove a causal effect.

A Beta-Bernoulli action-outcome estimate is evaluated prequentially: its variance immediately before an action is compared with the variance after observing that action's actual success/failure. The difference represents provisional predictive information and diminishes with repeated identical outcomes. A configurable floor (`min_progress_gain`, default 0.002 in this first protocol) prevents near-zero variance changes from defeating stagnation detection.

No action names, visible feature names or benchmark-specific semantics are hard-coded in the core. Context comparisons use public *structural* feature signatures and may conflate hidden states. They must not be called latent-state equality.

## Algorithm and ablation

`ContractExperimentCoordinator(use_progress_scoring=True,progress=ledger)` activates an opt-in selector that uses **existing ScienceKernel EIG as lexicographically primary** when positive, and evidence-based progress/stagnation/observed intervention cost for otherwise uninformative candidates. A penalty is bounded: it never removes the only legal action, and it does not block independent, informative replications.

`use_progress_scoring=False` preserves the original ranking. Two anonymous in-memory RawEnvironmentPort fixtures with disjoint action vocabulary verify symbol independence, finite memory, behavior with no informative actions and equal number of actions across the two policy arms. Tests deliberately avoid any DiscoveryWorld import.

Known limitations: current causal-contract updates are not automatically certified. The progress ledger only stores recent observation IDs for deduplication; for long-duration exact provenance an external durable store is necessary. The positive EIG protection also means a miscalibrated nonzero EIG backend can keep selecting a poor action: this is a limitation to measure, not a reason to hard-code a name blacklist.

## Verification

- TDD RED: missing module `ecsa.world_model.learning_progress`, followed by success.
- TDD RED: missing `ecsa.world_model.progress_selection`, followed by success.
- TDD RED: unimplemented coordinator constructor/breakdown, followed by success.
- TDD RED: lack of prequential progress signal in real agent loop, followed by success.
- Last focused workflow: `universal-causal-core` 38 passed (includes legacy autonomy/world-model selection tests).
- A full repository test suite is a separate gate and must be reported by CI; the targeted 38 do not guarantee all optional library extras.

Next PRs: background/action attribution and canonical public outcomes, not benchmark-specific hacks.
