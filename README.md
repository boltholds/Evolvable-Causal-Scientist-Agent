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

These components are not being discarded when another component overlaps with them. The architecture assigns each method a specific responsibility and removes duplicated ownership.

## Core principle

The system separates three concerns:

1. **Scientific coordination** — which theories remain plausible, which experiment should be performed next, and when the current theory class is insufficient.
2. **Causal semantics** — what a concrete model predicts observationally, under intervention, and counterfactually.
3. **Specialized discovery and planning engines** — finite-state learning, program synthesis, abstraction learning, regime detection, diagnosis, causal-structure discovery, uncertainty-aware planning, and neural dynamics.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the architecture contract and [docs/INTEGRATION_PLAN.md](docs/INTEGRATION_PLAN.md) for the staged implementation plan.

## Status

Architecture v1 is being specified. The repository does **not** yet claim that the listed external systems are integrated or interchangeable. Each integration will be added behind explicit adapters and validated by end-to-end experiments.

## License

MIT. External projects retain their own licenses; code must not be vendored or copied unless its license permits it.
