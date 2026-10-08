# Action-conditioned JEPA to active causal commutator — design

## Motivation and intended outcome

ECSA should learn its own dynamic-state representation from observations and
actions. The user-approved research question is whether that representation
can support a *different* researcher discovering and testing causal laws.
No pretrained LLM latent state, raw hidden simulator state, symbolic predicate,
or teacher-forced law label is an allowable training input.

## Research question and hypotheses

- H0: learned latent representation is collapsed, does not predict the next
  state better than a shuffled-action diagnostic, or does not retain sufficient
  information to distinguish action-order effects.
- H1: online representation/predictor trained on self-supervised transitions
  supports heldout action-specific next-state prediction, retains variance,
  and a separate intervention-based researcher distinguishes commuting from
  noncommuting actions on new trials.
- H2 (added value): with noisy observations, learned latents allow the same
  predeclared decision procedure to distinguish mechanisms when raw and
  randomly initialized representations do not. This does **not** prove the
  baseline algorithms are incapable with other estimators.

## Candidate implementations considered

1. Frozen Qwen/BehR features plus trainable head: rejected for the main
   hypothesis, since it tests information *inside an existing LLM* rather
   than learned latent dynamics.
2. Full LeWorldModel SIGReg/video model: most faithful to the literature but
   far too costly/complex for a minimal causality experiment; reserve as a
   subsequent ablation.
3. Small EMA-target, action-conditioned JEPA with anti-collapse constraints:
   chosen because it trains CPU-local, preserves the JEPA prediction objective,
   and makes all controls tractable on one machine.

## System boundaries

Evaluator owns two worlds (additive/gated), a fixed opaque sensor mixing map
and randomized measurement nuisance. Learner sees only `(o, a, o')`.
Scientist receives only a resettable observation/action port and encoder (plus
predictor for proposing action pairs). It tests `a;b` against `b;a`, records
paired evidence and repeated-sequence null distance; it does not read world
mechanism enum or private state. The arena alone grades the prediction.

No changes to production `AutonomousScientist` or symbolic world-model
contracts in this first gate; the new typed protocols can be integrated only
once numerical evidence justifies that step.

## Testing and acceptance

TDD tests cover typed inputs, deterministic reset and fixture distinction,
transition dataset law-label exclusion, no latent collapse, finite training,
heldout action prediction, scrambled-action control, evidence provenance,
noise null control, all representations scored on the same observations,
active pair proposal, and CLI report generation. CPU smoke runs without model
downloads; full mode produces seed-indexed JSON reports.

Explicit limitations: synthetic mechanisms, sensor fixed across train/test,
separate model per mechanism, uncalibrated effect/noise ratio, action pair
known to primary evaluator, no general causal identification or TextWorld
success claim. Failures must be reported instead of tuned away after seeing
heldout results.
