# DiscoveryWorld: self-induced observational interfaces for Action-JEPA

Date: 2026-10-09 · Status: experimental first vertical slice

## Research claim and non-claims

An ECSA agent should not be handed `Reactor`, `crystal`, `instrument`, or
`MeasurementKind` as the semantics of the world. Its semantic interface must be
*proposed from observable transitions*, revised when contradicted, and tested on
independent evidence. The official DiscoveryWorld API remains the fixed **wire**
protocol: JSON actions, observation records and public object UUIDs are observed
syntax, not automatically correct scientific hypotheses.

An action contract consists of a discovered action symbol and argument fields,
observed argument-to-object bindings, typed public effects, optional categorical
precondition candidates from both successful and failed actions, and provenance.
All are hypotheses. Offline replication can grant `SUPPORTED` / `REJECTED` /
`INSUFFICIENT` status, **never** `ADMITTED` as a causal law. Promoting a causal
mechanism requires prospective, repeated interventions or an equivalent
identification argument in an *online* arena. Do not use evaluator hidden state.

## Existing interfaces retained

- `ReplayTransitions`, `ReplaySplits` and `train_jepa` stay intact.
- `ScienceKernel`, `WorldModelAcquisitionKernel`, and `ContractExperimentCoordinator`
  remain the authoritative scientific core. This experimental learner produces
  candidate projections, not a competing implementation of scientific admission.
- The existing `--representation public_hash` path remains the default and baseline.
- `--representation induced_interface` fits the interface only on training logs.
  Across official seeds 0–4: train 0–2; validate on 3; test on 4.

## Method

1. Read ONLY `run.json`, `actions.jsonl`, `observations.jsonl`, and if present
   `action_outcomes.jsonl`. Reject oracle/scorecard fields at any nested depth.
2. Associate duplicate public object records by UUID only if compatible; use
   transport IDs to match observations or bind action arguments, NEVER as feature
   magnitudes or memorized object identities. Unknown identity remains unknown.
3. Induce each action's observed parameter names and role-binding rates.
   Generate typed effect proposals from changed entity fields **and public
   non-object readings/text messages**, handling unidentified lists as
   permutation-invariant multisets; no domain role map.
4. If both successful and failed examples are available, propose categorical
   preconditions that distinguish outcomes. Other preconditions (numeric
   thresholds, nonlocal relations) are not yet covered.
5. Keep independent evidence IDs for each prediction and validation cohort;
   prevent training/validation overlap. Classify observed predictive replication
   separately from causal confirmation.
6. Fit a frozen train-only feature vocabulary; encode `E(o)` without access
   to action `a`. Encode `a` and its bound-object context in a separate vector.
   This preserves action-independent JEPA target coordinates.
7. Permit online update of the candidate learner with a bounded recent window.
   Preserve immutable old contracts and create a new proposed revision, with
   explicit `requires_independent_validation=True`.
8. Compare normal JEPA, shuffled-action JEPA, no-action JEPA and Raw+Ridge.
   Score every observed alternative *type* and per-action macro averages, while
   labeling the result **observational**, because alternative argument packets
   have not been tested for applicability in the same state.
9. Add a separate frozen-latent readout to **common public after-before feature
   deltas**, using training examples only; compare to same-coordinate persistence
   and Raw+Ridge. Report latent effective rank and warnings for low rank.
   A zero-change public target gets null, not a fake +100% improvement.

## Acceptance and limitations

- TDD tests prove induction with unknown action names and object fields, UUID
  permutation invariance, detection of changed contracts, disjoint seed
  validation, no oracle exposure, and the action-independent JEPA target.
- Tests ensure all observed alternative action types are scored, not just the
  first alphabetic alternative, including a per-type breakdown.
- DiscoveryWorld integration and actual task success require **real benchmark
  logs and a later paired interactive policy**. The current implementation is
  an offline experimental representation and contract learner, not an autonomously
  solving DiscoveryWorld agent.
- Published source hash features can still collide; aliases without UUID cannot
  be reliably grounded. A hard-coded *wire* UUID field is a transport hint;
  general identity induction from pixels or text is not solved here.
- Independent heldout replication is not causal identification. A scientific
  precondition from correlated successes/failures can be confounded.
- Inability to represent unseen features at heldout time is explicitly logged,
  not patched by fitting on the test seed.
