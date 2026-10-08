# Action-JEPA continual adaptation: recurrent mechanism shift

## Scientific question
Can a *single pretrained* action-conditioned JEPA update its predictive dynamics after an unlabeled change, preserve its original dynamics through rehearsal, and recognize the return of a previously seen regime? The protocol must permit negative outcomes and must not use mechanism IDs or switch times as training signals.

## Decisions
- Keep the existing opaque sensor renderer, public action vocabulary and independent trials; a new **evaluator-only** world changes `additive -> gated -> additive` after fixed numbers of actions. Each `TransitionDataset` remains only `(before, actions, after)`.
- Pretrain one JEPA on additive transitions; retain an immutable checkpoint. The existing frozen residual monitor inspects a blinded additive->gated stream with independently calibrated thresholds. A *post-switch* alarm must be confirmed through the existing active pilot + heldout commutator test before adaptation begins. Early/unconfirmed alarms yield a negative outcome, not oracle-triggered adaptation.
- The learner receives only transitions sampled from the confirmed changed epoch (bounded N). Evaluate three *matched* subsequent paths: frozen checkpoint, gated-only predictor fine-tuning, and fine-tuning with a bounded **training-only** additive rehearsal buffer. An independently trained-from-scratch gated checkpoint and a strong raw Ridge are labeled secondary controls.
- Freeze the existing encoder and EMA target during predictor adaptation so latent distances remain comparable. This tests predictor adaptation, *not* joint representation plasticity. Keep adaptation update counts identical between replay and gated-only and expose memory/intervention budgets.
- Compute paired evaluation metrics on sealed additive and gated heldout cohorts using frozen coordinates: action ranking, latent MSE, noncollapse rank, gated recovery vs frozen, additive forgetting vs frozen, first validation update reaching 20% gating MSE reduction. Validation is *report-only*, never used for checkpoint selection.
- After adaptation, route between stored initial and adapted model by comparing average action-conditional predictive residuals in a fixed-length window. The mode selector sees neither mechanism label nor switch boundary; evaluator-only scoring checks recognition on fresh gated and additive-return transitions. This is **two-checkpoint recall**, distinguished from retention in a single updated checkpoint.
- After return, independently probe action order with a new evidence cohort. Failure to reject order effect is reported as inconclusive, not proof of commutativity.

## Success and falsification
One confirmed change; replay improves retained additive scoring vs gated-only *without destroying* gated acquisition on heldout; and mode selector chooses the old expert on return without ground truth. Report all seeds and failure rates. Compare against raw Ridge, frozen JEPA and shuffled-action controls. A successful code/test run does not establish that JEPA beats baselines or transfers to real systems.

## Safety and leakage boundaries
The evaluator alone reads `Mechanism` and change boundaries; training code never imports them. Calibration, adaptation and evaluation must be disjoint. Pilot and confirmation interventions must not overlap. Stored old memory comes solely from the additive training split. Explicitly count intervention actions; do not count passive stream steps as acquired active samples. Use deterministic seeded schedules and reproducible JSON metadata.
