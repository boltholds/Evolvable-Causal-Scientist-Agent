# ECSA: blind recurrent Action-JEPA adaptation (v1)

**Scientific question.** Can a JEPA predictor, after detecting an unannounced causal shift and confirming an action-order hypothesis, acquire the changed dynamics without destroying its prior knowledge, and detect the return of an older mechanism?

## Scope

This is a small **synthetic evaluator-controlled** world with three public actions `pulse`, `flip`, `wait` and opaque 24-dimensional nonlinear noisy observations. It changes `additive -> gated -> additive` at two evaluator-only boundaries. A public reset between independent controlled trials does not reset the regime epoch. The learner and scientist never receive a mechanism label or boundary; the returned JSON includes them **for grading only**.

The learner is the existing 8-dimensional action-conditioned JEPA. We intentionally freeze encoder and EMA target during online repair and tune only the action-conditioned predictor: this creates a fixed, comparable latent coordinate system. This experiment tests continual prediction, **not** end-to-end encoder plasticity or faithful LeWorldModel SIGReg.

## Protocol

1. **Pretrain** one JEPA on 2,048 additive action/observation transitions (`smoke`: 256); fit a strong action-interaction raw Ridge baseline and an action-shuffled JEPA. Calibrate the change monitor on 1,024 independent additive transitions (`smoke`: 192).
2. **Observe without labels** a 240-transition stream (`smoke`: 144), with changes at evaluator indices 80 and 160 (`smoke`: 48 and 96), plus matched stationary additive placebo. The first action-stratified residual alarm triggers a pilot/independent confirmation experiment on the **frozen physics at the alarm index**. Early or unconfirmed alarms stop the protocol and are recorded as failures, never repaired using oracle labels.
3. **Acquire after confirmation:** acquire at most 64 new transition examples from that epoch for training, plus 256 separate observed changed-epoch calibration examples (`smoke`: 24+96) and a disjoint action-order evidence anchor. The calibration examples never update weights; they only provide a report-only learning curve and reference for a new residual monitor.
4. **Compare matched repairs:** freeze the pretrained checkpoint, gated-only predictor fine-tune, same-step rehearsal with at most 128 **prechange-training-only** examples (`smoke`: 64), and gated-only scratch training. Also report a raw Ridge re-fit from the same 64 new examples. All predictor update arms start from the identical pretrained checkpoint, use 180 optimizer steps (`smoke`: 64), and the same changed observations.
5. **Evaluate sealed cohorts:** independent additive and gated test transitions were never presented for optimizer updates, replay, calibration or experiment selection. Record action discrimination, latent prediction MSE, difference from frozen additive MSE (**forgetting**) and frozen gated MSE (**new acquisition**), effective rank, and first *report-only* validation update reaching 20% improvement. Scratch JEPA has its own latent frame and raw Ridge has observation-space units: compare those models' **action discrimination or within-model changes**, never raw cross-model MSE.
6. **Return detection:** rearm a new calibrated residual monitor **after the first confirmed experiment** on the observed changed-epoch calibration cohort. Use only the remaining public stream actions/observations, plus a no-return gated placebo. At any return alarm, test the **same previously confirmed action pair**, with a new observed return-epoch cohort and a one-sided permutation test for reduced effect relative to the independent changed-epoch anchor. A significant reduction combined with no currently supported order effect is *consistent with a return*, not a proof of restoring an identical causal model.
7. **Two-checkpoint recall:** independently evaluate a simple loss-minimizing selector across a saved prechange checkpoint and the adapted predictor, with no regime input. Its gated and additive-return scores are **retrospective evaluator checks**. Checkpoint routing is separate from within-single-checkpoint forgetting, and window-level disagreement must be recorded.

## Data and compute

Full protocol: 700 JEPA pretrain steps, 180 repair steps, 64 new adaptation transitions, 128 old replay examples, 256 independent observed calibration transitions, 16 pilot trials per candidate pair, 64 first/return confirmation trials, 4,095 permutation draws. The anchored changed-epoch evidence bank costs a further 64 paired commutator trials. Report **all** simulator actions, including calibration and confirmatory interventions, separately from the weight-update sample count. No reward, oracle law, or perfect mechanism ID is passed to the model.

```bash
python scripts/benchmarks/run_jepa_adaptation.py --mode smoke --seed 0 \
  --output results/jepa_recurrence_smoke.json
python scripts/benchmarks/run_jepa_adaptation.py --mode full --device cuda --seed 13 \
  --output results/jepa_recurrence_seed13.json
```

On CPU, set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` if desired; a 12 GB RTX 4070 Ti is much larger than this small MLP requires. `--adapt-steps` and `--budget` change the learning and acquisition budgets. Seed controls data sampling and initialization.

## Limits / what it cannot demonstrate

- It uses resettable, clonable synthetic labs; real equipment requires safe reset/intervention handling.
- Two scripted switches are known only to the evaluator; these synthetic laws, actions and sensor geometry are fixed. No transfer to unseen environments, robust causal identification, or latent regime discovery is established.
- The first detector can miss a shift or false-alarm; neither is silently corrected. The second monitor has only a finite-sample action-conditional bootstrap null, not an anytime guarantee.
- Repair trains only the predictor on a frozen embedding; this cannot prove end-to-end JEPA continual representation learning.
- The old snapshot is *checkpoint memory*, which may rescue return recognition even if the single adapted predictor forgot. These outcomes must not be conflated.
- A failed order-effect test is not proof of commutativity. Even significant effect reduction does not establish equality to the original causal mechanism.
- Models trained from scratch have differing embedding geometry; comparing their absolute latent MSE against the pretrained model is invalid.
- Passive stream steps, actively requested transition samples, calibration actions, and multi-step intervention trials have distinct costs. Count them all.
- A few seeds under one sensor geometry are a limited benchmark; robust results demand multiple independent environments and unseen geometries.

## Seed summaries

See the separate JSON reports for per-seed outcomes, counterfactual evidence IDs, fixed holdouts, and costs. **No success rate should be inferred from only triggered cases**: denominator includes missed shifts and disconfirmed early alarms. The protocol measures whether the JEPA improves and preserves known dynamics; it does not assume replay must succeed.

## Measured CPU results: 14 seeded full runs

The source files and tests were frozen and SHA-256 recorded **before** running heldout seeds 9–12; their hashes were verified unchanged afterwards. Seeds 0–8 and 13 are prior diagnostic runs, not blind postselection; seeds 9–12 are a small additional confirmation cohort. Sensor geometry is fixed across all seeds. See `2026-10-09-jepa-continual-adaptation-results.json` for per-metric values and audit the individual JSONs in the companion archive.

| Outcome | 14 full runs | Comments |
|---|---:|---|
| JEPA confirmed initial changed-order law | 12/14 | seed 7 missed, seed 8 raised early false alarm and was not repaired |
| Naive fine-tuning + causal return consistency check | 11/14 | seed 5 did not confirm return |
| Raw observation/Ridge initial change alarms | 14/14 | stronger initial detection baseline |
| No-change additive JEPA false alarms | 1/14 | monitored placebo, not hidden |
| No-return gated naive false alarms | 1/12 adapted runs | monitored placebo after calibration |
| No-return gated rehearsal false alarms | 2/12 adapted runs | monitoring only; no independent causal confirmation |
| Replay improves retention relative to gated-only | 12/12 adapted runs | paired comparison using the same frozen encoder |

Among the **12 conditionally adapted** runs (do not confuse with all 14):

| Heldout metric (fixed latent coordinates) | Naive fine-tuning | Replay |
|---|---:|---:|
| Mean additive forgetting: MSE increase vs frozen | +0.01566 | -0.00193 |
| Mean gated MSE reduction vs frozen | +0.02704 | +0.02035 |
| Gated action-choice accuracy | 90.95% | 90.80% |
| Additive action-choice accuracy | 93.77% | 98.34% |

For context, the **frozen** pretrained model already attains 90.59% gated action-choice accuracy on these heldout samples; the MSE improvement in the fixed embedding frame is more informative than the very small action-choice change. Scratch predictor coordinates are independently trained and thus have no aligned MSE. The replay model does **not** consistently issue an alarm when the law returns: strong retention may reduce its surprise, and the two-checkpoint router still makes window-sensitive decisions. Distinguish retention from explicit detection of a prior regime.

A successful first confirmation requires a new, heldout intervention; return confirmation re-tests the **original confirmed** action pair with disjoint changed-epoch anchor and return cohorts and demands a significant decline, while explicitly abstaining from claims of proving the exact old mechanism. Including 64 repair transitions, 256 changed-epoch calibration transitions, pilot/confirmation experiments and changed-epoch anchor, fully successful runs spend **1,760 active simulator steps** (seed 5 spent 1,376 because no second confirmation was run). This is much larger than the number of repair examples; reporting only 64 as the total budget would be misleading.

**Interpretation.** Predictor-only rehearsal protected prior action dynamics on this fixed-geometry synthetic environment. It did not establish end-to-end JEPA world-model adaptation, guaranteed return recognition, sample efficiency against a strong model-based baseline, or robust transfer to new sensors. Those remain separate experiments.
