# Evolutionary KAN hypothesis search: matched interventions

Date: 2026-10-08. Status: research-only experiment, no production ScienceKernel changes.

## Hypothesis and design

The original two-sided architecture is retained: X -> Encoder X, Y -> Encoder Y, then a trainable spline-KAN implicit-relation head measuring pair compatibility. No Y is supplied before the experimental action.

Primary four-arm design:
1. Frozen KAN ensemble + random do(X).
2. Frozen KAN ensemble + maximization of ScienceKernel expected information gain (EIG).
3. Evolving KAN ensemble + random do(X).
4. Evolving KAN ensemble + adaptive EIG when validation-based reliability clears a predefined threshold, otherwise random, with a 20% random floor.

An additional **retrain-only random** diagnostic receives the same online observations and mutation-generation training steps as arm 3, but makes no parameter or structural perturbations. This is essential to distinguish mutation effects from online fine-tuning.

- At initialization and at step 4 and 8, evolution considers three children and retains two top validation-ranked unmodified parents and the strongest child. Mutation types alternate between Gaussian parameter perturbations and changes to the spline knot count / hidden width; encoder weights are copied, overlapping spline functions are interpolated. Each proposed child gets 32 optimization steps on the initial training partition plus observations from interventions already executed.
- Warmup: 64 observations, with 16 reserved from training for calibration, identical for all strategies. Three structural KAN hypotheses start from identical weights per seed. Each arm has 12 world interventions, 24 candidate X values per step, identical candidate pools, and 96 independent held-out observations. Laws: additive, multiplicative, implicit two-branch. Seeds 0 through 11.
- Outputs: heldout categorical NLL, multiclass Brier, top-class ECE, actual and nominal probability of an 80% prediction set, every selected X and observed Y, per-intervention EIG and posterior, mutation kind / lineage, epoch IDs and compute counts.
- Selection reliability gate: ensemble calibration NLL on warmup-heldout data must beat the training-Y empirical marginal by at least 0.08 nats, with prospective EIG >0.005 bits. Otherwise random; even when gate is passed, reserve 20% random exploration.
- New model generations receive new theory IDs and a fresh empirical validation-weighted **pseudo-prior**. Online observations used for refitting are NOT replayed as independent evidence. Within an epoch, the existing ScienceKernel updates the posterior sequentially. This is not a coherent cross-generation Bayesian posterior.

## Local 12-seed results: 144 primary runs, plus 36 extra controls

Mean heldout categorical NLL (lower better):

| Law | Frozen Random | Frozen EIG | Evolving Random | Evolving Adaptive EIG | Retrain-only Random |
| --- | ---: | ---: | ---: | ---: | ---: |
| Additive | 0.9412 | 0.9569 | 0.7423 | 0.7818 | 0.7999 |
| Multiplicative | 2.3822 | 2.4332 | 2.2984 | 2.2781 | 2.2883 |
| Implicit two-branch | 2.7176 | 2.7197 | 2.6060 | 2.6303 | 2.6028 |
| **All** | **2.0137** | **2.0366** | **1.8822** | **1.8968** | **1.8970** |

Mean multiclass Brier over all laws: fixed Random 0.7441, fixed EIG 0.7485, evolving Random 0.7028, evolving adaptive EIG 0.7078. Actual vs predicted coverage of 80% prediction regions: fixed Random 0.878 actual / 0.845 nominal; evolving Random 0.840 actual / 0.864 nominal. Calibration remains imperfect and law dependent.

Mean number of active EIG selections per 12 interventions: frozen EIG 11.94; evolving adaptive EIG 9.06. The validation reliability gate passed on 11.78/12 steps on average, so it was relatively permissive in these tasks.

Paired 95% cluster bootstrap confidence intervals over **12 seed clusters** (within each seed, average across the three laws), in final heldout NLL:
- Evolving Random minus Fixed Random: -0.1314, 95% CI [-0.2105, -0.0624]; 11/12 seed clusters improved.
- Evolving Adaptive EIG minus Evolving Random: +0.0145, 95% CI [-0.0434, +0.0666]; no EIG advantage established.
- Evolving Random minus Retrain-only Random: -0.0148, 95% CI [-0.0549, +0.0186]; **no isolated benefit of random mutation established**.

Results favor **online hypothesis updating and reassessment**, not a claim that randomness or EIG is intrinsically superior. Relative to the frozen model, the evolving and retrain-only arms use substantially more compute (288 additional optimizer steps per trajectory vs 0). They have equal environment-query budget but are not compute-budget-matched with frozen controls; the retrain-only comparison matches additional optimizer steps.

## Limitations / next gate

- Evaluation uses three uncomplicated synthetic structural laws; not image observations, real causal confounders or unknown physical mechanisms. No symbolic equation was extracted.
- Conditional Y probabilities are constructed from a discriminative relation model and training-only Y marginal. Calibration thresholds are predefined but tested on only 16 calibration examples and reused over generations; selection can overfit.
- At each generation, posterior is reinitialized from validation scores; although real observations update ScienceKernel between version changes, these are NOT posterior odds over fixed model families across the whole trajectory.
- Confidence intervals are exploratory, over 12 seed clusters within the same three synthetic problems. Larger independent laws and intervention tasks remain necessary.
- Need a compute-matched, no-mutation/no-online-data control, independent calibration slices per epoch, and a versioned hypothesis ledger with prequential evidence to claim stronger benefits.

## Reproduce

    pip install -e '.[test,neural-relations]'
    pytest -q tests/test_evolutionary_relations.py
    python -m ecsa.experimental.evolutionary_relations \
      --laws additive multiplicative implicit \
      --seeds 0 1 2 3 4 5 6 7 8 9 10 11 \
      --bootstrap 64 --steps 140 --interventions 12 --heldout 96 \
      --candidates 24 --mutation-steps 32 --include-retrain-control \
      --output evolution-results.json

See GitHub Actions neural-evolution-four-arm for parallel per-law reproductions and artifacts.
