# Dual-encoder KAN implicit relation discovery — experiment 2026-10-08

## Architecture

Two independently trained encoders supply the same relation head:

- X -> E_x(X) = z_x
- Y -> E_y(Y) = z_y
- Relation head R(z_x, z_y) scores compatibility (not direct X -> Y regression)

The experimental KAN has trainable first-degree, fixed-grid B-spline edge functions. It is not a complete pykan/MultKAN implementation. An identically trained MLP relation head is the baseline. Three KAN architectures with different grids and widths form structural hypotheses.

## Protocol

- Synthetic hidden laws: smooth additive; multiplicative; implicit two-branch Y^2 = 0.35 + 0.5 X1^2 + 0.3 X2^2.
- Fixed seeds 0,1,2; 128 observed positive pairs per law and seed.
- 200 optimizer steps per initial model, with shared training budget and AdamW settings.
- 160 held-out positive and 160 oracle-filtered mismatched pairs per task and seed. Test outcomes are never used for training.
- X train domain [-0.9,0.9]^2, test domain [-1.05,1.05]^2 (mild shift).
- Training negatives are shuffled observational Y; NO hidden-law oracle is consulted for training negatives.
- Total parameters including separate encoders: MLP 505; single KAN 506; structural KAN ensemble members 509/506/494.

Mean AUROC across three seeds:

| Law | MLP | KAN | 3-KAN structural ensemble |
| --- | ---: | ---: | ---: |
| Additive | 0.9978 | 0.9997 | 0.9992 |
| Multiplicative | 0.9412 | 0.9842 | 0.9902 |
| Implicit two-branch | 0.6059 | 0.9054 | 0.8629 |

Active pair-query experiment: 12 queries selected as a single batch either by binary ensemble EIG or uniformly at random. Both arms use exactly the same candidate pool, starting seeds, observation and retraining budget. Across all nine law/seed combinations:

- EIG batch: AUROC 0.9512
- Random batch: AUROC 0.9570

**No EIG advantage was established.** Compatibility labels for the queried (X,Y) pairs come from a benchmark-only membership oracle. This is a stronger interface than intervening on X and observing the resulting Y. Equal-weight ensemble scores also are not a calibrated posterior over possible mechanisms.

The X/Y-shuffled negative control lost discriminative accuracy (AUROC approximately chance or worse). Both encoders had nonzero latent spread in the tested runs.

## What this does NOT establish

- No symbolic formula was automatically recovered. The hidden equation is only in the benchmark oracle.
- Recognition of compatible pairs does not demonstrate causal identification, counterfactual robustness or OOD transfer beyond the modest tested range.
- A membership-query batch is not a closed-loop sequence of physical interventions.
- Results apply only to these synthetic cases. KAN is not universally better than MLP.

## Reproduce

Install optional dependencies:

    pip install -e '.[test,neural-relations]'

Run:

    python -m pytest -q tests/test_relation_discovery.py
    python -m ecsa.experimental.relation_discovery --seeds 0 1 2 --steps 200 --train-pairs 128 --test-pairs 160 --active-queries 12 --active-steps 55 --output results.json

Next gate: formulate a calibrated distribution over Y after an intervention on X, update ScienceKernel posteriors sequentially, and test strong distribution shift and actual symbolic invariant extraction.
