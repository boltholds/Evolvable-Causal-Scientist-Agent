# ECSA: dual-encoder neural relation hypotheses under real interventions

Date: 2026-10-08. Experimental, synthetic-world closed-loop evaluation.

## Research question

Can independently encoded X and Y feed competing KAN implicit-relation models,
which use actual do(X)->observed Y results to choose experiments and update
Bayesian model probabilities, outperforming random interventions and comparable MLPs?

The agent observes X candidates BEFORE deciding. There is NO
pair-membership / success-label query. After choosing an X, the world produces
a new stochastic Y. Only then does ScienceKernel update the posterior.

## Method

1. A hidden synthetic world emits observed Y for 64 initial do(X) trials.
   The model fitting uses 48 positive pairs and 16 additional calibration pairs.
2. Each hypothesis has independent X/Y encoders and an implicit binary
   pair-relation head. Three structural spline-KAN models are compared with
   three MLP models with different initializations, all using the same warmup.
3. Contrastive matched-vs-shuffled-Y training learns a density-ratio-like
   compatibility score. A training-only KDE estimate of q(Y), finite Y bins,
   positive probability floor and warmup-only temperature calibration give an
   approximate categorical predictive p(Y-bin | do(X), model).
4. ScienceKernel computes expected information gain over the observed Y bins
   for 24 proposed X interventions; the EIG arm chooses the best. A random
   arm uses the identical candidate pools and exactly the same 12 interventions.
5. The chosen X is executed; actual Y is observed; only then does
   ScienceKernel.update revise the model posterior.
6. Neural hypotheses remain FROZEN through the online experiment. New observations
   update only model weights; allowing model refits without revising the
   posterior ledger would invalidate its likelihood interpretation.
7. Predictive NLL is measured on 96 independent held-out do(X) observations,
   which are NEVER shown to the learner. In addition to KAN/MLP random and EIG,
   compare with a KDE nearest-neighbor conditional predictor and an unconditional
   histogram. Data/domain: three hidden law families, seeds 0/1/2, intervention
   range [-1.05,1.05]^2 versus warmup [-0.9,0.9]^2.

## Local CPU results (3 seeds per law, mean held-out categorical NLL; lower better)

| Law | KAN before | KAN EIG | KAN random | MLP EIG | MLP random | KNN reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| additive | 1.122 | 1.268 | 1.162 | 1.204 | 1.210 | 2.185 |
| multiplicative | 2.402 | 2.549 | 2.505 | 2.841 | 2.789 | 2.756 |
| implicit two-branch | 2.656 | 2.662 | 2.725 | 2.857 | 2.865 | 2.235 |
| average | 2.060 | 2.160 | 2.131 | 2.301 | 2.288 | 2.392 |

All 36 runs used 12 post-warmup interventions. EIG vs random:
KAN 2.160 vs 2.131; MLP 2.301 vs 2.288. **EIG does not win.**
KAN heads learn useful conditional relationships on additive and multiplicative
tasks, but in this protocol sequential reweighting frequently degrades the
initial equal-weight KAN mixture (7 of 9 EIG cases). The KNN reference
outperforms KAN on the two-branch implicit law.

This is an honest negative result for the proposed active-advantage
hypothesis, not a test infrastructure failure.

## Limitations

- Finite-bin conditional probabilities are derived from discriminative energies
  and a fitted Y reference marginal; their probabilistic calibration remains
  imperfect. Some held-out Y fall outside the finite grid (reported as clips).
- The Bayesian posterior is over candidate neural implementations, NOT a
  posterior over uniquely identifiable physical laws.
- The test world is deterministic/stochastic only through predefined simple
  synthetic mechanisms. do(X) fixes X, but there are no latent confounders,
  sequential planning requirements, or unknown object groundings.
- The KAN is an edge-wise first-degree spline network, not the full
  pykan/MultKAN library. No symbolic equation was extracted.
- 3 seeds, short horizon, modest distribution shift: no general transfer claim.
- Warmup sets the model before interventions; the active step does NOT fine-tune
  network parameters, only tests their forecasts and changes posterior weights.

## Reproduction

    pip install -e '.[test,neural-relations]'
    python -m pytest -q tests/test_intervention_relations.py
    python -m ecsa.experimental.intervention_relations \
      --seeds 0 1 2 --bootstrap 64 --steps 140 \
      --interventions 12 --heldout 96 --candidates 24 \
      --output results.json

## Next gate

Calibrate conditional likelihood and posterior reliability on independent
intervention observations; test a tempered and a hierarchical model prior
without tuning on heldout outcomes. Compare posterior predictive skill at
each intervention budget and require positive EIG-minus-random improvement
on heldout distributions across substantially more seeds before admission.
Then test a genuine streaming/refit design with a versioned hypothesis ledger.
