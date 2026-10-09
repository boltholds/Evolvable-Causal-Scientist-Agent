# ECSA online JEPA mechanism router — experimental result

**Protocol:** `jepa_mechanism_router_v1` · **Source date:** 2026-10-09 · **Scope:** small synthetic six-epoch arena, frozen JEPA encoder.

## Research question

Can ECSA recognize saved predictive regimes without retraining, abstain when none explains new observations, acquire an informative experiment, and admit a new predictive checkpoint only after independent heldout verification?

The evaluator cycles through six 96-action epochs: Additive → Gated → Additive → Reverse-Gated → Additive → Reverse-Gated. The router receives only observation vectors, public one-hot actions and checkpoint IDs. It does not receive the source law, sensor latent state, or the change times. Initial checkpoints were prepared separately from additive data and gated adaptation. Reverse-Gated is absent from the initial bank. The third mode changes the sign of a gate-dependent pulse; it is not a new action vocabulary.

## Method

- A two-dimensional latent projection fitted only on the initial calibration cohort defines a shared grid of 36 observable next-state bins. For each known checkpoint, residual statistics on a separate mode-specific cohort parameterize a diagonal Gaussian predictive distribution over the **same bin outcome**.
- `ScienceKernel.update` updates the posterior using an explicit switching hazard. In ambiguous circumstances `ScienceKernel.score_experiment` chooses among public actions according to expected information gain.
- Repeated low-tail residuals under **every** known expert create a *pending proposal*, not an admitted mechanism. Each proposal is addressed at its own alarm epoch, including false alarms. Candidate training (64 transitions), calibration (512), and independently seeded heldout intervention confirmation (64) are separate. A candidate must outperform existing predictions on heldout data and not merely improve prediction on an already established regime.
- Each source-expert port copies its checkpoint and verifies the frozen encoder fingerprint. The new model is a predictor-only adaptation from the original anchor. No gradients reach the stored encoder.
- Optional existing `bocd` dependency provides a separate change-point diagnostic. The Bayesian hypothesis router does not require that optional package to run; BOCPD signals never admit hypotheses directly.

## Initial and independent evaluations

The first development group, full seeds 10–17, identified a sixth-phase recurrence blind spot and motivated adding immediate alarm-epoch verification. It is *not* used as a holdout confidence estimate. The intermediate six-seed group 18–23 (after the core rule changes but before a reporting fix) produced six admissions and five majority recognitions of the returning third mode; seed 21 produced an extra duplicate proposal for an already registered mode and spent twice the normal intervention budget.

The final frozen-source verification used **fresh seeds 24–29**, with implementation SHA-256 recorded in `results/routers/source_hashes_before_fresh_holdout.txt`. Files `results/routers/fresh_holdout_seed_24.json` through `fresh_holdout_seed_29.json` contain all per-phase decisions, proposals, and costs.

| Outcome, seeds 24–29 | Observation |
| --- | ---: |
| New predictive-mode proposal after third-law change | **6/6** |
| Independent validation admitted candidate into memory | **6/6** |
| False proposal attempts across all six epochs | **0/6** |
| Recurrent third-law majority selection in final 20 observations | **6/6** |
| Correct third-law selections in final 20 observations (mean) | **18.5/20** |
| Known Additive reuse after Gated, final 20 (mean) | 16.5/20 |
| Known Additive reuse after novel third mode, final 20 (mean) | 12.8/20 |
| First novel proposal delay after true change | 6, 7, 16, 18, 21, 24 actions |
| Active actions acquired for new candidate (per seed) | **640** |

Notably, the second return to Additive was incorrectly/ambiguously classified in several observations (one run had 0/20 correct selections). A return classification is therefore not universally reliable, even though the candidate model is retained and later reused. The short 90-step smoke-training mode often produces poor or spurious novelty decisions; it is a schema/runtime test, not a performance claim.

## Interpretation

The current prototype provides a reproducible *known → unknown → propose → acquire → independently validate → remember → reuse* loop without simulator labels entering the router. The six holdout seeds are a small synthetic test set, not proof of universal learning or calibrated uncertainty. Some old-mode predictions and the first-period identification are still imperfect. The false duplicate proposal in the earlier seed 21 is an important negative outcome, retained in the record.

**Do not claim:** identifiable causal graphs, semantics of latent variables, a fully autonomous research scientist, statistical validity of the repeated empirical tail-test, superiority to a strong non-neural baseline, or correctness on non-resettable real systems. In particular, the independent heldout gate verifies predictive novelty, *not* a unique causal mechanism. Future tests should include raw/ridge and symbolic baselines, nonstationary sensor geometry, multiple simultaneous unknown mechanisms, and a statistically sequential-calibrated open-set test.

## Reproduce

```bash
# Existing dependencies: numpy, torch, pytest. bocd is an optional existing extra.
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmarks/run_jepa_router.py \
  --mode smoke --device cpu --seed 14 \
  --output results/router-smoke-14.json

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmarks/run_jepa_router.py \
  --mode full --device cuda --seed 30 \
  --output results/router-full-30.json
```

The full-mode model is intentionally small (latent width 8, hidden width 64), so training does not need a 7B checkpoint. The GPU run is left for the user's CUDA environment; the attached research runs were performed on CPU.
