# Universal shared-outcome evaluation on public DiscoveryWorld trajectories

This report covers the **evaluation interface**, not a confirmed improvement of JEPA or completion of Reactor Lab.

The target is discovered from the `PerceptionFrontend` of **training seeds 0, 1, 2 only**. Each numeric coordinate is normalized by training variance (with safe fallback for constant features), while categorical, boolean and attempted-action success coordinates are explicitly typed. All predictors use exactly the same frozen target frame, mask, action list, and heldout transitions. A test-only feature increments `unseen_feature_count`; it cannot enter the target vocabulary.

The evaluation distinguishes:

- **Observed behavior:** before/action/public result/after.
- **Predictive progress:** lower error or better calibrated forecast on frozen heldout public targets.
- **Observational effect:** action-correlated change after context matching.
- **Interventional evidence:** a separate, externally attested prospective controlled contrast.

A low error does **not** indicate discovery of a causal law.

## Local base evaluation

Use the output directory with 1000-action `Reactor Lab / Normal` cold runs for seed 0, 1, 2, 3, 4, as previously collected:

```bash
python -m ecsa.benchmarks.discoveryworld.shared_probe \
  --runs results/discoveryworld_interface_source \
  --representation public_hash \
  --output results/universal_public_hash_common_target.json

python -m ecsa.benchmarks.discoveryworld.shared_probe \
  --runs results/discoveryworld_interface_source \
  --representation induced_interface \
  --output results/universal_induced_interface_common_target.json
```

Those two baseline-only reports should have the same `shared_frame_fingerprint` and score denominators. This does **not** train JEPA.

## JEPA comparison on the RTX 4070 Ti

From an environment with the optional neural dependencies installed:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
python -m ecsa.benchmarks.discoveryworld.shared_probe \
  --runs results/discoveryworld_interface_source \
  --representation public_hash \
  --with-jepa --device cuda --seed 0 --steps 700 --batch-size 128 \
  --output results/universal_public_hash_jepa.json

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
python -m ecsa.benchmarks.discoveryworld.shared_probe \
  --runs results/discoveryworld_interface_source \
  --representation induced_interface \
  --with-jepa --device cuda --seed 0 --steps 700 --batch-size 128 \
  --output results/universal_induced_interface_jepa.json
```

The model sees only public trajectories for training, with a supervised train-only linear projection from its predicted latent state into the **same** frozen public-outcome frame as RawRidge and persistence. It has no access to the heldout test seed when fitting the representation or projection.

**Caveats:** Public observations can be incomplete or nonstationary. Heldout action distribution may be highly skewed; report per-action and macro metrics. The source learner can remain trapped in repetitive exploration; running offline JEPA does not fix its policy or increase task success. In the current adapter, published latent probe and this common-frame result are distinct experiments; compare their definitions rather than concatenating their MSE values.

## Acceptance

- Equal target fingerprints when only the predictor input representation changes.
- Equal `row_count`, `coordinate_ids`, and target masks across all backends.
- No model fit or feature-vocabulary extension from validation/test.
- Explicit coverage/abstention and per-action scores.
- Regresion tests on a second environment with unrelated action symbols and observed feature names.
- An actual JEPA performance claim requires running both commands with `--with-jepa` on real logged trajectories, checking CUDA outcomes and at least multiple model initialization seeds.

These conditions are necessary but not sufficient for identification of a causal mechanism.
