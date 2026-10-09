# ECSA / DiscoveryWorld: Public Action-JEPA Replay v1

This experiment trains the existing ECSA JEPA on **genuine observed
DiscoveryWorld `Reactor Lab / Normal` transitions** previously saved by
`ecsa.benchmarks.discoveryworld.arena`. It does **not** alter the policy,
actions, official task score or scientific hypotheses. The seed-0 smoke
summary with a 0.16 official score and zero admitted mechanisms remains the
baseline until an interactive paired JEPA evaluation is built.

## 1. Use the already collected 100-step seed-0 run (smoke only)

From the ECSA repo and activated `.venv-wsl` (do not use `sudo`):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
python -m ecsa.benchmarks.discoveryworld.jepa_probe \
  --runs results/discoveryworld_smoke \
  --arm cold --mode smoke --device cuda \
  --seed 0 --steps 400 --batch-size 64 \
  --output results/discoveryworld_jepa_smoke.json
```

The loader requires the actual `seed-0/cold/observations.jsonl`,
`actions.jsonl` and `run.json`, **not** only `summary.json`. If those files
are missing, rerun Arena A. The smoke result divides one episode
chronologically and provides no evidence of cross-seed generalization.

## 2. Collect heldout benchmark trajectories

With the same baseline policy (and a fresh directory):

```bash
printf '{}\n' > /tmp/dw-policy.json
python -m ecsa.benchmarks.discoveryworld.arena \
  --scenario 'Reactor Lab' --difficulty Normal \
  --seeds 0,1,2,3,4 --arms cold,reuse --max-steps 1000 \
  --policy-factory ecsa.benchmarks.discoveryworld.policies.autonomous_scientist:create_policy \
  --policy-config /tmp/dw-policy.json \
  --output results/discoveryworld_jepa_source
```

This collects real observations. The JEPA probe only selects one arm (`cold`
by default), so matched cold/reuse episodes of the same seed cannot leak
between training and heldout sets. Beware: the full paired command has a
large 10-episode action budget; optional smaller caps are fine for pipeline
smoke tests, but are **not** official full-budget performance measurements.

## 3. Independent heldout seed: JEPA and matched controls

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
python -m ecsa.benchmarks.discoveryworld.jepa_probe \
  --runs results/discoveryworld_jepa_source \
  --arm cold --mode study --device cuda \
  --seed 0 --steps 700 --batch-size 128 \
  --output results/discoveryworld_jepa_study.json
```

For five available seeds: train `0,1,2`, validation `3`, heldout `4`.
The same public transitions are provided to three neural arms and raw Ridge:
JEPA; shuffled-action JEPA; no-action JEPA; and a raw-feature Ridge delta
regressor. No official reward/scorecard is an input to the representation.

## 4. Report interpretation

- `heldout_mse` is only comparable to `persistence_mse` **within each arm**.
  JEPA ablations use distinct learned latent scales; Raw+Ridge uses public
  hash-feature coordinates. Compare `improvement_over_persistence`, plus
  `action_choice_accuracy` on true versus *different logged action types*.
- `action_choice_accuracy` is an *observational scoring diagnostic*, **not**
  a counterfactual intervention guarantee; alternative actions were not
  actually executed from the same world state.
- Low action diversity or near-static screenshots may make all methods
  perform weakly. This is an actual negative result, not proof JEPA is broken.
- `mean_latent_std` and `effective_rank` help identify trivial collapsed
  representations; noncollapse alone does not imply learned world dynamics.
- The sensor adapter uses deterministic hashed public JSON, not image/text
  JEPA or learned identity/grounding. Cross-seed generalization is limited by
  action/entity distribution and public observations.

**Next PR:** pair the existing Arena B scientist with and without a JEPA
observation/prediction sidecar using identical environment budgets and no
oracle state. Measure official score, intervention count, false discoveries
and scientific-memory transfer — none are measured by this offline probe.
