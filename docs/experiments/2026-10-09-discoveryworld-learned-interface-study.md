# ECSA / DiscoveryWorld: learned observational interfaces + Action-JEPA

This is the FIRST observational-contract experiment, not a DiscoveryWorld
success result and not a demonstration of autonomous causal induction.

The baseline one-seed/100-step smoke showed task score 0.16 in cold and reuse,
with no admitted mechanisms. In the initial fixed-hash JEPA smoke, the main
arm had −35.4% gain against latent persistence, with effective rank ~1.
The shuffled-action arm scored 95% against one alphabetically chosen foil,
which motivated the redesigned observational contrast diagnostics.

## Run on official public logs

Collect official Reactor Lab Normal seed 0–4 cold trajectories using the
already existing Arena A in a **new** output directory. Keep the same policy
and budget in all arms. The 100-step smoke is not an official final score.

```bash
python -m ecsa.benchmarks.discoveryworld.arena \
  --scenario 'Reactor Lab' --difficulty Normal \
  --seeds 0,1,2,3,4 --arms cold,reuse --max-steps 1000 \
  --policy-factory ecsa.benchmarks.discoveryworld.policies.autonomous_scientist:create_policy \
  --policy-config /tmp/dw-policy.json \
  --output results/discoveryworld_interface_source
```

Now, with activated WSL venv and without sudo:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
python -m ecsa.benchmarks.discoveryworld.jepa_probe \
  --runs results/discoveryworld_interface_source \
  --arm cold --mode study --representation induced_interface \
  --device cuda --seed 0 --steps 700 --batch-size 128 \
  --output results/discoveryworld_induced_interface_study.json
```

Control with unchanged feature hashing and same data:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
python -m ecsa.benchmarks.discoveryworld.jepa_probe \
  --runs results/discoveryworld_interface_source \
  --arm cold --mode study --representation public_hash \
  --device cuda --seed 0 --steps 700 --batch-size 128 \
  --output results/discoveryworld_public_hash_study.json
```

Training seeds 0–2; validation 3; test 4. Do not tune on seed 4.
For exploratory smoke use one existing seed, `--mode smoke`, with the clear
warning that no cross-seed inference is possible.

## What the report means

- `contract_induction.proposals`: discovered source-blind action symbols,
  parameter names, object-binding frequencies, effect/precondition hypotheses.
- `contract_induction.validation` / `heldout`: matching public effects and
  applicability in truly disjoint seeds (not causal confirmation).
- `contract_induction.unseen_test_properties`: representation shift absent
  from the train-only interface vocabulary; not backfilled.
- `action_choice_by_type` and `action_choice_macro_accuracy`: all distinct
  *logged action types* versus their recorded outcomes. Candidate arguments
  may be inapplicable; this diagnostic is **NOT** a counterfactual test.
- `public_delta_mse` and `public_delta_improvement`: train-only linear readout
  from JEPA predicted latents into the SAME public feature-delta coordinates.
  Compare to public persistence / Raw+Ridge, unlike arm-local latent MSE.
- `collapse_warning`, `effective_rank`, and `mean_latent_std`: a warning, not
  a proof of collapse. Values from tiny heldout cohorts have high variance.

A causal hypothesis should be admitted only by a later interventional
contract test in the live environment using independent prospective evidence;
this v1 tool **does not** make that claim. The current source-blind inducer is
experimental and does not replace `WorldModelAcquisitionKernel` or
`ContractExperimentCoordinator`.
