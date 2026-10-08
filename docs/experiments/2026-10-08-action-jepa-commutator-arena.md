# ECSA: action-conditioned JEPA and causal commutator (first empirical gate)

This is a **small, deliberately falsifiable experiment**, not a benchmark of
V-JEPA 2, LeWorldModel, physical causal discovery, or real robots. It tests a
specific claim:

> Can a representation learned only from `(observation_before, action,
> observation_after)` let an independent scientist infer and verify that two
> interventions have a noncommuting (order-dependent) effect, without reading
> privileged world-state labels or mechanism names?

It also tests whether that representation adds value over observations and an
untrained encoder. The pretrained-LLM hidden-state comparison is **not** part
of this experiment.

## Implementation and boundary

- `jepa_world.py` owns the **evaluator-only** simulator truth, an opaque
  nonlinear sensor renderer, known intervention tokens (`pulse`, `flip`,
  `wait`), and unlabeled transition collection.
- `action_jepa.py` owns an 8-dimensional online encoder, EMA target encoder,
  action-conditioned predictor, and variance/covariance anti-collapse terms.
  **Not** the exact V-JEPA 2 or LeWM/SIGReg loss. Encoder is learned from
  scratch, without LM states, coordinates or mechanism labels.
- `jepa_causal_scientist.py` owns public observation/action protocols,
  action-order intervention acquisition, latent comparison, evidence IDs,
  noise controls, abstention on collapse, and an optional model-based action
  pair proposal. It has **no dependency on `Mechanism` or simulator internals**.
- `jepa_commutator_arena.py` is the only component that joins those components
  and grades hypotheses against privileged ground truth. Its mechanism labels
  never enter learner or scientist.

The primary controlled experiment compares the final outcomes of
`pulse → flip` and `flip → pulse` from *exactly the same initial state*.
Under the additive mechanism, `pulse` moves the controlled level regardless of
flip status and the two orders commute. Under the gated mechanism the pulse
only acts while enabled, so the two orders do not commute. The independent
scientist does not receive either rule. An identical repeated sequence
estimates sensor noise. The score reports the mean distance between
intervention outcomes, the within-sequence noise distance, and the
representation spread. A collapsed encoder abstains instead of declaring
both actions commuting.

The primary decision rule is preregistered and deliberately crude:
`gap > 2.5 × null_gap` **and** `gap - null_gap > 0.05 × spread`.
This is **not** a calibrated significance test. Multiple seeds and real-world
noise models are required before stronger claims.

## Controls

1. JEPA with correct action conditioning: predicts heldout next-observation
   embeddings from the observed pre-state and intervention token.
2. JEPA with **train-only permuted action tokens**, holding observations,
   targets, update budget, architecture, and RNG seed fixed. Tests whether
   action conditioning matters for next-state prediction; its encoder is also
   scored for commutator discovery.
3. Frozen randomly initialized **same-width** encoder.
4. Raw observation vectors as the representation.
5. A repeated identical intervention sequence to calibrate nuisance distance.

All four representation choices score the **identical acquired intervention
observations**, not independently regenerated paths. The heldout state/action
split is separated from training by trial seeds, but sensor geometry is shared:
this is **not** an appearance domain-shift benchmark. No numeric target,
operator law, identity label, or mechanism ID is fed into training.

Primary grades are against the preregistered `pulse/flip` pair. Independently,
the JEPA predictor ranks three action pairs by predicted latent
noncommutativity, and the highest-ranked pair is executed and reported as
`active_scientist` evidence. Do not confuse a correct preregistered grade
with autonomously choosing the correct experiment.

## Local run (Python >= 3.12)

```bash
python -m pip install -e '.[test,neural-relations]'
python -m pytest -q tests/test_action_jepa.py

OMP_NUM_THREADS=1 python scripts/benchmarks/run_action_jepa.py \
  --mode smoke --device cpu --seed 0 \
  --output results/action_jepa/smoke.json

# All checkpoints are small and trained from scratch; use local CUDA if desired.
for seed in 0 1 2 3; do
  OMP_NUM_THREADS=1 python scripts/benchmarks/run_action_jepa.py \
    --mode full --device cuda --seed "$seed" --noise-std 0.16 \
    --output "results/action_jepa/full-seed-${seed}.json"
done
```

Smoke: 192 train / 72 heldout transitions **per mechanism**, 70 gradient
steps per arm, 16 matched commutator trials. Full: 2048 train / 256 heldout,
700 steps per arm, 64 trials. Each mechanism trains two models sequentially;
no simultaneous multi-billion-parameter model loads. No download required.
On CPU use `--device cpu`; `--steps N` is a documented speed override.

## Acceptance and interpretation

A useful exploratory signal is (a) noncollapsed embeddings; (b) action ranking
above the shuffled-action control on heldout transitions; (c) correct
commutator judgments for both mechanisms across multiple seeds; (d) evidence
that the learned representation enables a commutator decision under nuisance
where random/raw controls using the **same statistic** do not. Report
per-seed outcomes, not dozens of architecture cells as independent trials.

A discovered **noncommutative action pair** is one causal structural property.
It is not identification of a unique causal graph, internal state variables,
mechanistic equations, stochastic causal effect, or transfer between
environments. Since separate JEPA weights are trained per mechanism, this
experiment does not test online mechanism-shift adaptation. An encoder trained
with scrambled action inputs may still learn a usable denoised representation;
therefore it must be reported separately from action-conditioned predictive
accuracy. Other estimators operating on raw observations might detect the
same interaction if trained appropriately. The current scientist is an
intervention-order test with a model-based proposal, **not** full ECSA
hypothesis generation and repair.

Reference directions, not copied implementations: V-JEPA 2-AC
<https://arxiv.org/abs/2506.09985> and LeWorldModel
<https://arxiv.org/abs/2603.19312>. This experiment has an EMA teacher and
VICReg-like variance/covariance regularizers, unlike LeWM's exact SIGReg.

## Exploratory local CPU measurement, 2026-10-08

Run conditions: Python 3.13, CPU PyTorch, `--mode full --noise-std 0.16`,
700 updates, 2048 train and 256 heldout transitions **per mechanism**, 64
paired commutator trials. Each row below averages **four seeds**, not the
256 correlated heldout samples. Noise `0.16` was selected after a pilot sweep:
seeds 0–3 are **exploratory**, seeds 4–7 were evaluated afterward without
changing the implementation or hyperparameters. A separate JSON summary
records all eight seed-level cases and SHA256 hashes of individual reports.

| Cohort | Mechanism | JEPA heldout action ranking | Shuffled-action ranking | JEPA scientist correct | Raw / random scientist correct | Shuffled encoder scientist correct |
|---|---|---:|---:|---:|---:|---:|
| Exploratory 0–3 | additive | 98.29% | 50.88% | 4/4 | 4/4 each | 4/4 |
| Exploratory 0–3 | gated | 91.31% | 38.53% | 4/4 | 0/4 each | 4/4 |
| Confirmation 4–7 | additive | 97.66% | 59.38% | 4/4 | 4/4 each | 4/4 |
| Confirmation 4–7 | gated | 91.31% | 47.02% | 4/4 | 0/4 each | 4/4 |

The learned JEPA representation let this particular intervention-order
statistic recover the gated effect under nuisance while the raw/random
representation controls did not. **Crucially, the shuffled-action encoder
also succeeded at commutator detection**, even though action-conditioned
heldout prediction was substantially weaker. Consequently this experiment
does *not* establish that correctly action-conditioned representation learning
is necessary for identifying this causal property. Nor does failure of the
raw/random statistic rule out a more powerful raw-observation baseline.

In all eight gated cases the model's optional action-pair proposal chose
`pulse/flip` and its follow-up intervention test returned a positive
interaction. For additive worlds all tested proposals gave a negative
interaction; the model proposed the `pulse/flip` pair in four of eight seeds.
These are 2 action-composition laws in one synthetic environment, not
arbitrary scientific hypothesis discovery or a test on previously unseen
physical worlds. The four new confirmation seeds remain too few for a strong
population-level reliability claim.
