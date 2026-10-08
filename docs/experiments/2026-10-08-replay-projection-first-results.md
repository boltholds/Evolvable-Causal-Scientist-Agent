# Replay-supported KAN pairs and projection-width ablation: first real-weights result

Date: 2026-10-08. Experimental result, not a general capability claim.

## What was actually executed

Successful [neural-verified-relation-kan CI run 37822859838](https://github.com/boltholds/Evolvable-Causal-Scientist-Agent/actions/runs/37822859838)
on the real frozen **Qwen/Qwen2.5-0.5B** checkpoint,
revision **060db6499f32faf8b98477b0a26969ef7d8b9987**.

The pretrained **BehR 7B checkpoint was NOT executed** in this study.
Do not compare the present numerical values directly to the earlier BehR 7B
result: both the backbone and negative-pair protocol have changed.

The study used two synthetic deterministic fixture laws (precision, operator),
**one seed**, 48 train X/action groups, 12 calibration groups and 12 heldout
groups per law; each X was replayed three times in the simulator.
True and alternative Y pairs for training/calibration/heldout came only from
observed replay outcomes with disjoint deterministic outcomes and matching
world epoch/action schema. The hidden fixture law was not used by the miner.

Four latent widths (2,8,16,32), two frozen feature pools (full last-token
state, masked token mean), two training-only projections (random orthogonal,
PCA), and **two negative-label schemes** (replay-supported and historically
unsafe rolled Y) yielded **64 individual score rows**.

The two label schemes used the same X, positive Y, projection, initial
KAN parameters, calibration partition, update budget, and replay-supported
heldout negative outcome. The unsafe scheme is solely an explicit diagnostic
of label contamination, not a recommended training procedure.

## Numeric summary

Mean heldout pair AUROC over eight projector/width combinations per pool.
These eight cells are **not eight independent random seeds**.

| Law | Frozen pool | Witnessed negative training | Shuffled-Y unsafe control |
| --- | --- | ---: | ---: |
| precision | full hidden state | 0.5990 | 0.5946 |
| precision | masked token mean | 0.6146 | 0.5616 |
| operator | full hidden state | 0.3029 | 0.3221 |
| operator | masked token mean | 0.4002 | 0.3481 |

Best observed witness-trained *precision* cell was token mean + random
projection, width 16: AUROC **0.6944**. Best observed witness-trained
*operator* cell was full final-token state + PCA, width 32: AUROC **0.6458**.
The matched shuffled-Y operator control in **that single cell** had AUROC
**0.5000**. These are exploratory maxima across many tried settings, so they
are **selection-biased**, not heldout validation of an architecture winner.

Diagnostics of the old circular shift among the original train witness groups:

- precision: 0.2083 (10/48) of the shifted alternatives had exactly the
  same observed Y as the real transition;
- operator: 0.5417 (26/48) had the same Y as the real transition.

The successful repeat-supported sampler filtered all identical Y outcomes
for the mined pairs **under the explicit deterministic full-state fixture
contract**. This does not prove incompatibility in general stochastic worlds.

## What the results imply

1. Label contamination from shuffled outcomes is real and observable.
2. Corrected negatives **do not automatically improve every architecture**;
   the operator law is still poorly recognized on average using 0.5B weights.
3. The projected input width matters and a 2-dimensional bottleneck is
   scientifically confounding, but PCA is not a universal winner. Some
   random projections outperform PCA; in a particular operator run PCA32
   recovered more useful information.
4. Even after removing these confounders, current results DO NOT establish
   that a pretrained LM has acquired transferable quantitative or causal
   state representations for ECSA.
5. Actual BehR 7B versus base Qwen2.5-7B comparison at matched width,
   samples and repeat-supported negatives remains OPEN.

## Reproduction

The full score JSON is uploaded by [CI run 37822859838](https://github.com/boltholds/Evolvable-Causal-Scientist-Agent/actions/runs/37822859838).
Run this on a CUDA workstation to produce the next actual BehR result:

    pip install -e '.[test,pretrained-encoder]'
    pip install bitsandbytes
    python -m ecsa.experimental.verified_relation_kan \
      --candidate behr_textworld \
      --revision 6a4326a60540cc33ffa42ee7a16fc2bad8f6f613 \
      --device cuda --4bit \
      --laws precision operator --seeds 0 1 2 \
      --pool sentence_full token_mean \
      --projection random train_only_pca \
      --dimensions 2 8 16 32 \
      --pairing witnessed shuffled_y_unsafe_control \
      --train 48 --calibration 16 --heldout 24 \
      --replays 3 --steps 100 --max-length 1024 \
      --output results/behr_verified.json

A companion run using the same-family base Qwen2.5-7B and the same splits
is the proper pretrained world-model comparison. Do not conflate the
different backbone or generation prompt formats.

This benchmark still assumes deterministic, repeatable, fully observed
interventions and tests mainly **state-conditioned outcomes under one
action schema**, not varying action schemas, hidden state, stochastic
transition support or model-based planning.

The KAN-to-encoder negative feedback remains retired: all encoder
weights and PCA/random projection bases are frozen while only the KAN
hypotheses receive gradient-based updates.
