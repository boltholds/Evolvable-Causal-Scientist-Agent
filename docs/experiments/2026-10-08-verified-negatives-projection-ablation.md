# ECSA: replay-supported negatives and train-only latent projections

Date: 2026-10-08. Status: implemented experimental benchmark, no BehR-7B
performance claim until an actual checkpoint run produces scores.

## Motivation

In the previous BehR-WorldModel-Textworld-Qwen2.5-7B evaluation, precision
reached AUROC ~0.65 using the full sentence representation but operator
relations remained near random. The numeric control was stronger on precision.
Two protocol confounders were identified:

1. Every input representation passed through an untrained frozen
   width -> 24 -> 2 projection before the spline-KAN. This can discard
   relevant dimensions of a 3584-dimensional causal-LM state.
2. Random Y rotations were treated as negative labels, although valid
   transitions often have the same Y across multiple X. In a binary
   operator environment this can produce roughly 50% false negatives.

The new study tests both issues separately without returning KAN -> encoder
backpropagation.

## Architecture

    pre-action state + action -> frozen causal LM -> X context features
    post-action observation  -> frozen causal LM -> Y context features
    training-only unsupervised projection X/Y, separately
        random orthogonal     OR  PCA fit only on training X/Y
        width 2 / 8 / 16 / 32
    competing spline-KAN heads on (z_X, z_Y), input width = 2 * latent_width

Pooling can use final-token full hidden state or masked mean of contextual
tokens. Encoder weights, feature pooling and projection basis are fixed after
training-only preprocessing. Only KAN heads receive optimizer steps.
No separate adapter-gradient "feedback" or LoRA is used.

**Caveat:** different widths change KAN parameter count. For a given width,
PCA and random controls have exactly the same KAN architecture, number
of parameters, observed data, training/validation splits and optimizer steps.
PCA is an unsupervised variance-preserving baseline, not guaranteed to
capture low-variance causal features. Frozen random and PCA bases are fitted
separately for X and Y.

## Replay-supported counterexamples

We do not automatically assert every mismatched Y to be impossible.
The negative sampler only accepts a candidate Y when all of the following
hold:

- There is a **declared deterministic, fully-observed replay contract**
  from the experimental environment. This is not inferred from 2-3 samples.
- The same pre-action observation and exact same action was independently
  executed at least three times and produced an identical canonical Y.
- A separate witnessed input X' with the same action schema/world epoch
  also has at least three consistent replays and a distinct observed Y'.
- Thus (X,Y') can be treated as incompatible **conditional on that
  deterministic Markov/world-epoch contract**.

No hidden law, precomputed counterfactual Y, or alternate-world oracle is
used inside the miner or KAN optimizer. Training groups are drawn from
repeated *observed outcomes*. A synthetic world evaluator has a private
transition function that is called independently at each replay; the
learner only sees its observed outputs.

If observations are stochastic, the action/schema varies, replays disagree,
a deterministic contract is missing or no distinct outcome exists, the
miner **abstains**. It does not invent a negative. This is a deliberately
conservative experiment, NOT a universal solved method for stochastic worlds.
For stochastic environments, eventually learn a conditional outcome-support
model or valid likelihood instead of incompatible-pair classifiers.

The fraction of collisions from the old shifted-Y protocol is reported
as a *diagnostic* (not a training target). A balanced binary example yields
50% shifted-label collisions and 0% collisions among mined disjoint outcomes.

## Train/test protocol

Use separately seeded deterministic-world **train / calibration / heldout**
episodes. No technical identity is reused across partitions. Fit PCA,
random-projection centering, and standardized scales using *train only*.
Train KAN using mined training witnesses and select checkpoints on mined
calibration witnesses. Heldout witnesses are used only for AUROC, paired
accuracy, Brier and heldout contrastive NLL.

Latent dimensions are 2, 8, 16 and 32. The heldout sets, KAN-head population,
learning-rate schedule, seeds and budgets remain matched for every projection
kind at the same width. Report pairs mined and skipped, because restricting
to replays with definite outcomes can introduce selection bias.

**The relation classifier is not a calibrated p(Y | do(X)).** Even
repeat-supported negatives need explicit Markov and deterministic assumptions.
This experiment does not establish universal causal mechanisms, numerical
precision, action planning or physical interventions beyond the synthetic
world generator.

## Reproduce without pretrained downloads

    pip install -e '.[test,neural-relations]'
    python -m pytest -q tests/test_verified_relation_kan.py

The GitHub Actions workflow runs all contracts on an offline frozen-vector
stub and runs the same end-to-end pipeline with actual
\`Qwen/Qwen2.5-0.5B\` weights on CPU. This is an inference smoke and limited
comparison, **not a BehR result**.

## Actual BehR test on a GPU

    pip install -e '.[test,pretrained-encoder]'
    pip install bitsandbytes

    python -m ecsa.experimental.verified_relation_kan \
      --candidate behr_textworld --device cuda --4bit \
      --laws precision operator --seeds 0 1 2 \
      --pool sentence_full token_mean \
      --projection random train_only_pca \
      --dimensions 2 8 16 32 \
      --train 48 --calibration 16 --heldout 24 \
      --replays 3 --steps 100 --max-length 1024 \
      --output results/behr_verified.json

Record the actual Hugging Face revision with the JSON result, and repeat
with \`--candidate base_qwen25\` at the same size and settings. For
fully reproducible checkpoint pinning, set \`--revision\` explicitly.
The supported frozen-causal LM port refuses 7B loads on ordinary CPU
unless a special override is used; Qwen-AgentWorld is a different
simulator and is not silently cast to a Qwen2 encoder.

## No-go / interpretation

No claim that BehR has become useful for ECSA follows from a passing test.
Success requires a multi-seed **heldout advantage against base Qwen2.5-7B**
that survives new numerical values, operators and new object identities.
Failure of PCA is not failure of the original frozen LM; it may still lack
supervised features necessary to express the physical law.

The previous \`world_model_encoder_benchmark.py\` remains a **legacy
shuffled-Y comparator** for historical reproducibility, not the accepted
negative-sampling evaluation protocol.
