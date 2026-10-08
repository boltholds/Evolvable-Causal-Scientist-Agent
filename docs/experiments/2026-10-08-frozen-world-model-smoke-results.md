# Frozen pretrained world-model encoder: first verification

Date: 2026-10-08
Implementation: [2c08c306](https://github.com/boltholds/Evolvable-Causal-Scientist-Agent/commit/2c08c306298e4525c27e0781a68c3c2af5830976)
CI: [neural-world-model-encoder run 37810821112](https://github.com/boltholds/Evolvable-Causal-Scientist-Agent/actions/runs/37810821112)

## Verification

- Offline contracts: **34 passed**. These tests use a tiny injected synthetic
  model and are *not* evidence of pretrained capability.
- Real pretrained inference smoke: **success**, loaded
  `Qwen/Qwen2.5-0.5B` revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987` on CPU.
- Frozen contextual hidden states: verified; 55 model forward passes.
- Correct teacher-forced target-shift NLL: verified; zero gradient to backbone.
- Bootstrap/calibration/test split: verification checks passed.
- Core world-model and autonomous-discovery CI: successful.

The smoke model is not BehR. Do not label any score here "BehR".

## Actual small Qwen smoke measurements

One synthetic **precision** law, seed 0, 8 heldout transitions.
Counterfactual pair compatibility AUROC through separate frozen projections
and trained spline-KAN hypotheses:

| Representation | AUROC |
| --- | ---: |
| Frozen small-Qwen final-token hidden state (full width) | 0.8750 |
| Frozen small-Qwen masked token mean | 0.1406 |
| Existing tanh-compressed numeric-hash control | 0.8750 |

For teacher-forced next-token likelihood, **only two** heldout
positive/alternative Y comparisons were scored:
- paired ranking accuracy: 1.0 (2/2)
- true Y mean token NLL: 2.9442
- alternative Y mean token NLL: 3.0935
- mean negative-positive NLL margin: +0.1493

These numbers are a **pipeline smoke test**, not a statistically meaningful
world-model benchmark. The model was not trained as a world model and was not
tested on other environments, laws or seeds. A high AUROC here is not
causal-identification evidence. The original numeric baseline is a lossy hash,
not a perfect numeric oracle.

## Remaining scientific question

The **7B BehR text-world-model checkpoint was not run** in this CI because it
requires substantially more CPU memory / GPU VRAM than the available CPU
runner. No trained BehR embeddings, accuracy or ability to transfer to ECSA
have been observed in this experiment. A local GPU/remote inference environment
is required for this next comparison.

The experiment should next compare actual BehR TextWorld 7B **against its
same-family Qwen2.5-7B base**, with identical samples and prompts. Also
separately evaluate native TextWorld-style prompts, since canonical JSON is
out-of-distribution relative to BehR's original training.

The 35B Qwen-AgentWorld is a separate **simulator** control; it has not been
executed and is not handled as a Qwen2 hidden-state embedding checkpoint.

All earlier KAN-to-encoder negative feedback remains **retired**. The
pretrained LM, input token representations, attention-pooling and projection
weights remain frozen, with only spline KAN hypotheses being updated.
