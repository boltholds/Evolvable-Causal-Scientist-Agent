# ECSA: frozen observation encoders, KAN-only hypothesis updates

Status: current architecture decision, 2026-10-08.

## Decision

**Retire negative feedback (ООС) from KAN hypotheses into the X/Y encoders.**
The prior encoder-feedback investigations are preserved in Git history and
dated experimental reports, but are no longer executable training modes in
the current main-branch experiment pipeline.

The active data flow is:

    RawObservation(before) + chosen GroundAction -> frozen Qwen BPE token states X
    RawObservation(after) + observed outcome -> frozen Qwen BPE token states Y
    frozen state pooling and fixed X/Y projections -> latent z_X, z_Y
    trainable spline-KAN hypotheses -> pair compatibility, mechanism comparisons

No optimizer is allowed to modify token-attention weights, gated-attention
coefficients, projection weights, warmup-only normalizer buffers or frozen
Qwen parameters on the basis of KAN prediction error. The only learned
parameters in the token-state relation experiment are the KAN heads.
Warmup-only feature normalization is deterministic estimation from the
training partition; it is not a gradient update.

## Retained

- Qwen3-Embedding-0.6B native BPE and full contextual token embeddings
- 64D vs full sentence embedding, masked mean, token attention, gated attention
- numeric-only baseline and the historical verbalization fixture generators
- three competing spline KAN heads, KAN-only warmup and optional extra head steps
- split integrity checks, hidden-law evaluation, masking/truncation checks,
  calibration acceptance/rejection and heldout measurement
- historical analyses, negative and positive results, and old source in git

## Retired from main

- KAN -> token attention / adapter gradients (encoder OOC)
- `attention_feedback` and `gated_feedback` modes
- `feedback_encoder.py` plus its KAN-to-X/Y-adapter experiment and CI
- the runnable feedback version of the number-verbalization benchmark
  (its raw synthetic transition generators remain in
  `src/ecsa/experimental/verbalization_study.py`).

Source of record for active experiments:
`src/ecsa/experimental/token_state_relations.py`
and `tests/test_token_state_relations.py`.

## Scope

This change intentionally does not claim better predictive accuracy.
It is an architecture simplification requested after several failed
KAN-to-encoder adaptation ablations. Earlier metrics are not directly
comparable once encoder optimization is removed. The opt-in Qwen-LoRA
pretraining experiment is a separate historical representation experiment
and is **not** part of the KAN hypothesis-update loop.

The ScienceKernel, autonomous action-selection and world-model contracts
remain unchanged.
