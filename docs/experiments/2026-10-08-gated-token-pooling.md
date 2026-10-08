# Mean-anchored gated BPE-token pooling for ECSA

Date: 2026-10-08. Follow-up to the Qwen full-context token study.

## Question

The original experiment found that learned attention pooling and KAN->attention
feedback underperformed a *fixed masked token mean* on its first real-Qwen
controlled tasks. Is this at least partly an optimization/initialization
failure, rather than evidence that attention pooling is inherently inferior?

This is an EXPLORATORY follow-up, not an independent preregistered test of a new
hypothesis. Qwen weights and tokenizer stay frozen.

## Intervention

Existing baselines are not modified. Introduce three opt-in arms:

- `gated_attention`: masked mean + a learnable residual, initialized near zero.
- `gated_kan_update`: further update only KAN hypotheses after shared warmup.
- `gated_feedback`: freeze KAN and adapt the two independent X/Y attention
  projectors through the relation loss, with calibration rollback.

Mathematically, for the sequence of contextual embeddings Z with padding mask M:

    m = masked_mean(Z, M)
    a = attention_pool(Z, M)
    alpha = 0.5 * sigmoid(g), initialized g=-5
    pooled = normalize(m + alpha * (a - m))

`alpha` starts at roughly 0.00335 and is always <0.5. This is an inductive
bias for preserving the original masked mean, **not a guarantee** of held-out
non-regression: nonlinear projectors and KAN can still overfit.

The gated variant uses the *exact same initialized projection and KAN weights*
as the token-mean control (attention's initialization does not advance PyTorch's
main random stream). Independent X and Y adapters are retained. Only training
observations contribute to normalization; the calibration partition is disjoint.
No test labels, oracle-generated negative outcomes or held-out pairs are used
for training or checkpoint acceptance. Probability weights are empirical
calibration scores, not a causal Bayesian posterior.

## Evaluation

Keep the previous published seven-way run intact by default. New arms are
selectable through `--arms`, and the optional CI workflow runs three new MiniLM
seeds and a *new* Qwen seed 1 on precision, comparison operators, range-shift
and sign mechanisms. For each law, keep the same X/Y train/heldout splits,
negative outcomes and KAN training budget across all arms. Compare mean-pool,
old attention, gated attention, gated-head-only, gated-KAN feedback, sentence
and numeric controls. The attention variants add parameters; record the true
parameter count and avoid claiming FLOP matching.

Primary metric: counterfactual heldout AUROC. Secondary: paraphrase AUROC,
small-decimal AUROC, Brier, accepted feedback updates and latent spread.
A smaller CPU offline deterministic stub is only for protocol smoke testing,
**not** evidence of Qwen's representation quality.

The original Qwen seed 0 is already observed. Data from that seed is not an
unbiased held-out test of the architectural design choice; seed 1 is more
informative but still part of the same synthetic law families.

## Reproduction

    pip install -e '.[test,pretrained-encoder]'
    python -m pytest -q tests/test_token_state_feedback.py
    python -m ecsa.experimental.token_state_feedback \
      --model Qwen/Qwen3-Embedding-0.6B \
      --revision 3d106eabb5535a84de3ae88f45887a78259b52de \
      --laws precision operator range_shift sign --seeds 1 \
      --arms sentence_64 sentence_full token_mean token_attention \
             gated_attention gated_kan_update gated_feedback numeric_control \
      --bootstrap 24 --adaptation 12 --calibration 8 --heldout 16 \
      --warmup-steps 45 --feedback-steps 24 --max-seq-length 256 \
      --output results/qwen_gated.json

Do not promote gated attention to the primary agent until it wins consistently
on unseen mechanisms and much larger independent samples; none of these arms
prove causal identification or successful autonomous planning.
