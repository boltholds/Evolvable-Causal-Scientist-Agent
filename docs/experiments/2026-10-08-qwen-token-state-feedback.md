# Full contextual Qwen BPE-token states with KAN feedback

Date: 2026-10-08. Status: experimental hypothesis; do not claim results before CI.

## Claim under test

Earlier ECSA experiments truncated Qwen3-Embedding-0.6B's 1024-dimensional **sentence embedding** to 64 dimensions. The Qwen encoder's embedding already uses the model's native BPE tokenizer; replacing BPE with another segmentation is not this experiment.

We test whether the information needed to discover relations between unknown X/Y states survives in the **full sequence of final-layer contextual token embeddings**, and whether an attention aggregator learns to preserve it when trained with a negative-feedback signal from frozen KAN hypotheses.

Preserve the two-sided architecture:

    raw before-state + action -> Qwen native tokens -> Qwen frozen -> token states X
    raw after-state + outcome -> Qwen native tokens -> Qwen frozen -> token states Y
    independent X/Y pooling + projection -> independent latents -> spline-KAN ensemble

No direct `Y = f(X)` regression, causal posterior, autonomous action-selection claim, or oracle-provided training labels are introduced.

## Comparisons

At fixed data splits, Qwen checkpoint, unmodified model tokenizer, KAN-head architecture, optimizer budget for initial training and equal 2-dimensional KAN input per side:

1. **Sentence 64:** normalize and keep the first 64 dimensions of full Qwen sentence embedding, then project.
2. **Sentence 1024:** project all 1024 dimensions of full sentence embedding.
3. **Token mean:** mask-aware mean of contextual token embeddings, then project 1024 -> 2.
4. **Token attention:** learned masked attention pooling of contextual token states, then project 1024 -> 2.
5. **Attention + extra KAN steps:** same warmup weights as (4), then adapt the KAN hypotheses only for a matched number of optimizer steps.
6. **Attention + KAN-to-encoder feedback:** same warmup weights as (4), then freeze KAN hypotheses and adapt **both X/Y attention pooling and projection** for the same number of optimizer steps.
7. **Numeric-only control:** original generic JSON numeric-hash baseline, with 64-dimensional input and the same KAN-head and warmup budget. This is NOT an exact-number oracle: magnitude is `tanh`-compressed and feature slots can collide.

The feature input and projector parameter counts differ across 64 vs 1024 vs attention variants, **by design**. The number of trainable model parameters is saved for every arm. This is a representation-capacity ablation, not a global parameter/FLOP-matched comparison. Extra-update arms also use more optimizer steps than the frozen controls; comparisons (5) vs (6) have the same adaptation budget.

## Avoiding data leakage and incorrect gradients

- Samples are `TextRelationSample` created by the existing `FullTextTransitionProjector`. X contains only pre-action observations and action; Y contains only post-action state and the observed outcome.
- Evaluation identities are disjoint from training identities. Warmup, adaptation and checkpoint calibration are disjoint *contiguous subsets* of a synthetic dataset; held-out samples and counterfactual outcomes are created separately.
- No benchmark hidden law or counterfactual outcomes are used in training or calibration.
- The Qwen backbone is **frozen** throughout, and its token/sentence embeddings are cached by exact text. Model weights and tokenizer/revision are pinned for Qwen.
- Each X/Y state projection is additionally centered and standardized **per coordinate using only bootstrap/warmup observations**, with a variance floor of 0.001. This corrects the Qwen and MiniLM collapse observed in the first CI run, where all full/pooled representations mapped to constant 2D latents (spread=0, AUROC=0.5). Warmup moments are frozen during feedback and validation; no heldout statistics enter normalization.
- Padding is masked before computing any pool. The real-model tokenizer preflight checks full raw token lengths, raising on possible truncation. The maximum token length and number of verified texts are reported.
- When feedback runs, KAN heads have `requires_grad=False` and the gradient flows from their relation score to X/Y attention weights and projections. The KAN-only control does the opposite.
- Contrastive loss uses positive observed pairs and shifted observed Y as unlabeled negatives. This can introduce false negatives when several Y values are valid for the same X.
- Warmup only uses its data. Feedback optimizes only warmup+adaptation, with checkpoint acceptance based exclusively on the calibration partition, latent variance and a preservation penalty. After each accepted encoder change, hypothesis weights are refreshed using that calibration partition.
- Mixture weights are empirical calibration scores, **not Bayesian posteriors over physical laws**. None of the evaluation metrics establish causal discovery.
- For tests, use deterministic fake-token backbones and verify that last-64-only hides information in higher coordinates, full-sentence keeps it, and attention can distinguish token differences invisible to the pooled sentence. A fake backbone is not evidence of Qwen accuracy.

## Heldout tests

Synthetic fixture laws `precision`, `operator`, `range_shift`, `sign`, generated without exposing the mechanisms to the model. Evaluate on:

- `counterfactual`: same X, but benchmark-only counterfactual Y calculated from an altered control condition.
- `paraphrase`: alternate text phrasing and counterfactual Y at the same X.
- `small_numeric_delta`: same X with a +0.20/-0.20 change in Y's numeric value and unchanged ancillary text.

Metrics: pooled relation-recognition AUROC, within-pair accuracy, discriminative Brier; max untruncated token count, trainable/attention/KAN parameter counts, calibration loss, accepted/rejected checkpoints and encoder version. **AUROC and sigmoid Brier are not calibrated conditional Y likelihoods.** Report laws separately. Do not assert whether numeric-only is dispensable without sufficient independent seeds and numeric checks.

## Reproduction

```bash
pip install -e '.[test,pretrained-encoder]'
pytest -q tests/test_token_state_feedback.py
python -m ecsa.experimental.token_state_feedback \
  --model Qwen/Qwen3-Embedding-0.6B \
  --revision 3d106eabb5535a84de3ae88f45887a78259b52de \
  --laws precision operator --seeds 0 \
  --bootstrap 24 --adaptation 12 --calibration 8 --heldout 16 \
  --warmup-steps 45 --feedback-steps 24 --max-seq-length 256 \
  --output results/qwen_full_token_states.json
```

The GitHub workflow `neural-token-state.yml` runs offline contract tests, multi-seed MiniLM as a faster real transformer control, and an **actual frozen Qwen3-Embedding-0.6B** CPU experiment; results are available as immutable CI artifacts.

If the attention-feedback arm underperforms the Qwen full-sentence / numeric baseline, the correct conclusion is that **this pooling/feedback training was insufficient**, not that contextual BPE token states lack the information in principle. If it improves, verify with additional seeds, shifts, failure modes and compute-matched baselines before any universality claim.

## First CI diagnostic and correction

The original commit `3a4980e` completed all jobs but exposed a projection failure: pretrained Qwen3/MiniLM `sentence_full`, `token_mean`, and `token_attention` produced zero latent spread, with AUROC exactly 0.5. This was a **training collapse** in the small high-dimensional projector, not a scientific negative result about retained BPE tokens. A subsequent version uses warmup-only per-coordinate centering and scale, increases the projection hidden width from 12 to 24 for all arms, and adds an explicit collapsed-feature regression test. Results from the two versions must not be pooled or compared as equivalent models.
