# Pretrained text-state encoder vs lexical hash vs numeric hybrid

Date: 2026-10-08. An independent experiment, not a claim of universality.

## Question

Does a capable pretrained whole-state text embedding eliminate the apparent advantage of a separate numeric channel in ECSA's implicit X/Y KAN relation recognition? Are observed differences attributable to the *encoder* rather than to representational necessity?

Preserve the architecture: `before observation + action -> E_X` and `after observation + outcome -> E_Y`, then `(z_X,z_Y) -> spline-KAN relation compatibility`. **No `Y=f(X)` regression or hidden-law access** is introduced.

## Arms and fairness

1. Original 64-dimensional hashed text front-end; numeric channel zero-filled.
2. Frozen pretrained `Qwen/Qwen3-Embedding-0.6B` (or explicitly selected pretrained ST model); same dimension 64, numeric channel zero-filled.
3. Frozen same pretrained model **plus** generic numeric JSON scanner; 64+64 dimensions.
4. Numeric-only JSON scanner; text channel zero-filled.
5. Optional Qwen LoRA adapter trained **only on training** X/Y positive pairs, then frozen and benchmarked with an identical KAN head. This arm uses *additional backbone optimizer steps*, so it is not a compute-matched comparison with frozen encoders.

All arms use exactly the same training/heldout transitions, X/Y strings, random seed for the downstream KAN, fixed `64+64` inputs, two learned encoder towers, KAN parameters, and downstream optimizer steps. The pretrained models have more frozen parameters than the hash layer by design; they are representation-capacity comparisons, not end-to-end parameter-count matches.

### Evaluation

- Three unknown environment fixtures: categorical, numeric and category×numeric interaction.
- Disjoint train vs heldout object identities and mild numeric support shift.
- Heldout natural-language **paraphrase** variant; numeric values and factual observations unchanged.
- AUROC, pair-ranking accuracy and discriminative Brier on heldout positive vs hard mismatched observed Y; hard mismatches selected *after* train/test split and never used for training.
- We do not claim calibrated `p(Y|do(X))`, causal discovery, symbolic formulas, or universality from these recognition metrics.
- Input retains entire `RawObservation` payload and action, including nuisance technical IDs. No hand-selected hidden-law fields are extracted for text input; for the separate numeric baseline, a generic JSON numeric scanner is used.
- Frozen encoders are always loaded through a strict, optional `TextEmbeddingPort`; no dependency on sentence-transformers in the core world model. Model/adapter inference caches are invalidated when LoRA updates weights.

## Code and reproducibility

Install CPU or GPU PyTorch and the opt-in extra:

```bash
pip install -e '.[test,pretrained-encoder]'
pytest -q tests/test_pretrained_text_study.py
```

Run frozen 0.6B Qwen on GPU (faster; CPU also works with short sequences):

```bash
python -m ecsa.experimental.pretrained_text_study \
  --model Qwen/Qwen3-Embedding-0.6B \
  --revision 3d106eabb5535a84de3ae88f45887a78259b52de \
  --arms hash_text frozen_pretrained_text frozen_pretrained_hybrid numeric_only \
  --laws categorical numeric interaction --seeds 0 1 2 \
  --train-samples 64 --heldout-samples 32 --feature-size 64 \
  --steps 80 --max-seq-length 256 --device cuda \
  --output results/qwen_frozen.json
```

**LoRA is a separate arm and needs GPU for the full model:**

```bash
python -m ecsa.experimental.pretrained_text_study \
  --model Qwen/Qwen3-Embedding-0.6B \
  --revision 3d106eabb5535a84de3ae88f45887a78259b52de \
  --arms hash_text frozen_pretrained_text frozen_pretrained_hybrid numeric_only lora_pretrained_text \
  --laws interaction --seeds 0 \
  --train-samples 64 --heldout-samples 32 --steps 80 --feature-size 64 \
  --max-seq-length 192 --device cuda --lora-steps 12 \
  --output results/qwen_lora.json
```

The LoRA objective is symmetric in-batch X/Y contrastive matching (not joint KAN training); other Y values in the batch can sometimes be true alternatives, so false negatives are a limitation. In the frozen runs the Qwen backbone is NOT adapted. Never report that LoRA was evaluated unless the LoRA arm executed successfully.

GitHub CI runs offline contract checks, CPU MiniLM pretrained baseline, and a small real frozen-Qwen interaction smoke/ablation. The full multi-seed Qwen + LoRA experiment needs a GPU. Compare confidence intervals over separate seed groups only after completion; do not infer universal numerical state encoding from a small experiment.

## Counterfactual mode-control gate (added after preliminary frozen runs)

The initial interaction negative-sampling protocol could be solved partly by
numeric *magnitude* alone, without reading the categorical state. It was not
an adequate isolation of text-encoder value. A new evaluation domain called
`contrasted_mode` therefore keeps X **exactly identical** and replaces only Y
with a benchmark-only outcome from the *opposite categorical mode*, holding
all numeric controls fixed. That test is only defined for `categorical` and
`interaction` fixture laws; it is never shown during fitting. It asks whether
the relation model detects that a different categorical state leads to a
different valid Y for identical numbers. It is an evaluation-only
counterfactual generated by a known synthetic fixture, NOT evidence that the
agent independently discovered a causal mechanism.

Always report this gate separately from the previous near-mismatch AUROC;
never combine the two evaluation populations into a single headline metric.
