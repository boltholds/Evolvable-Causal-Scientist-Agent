# ECSA: Verbalized numbers and operators in frozen text embeddings + KAN feedback

2026-10-08. Research-only CPU experiment, not a universal text-state-encoder claim.

## Hypothesis

A strong pretrained language embedding might preserve useful numeric order,
precision and comparison relations when JSON values are additionally rendered as
English words. The KAN mechanism and X/Y adapter-feedback loop remain unchanged:

`observed X -> frozen Text Embedding -> E_X adapter -> KAN population <- E_Y adapter <- frozen Text Embedding <- observed Y`.

Three lexical transforms use **all** original JSON fields, no hand-picked physics
features. Technical IDs are never verbalized as measurements.

- `raw`: exactly original canonical JSON
- `semantic_words`: path-tagged English number words, e.g. `20.10 = twenty point one zero`; comparators such as `>= = greater than or equal to`
- `digit_words`: each digit represented separately, decimal/sign markers explicit
- `raw_plus_words`: original JSON followed by the semantic verbalization; this
  arm is still **entirely textual**, with no explicit numeric feature channel
- `hash_control`: original deterministic hash n-grams
- `numeric_control`: exact JSON numeric scanner (zero text embedding features)
- `raw_plus_numeric_control`: raw pretrained text plus JSON numeric features

**All arms have identical 128-dimensional downstream KAN input shape.** The
pretrained embedding size is 64 for every text arm; the 64 numeric feature
slots are zeroed except in the numeric and hybrid controls. Every arm uses the
same 3 spline-KAN structural hypotheses and the same adapter architecture,
initialization seeds and fitting budget. The pretrained backbone is frozen.

`raw_plus_words` has higher character/token count; a token-length audit is
recorded with every pretrained arm where an HF tokenizer is present. An arm
whose input overflows the configured token context must not be described as
preserving the whole input in *the model's* representation, even though the
preprocessing string contains the raw document.

## Ground truth and splitting

Synthetic opaque actions induce five classes of relationships. They are hidden
from encoders/KAN and only used in the benchmark fixture and **evaluation-only**
contrasts: continuously varying numeric outputs, tiny numeric differences,
operator-dependent comparisons (`< <= > >= == !=`), sign dependence and
out-of-range magnitude transfer.

The training portion contains bootstrap observations and adaptation
observations. A distinct calibration partition controls the acceptance or
rejection of adapter/KAN feedback checkpoints. All models are scored on unseen
entity IDs and separate numerical values; test cases have original,
paraphrased and independent counterfactual Y variants. The counterfactual
observation is generated only after the trained system has been frozen. The
main `counterfactual` metric holds X identical and asks whether a substituted Y
from the wrong mechanism is recognized as incompatible. Model evaluation
consumes heldout Y but no training method sees it.

Feedback arms are `frozen` (warmup only) and `alternating` (matched additional
KAN/adapter gradient steps with calibration-only checkpoint selection).

### Acceptance and caveats

- Primary: within-law paired differences of counterfactual AUROC and Brier;
  report results per mechanism, not just a pooled number.
- Secondary: paraphrase, unseen ID, sign and range shift, token truncation.
- A verbalizer alone cannot guarantee that a pretrained encoder is monotonic
  or numerically accurate. LoRA is **not** trained in this experiment.
- This is a discrimination metric on matched-vs-counterfactual pairs, NOT a
  calibrated conditional probability `p(Y | do(X))`, causal identification,
  symbolic law extraction or proof of universal representation.
- Numeric/hybrid controls include explicit parsed floats and therefore test a
  genuinely different inductive bias.
- Randomized symbolic comparisons often require exact, discrete evaluation;
  no win should be inferred from passed unit tests alone.

## Commands

```
pip install -e '.[test,pretrained-encoder]'
python -m pytest -q tests/test_verbalized_state.py tests/test_verbalization_study.py
python -m ecsa.experimental.verbalization_study \
  --model Qwen/Qwen3-Embedding-0.6B \
  --revision 3d106eabb5535a84de3ae88f45887a78259b52de \
  --laws operator precision --seeds 0 \
  --arms raw semantic_words digit_words raw_plus_words hash_control numeric_control raw_plus_numeric_control \
  --bootstrap 16 --adaptation 10 --calibration 8 --heldout 16 \
  --warmup-steps 45 --feedback-steps 16 --dimension 64 \
  --max-seq-length 512 --device cpu \
  --output results/qwen_verbalization.json
```

Record both CI artifact JSON and CI status before reporting a winner. No
pretrained inference has been assumed solely from successfully importing a
local adapter. Interpret changes after alternating feedback separately from
the fixed-encoder comparison.
