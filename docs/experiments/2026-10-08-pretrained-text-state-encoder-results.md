# ECSA: Test-first pretrained encoder ablation

Commit: `e7dc84b0dd7a6720bd6ceee5c8a93cf93a427eef`. [CI run](https://github.com/boltholds/Evolvable-Causal-Scientist-Agent/actions/runs/37775205819).

Evaluation: X=entire before-state plus action; Y=entire after-state plus outcome; independent representation heads feed an identical spline-KAN implicit compatibility head.

## Frozen MiniLM, heldout AUROC by law (two seeds)

| Hidden relation | Hash | MiniLM text | MiniLM + numeric | Numeric only |
|---|---:|---:|---:|---:|
| categorical | 1.000 | 1.000 | 0.938 | 0.519 |
| numeric | 0.468 | 0.499 | 0.846 | 0.931 |
| interaction | 0.481 | 0.490 | 0.535 | 0.924 |

## Opposite-mode counterfactual (numeric X unchanged)

| Law / model | Hash | Pretrained text | Pretrained + numeric | Numeric-only |
|---|---:|---:|---:|---:|
| minilm / categorical | 1.000 | 0.999 | 0.947 | 0.514 |
| minilm / interaction | 0.766 | 0.580 | 0.703 | 0.478 |
| qwen / interaction | 0.773 | 1.000 | 0.990 | 0.439 |

## Ordinary near-mismatch Qwen interaction (one seed)

| Hash | Frozen Qwen3-Embedding-0.6B | Qwen + numeric | Numeric-only |
|---:|---:|---:|---:|
| 0.481 | 0.477 | 0.484 | 0.780 |

## Interpretation

- Semantic frozen embeddings are strong on categorical counterfactual compatibility, but are not uniformly sufficient for detailed numerical relations.
- Numeric-only performs strongly on near-mismatch Y comparisons even in the interaction fixture, showing a numerical-magnitude shortcut. Contrasted-mode testing deliberately removes that shortcut.
- Small train sizes and 64-dimensional Qwen truncation limit any claim about model universality or relative capacity.
- This study does not evaluate calibrated predictions of Y under do(X), symbolic formula discovery, or downstream action selection.
- Qwen LoRA was evaluated separately on one law/seed for six CPU optimizer steps. Its original near-mismatch AUROC improved slightly, but its categorical counterfactual discrimination collapsed; this is an objective-specific result, not a general proof of catastrophic forgetting.

Full per-run results are in the accompanying JSON and the GitHub Actions artifacts.

## Short CPU LoRA adaptation of Qwen3-Embedding-0.6B

[Successful LoRA CI run](https://github.com/boltholds/Evolvable-Causal-Scientist-Agent/actions/runs/37775205555). Same 48 train and 24 heldout transitions in one interaction law, seed 0. Six LoRA optimizer steps with batch size 2 and sequence limit 96. Both variants train a fresh identical KAN-head with 75 downstream steps. The adapted backbone spends extra compute.

| Evaluation | Frozen Qwen | Qwen with six LoRA steps |
|---|---:|---:|
| Heldout near-mismatch AUROC | 0.477 | 0.528 |
| Paraphrase near-mismatch AUROC | 0.462 | 0.519 |
| Opposite-mode counterfactual AUROC | **1.000** | **0.512** |

The chosen symmetric in-batch pair-contrastive adaptation objective improved near-mismatch rank discrimination but destroyed the initial counterfactual categoric ranking in this single small run. It is **not** evidence that LoRA inherently degrades embeddings; loss/objective, limited training data, update count, and target dimension can all matter.

## Verification

All workflows on commit `e7dc84b` succeeded: 228 full-suite tests passed (9 skipped), 113 world-model tests passed, 31 autonomous-discovery tests passed, 19 offline text encoder tests passed; frozen Qwen, MiniLM and actual CPU LoRA experiments completed. Numeric and text representations used matched KAN parameter counts, with the extra pretrained backbone capacity and LoRA compute explicitly unbalanced.