# ECSA — Verbalizing Numbers and Operators, 2026-10-08

## Result

Pretrained frozen embedding + separate X/Y adapters + unchanged spline-KAN hypothesis bank. The verbalization formats are compared on identical observed transitions and KAN optimization budgets. AUROC below measures matched X/Y compatibility versus evaluation-only counterfactual Y, **not** causal identification or exact regression.

| Representation | MiniLM 5 laws / 2 seeds | Qwen operator / 1 seed | Qwen precision / 1 seed |
|---|---:|---:|---:|
| Raw JSON | 0.494 | 0.492 | 0.660 |
| Semantic number/operator words | 0.566 | 0.523 | 0.586 |
| Digit words | 0.553 | 0.449 | 0.535 |
| Raw JSON + semantic words | 0.506 | 0.344 | 0.566 |
| Hash baseline | 0.552 | 0.234 | 0.574 |
| Hashed numeric baseline | 0.770 | 0.199 | 0.756 |
| Qwen/MiniLM + numeric | 0.756 | 0.320 | 0.797 |

The four text formats are evaluated without an explicit numeric channel. Qwen text input truncation rate was zero for both tested laws; Raw+words used at most 223 input tokens out of 512. On Qwen precision, Raw won among text formats, while the explicit numeric/hybrid control scored higher. On operators, semantic words were slightly higher than Raw, but neither performed far above chance. MiniLM showed heterogeneous effects across individual laws; the pooled improvement does not establish generalization.

## Method and limitations

- Heldout object IDs and numeric values are disjoint from training; outputs are never provided as prospective X input.
- Validation/calibration and train partitions are separate; `FeedbackMode.ALTERNATING` updates only adapters and KAN, not the frozen pretrained backbone.
- All numbers are rendered as English literal words with trailing decimal digits preserved. Raw+words contains byte-for-byte full original JSON.
- Technical ID fields remain unchanged; operators were transformed to unambiguous English phrases.
- MiniLM: 5 synthetic laws × 2 seeds × 7 representations × 2 training modes × 3 evaluation domains = 420 metric rows.
- Qwen: single seed and two separate laws, 7 representations × 2 modes × 3 evaluation domains = 42 metric rows per law.
- The numeric control uses the existing **hashed tanh** numeric scanner; it should not be treated as a perfect exact-value oracle.
- AUROC is a pair-ranking metric, not a calibrated conditional p(Y|do(X)) and not evidence that KAN inferred physical laws.
- No positive demonstration that word conversion alone solves numeric-state representation or that OOS improves it reliably.

## Reproducibility

GitHub CI: https://github.com/boltholds/Evolvable-Causal-Scientist-Agent/actions/runs/37781203839

Repository commit: https://github.com/boltholds/Evolvable-Causal-Scientist-Agent/commit/15fa63a4bdf9193a6555044ca8479231c3b5d1e3

All raw per-run metrics are retained in GitHub Actions artifacts and in the companion JSON aggregate.