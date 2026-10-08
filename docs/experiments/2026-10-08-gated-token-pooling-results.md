# Gated token pooling with KAN feedback — experimental results

Date: 2026-10-08. Code commit: 9027ba0bd948031cc233d2d6f2eb7faf0045e4a7.

## Research question

After finding that free-form attention pooling underperformed masked token mean
on the first real-Qwen tests, test whether a small, trainable, mean-anchored
attention residual improves relation recognition and whether extra training
should update KAN or propagate KAN errors to the pooling adapters.

The new gated model begins at alpha = 0.5 * sigmoid(-5) ~ 0.00335 in:

    pooled = normalize(masked_mean + alpha*(masked_attention - masked_mean))

The Qwen backbone and native BPE tokenizer remain frozen. The full hidden
sequence is available to the pooler. Initial KAN heads and X/Y projection
weights are matched between masked-mean and gated controls. Both branches
have independent pooling/adapters.

## Reproducibility

[Successful gated-token CI run](https://github.com/boltholds/Evolvable-Causal-Scientist-Agent/actions/runs/37790594340)

- Real frozen Qwen3-Embedding-0.6B, exact pinned model revision
  3d106eabb5535a84de3ae88f45887a78259b52de.
- **New Qwen seed 1**, four synthetic mechanisms; each with 24 bootstrap,
  12 online adaptation, 8 calibration, and 16 disjoint heldout pairs.
- MiniLM seeds 3,4,5, four mechanisms; 24 bootstrap, 16 adaptation,
  12 calibration, 24 heldout pairs.
- Compared full and truncated sentence vectors, masked mean, old attention,
  gated attention, gated + head-only updates, gated + KAN->attention feedback,
  and the existing numeric-hash control.
- Same action/outcome pairs and KAN architecture within each law/seed.
  Attention arms have additional trainable parameters, so there is no
  global parameter/FLOP match. Calibration partitions determine checkpoint
  acceptance; evaluation labels are never training inputs.
- Primary outcome: counterfactual heldout AUROC of positive/negative
  X/Y compatibility. This is **not** calibrated p(Y|do(X)) or a causal test.

### Mean counterfactual AUROC, MiniLM (12 law/seed experiments)

| Representation | AUROC |
| --- | ---: |
| Sentence 64 | 0.5366 |
| Sentence full | 0.5627 |
| Token masked mean | 0.5537 |
| Original token attention | 0.5247 |
| Gated attention | 0.5081 |
| Gated + KAN-only updates | 0.5074 |
| Gated + KAN-to-adapter feedback | 0.5156 |
| Numeric control | 0.7046 |

**Gated attention did NOT improve MiniLM.**

### Counterfactual AUROC, real Qwen (new seed 1)

| Law | Masked mean | Old attention | Gated | Gated + KAN update | Gated + feedback | Numeric |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Precision | 0.4727 | 0.4336 | 0.5430 | 0.5352 | 0.5312 | 0.5625 |
| Operator | 0.6484 | 0.5273 | 0.6328 | 0.6641 | 0.6836 | 0.5039 |
| Range shift | 0.5859 | 0.5938 | 0.5898 | 0.6055 | 0.5820 | 0.9707 |
| Sign | 0.8320 | 0.8477 | 0.8750 | 0.9062 | 0.8242 | 1.0000 |
| **Mean** | **0.6348** | **0.6006** | **0.6602** | **0.6777** | **0.6553** | **0.7593** |

Qwen gated + KAN-only outperformed masked mean by about +0.043 AUROC on
this one new seed. However, feedback to attention was weaker than
head-only adaptation; the numeric control remained strongest on average.
There is **no demonstrated universal advantage**, no confidence interval
from independent Qwen seeds, and no reason yet to replace the numeric path.

All three gating workflow jobs succeeded; the associated full ECSA CI
reported 240 passed, 12 skipped. The separate prior seven-arm experiment
remains reproducible with unchanged defaults.

## Critical data-quality diagnostic

The training objective in these experiments labels a *randomly shifted Y*
as incompatible with a given X. In deterministic synthetic fixtures,
many mismatched pairs remain valid or nearly equivalent to the actual
outcome. An observational-only diagnostic over 12 synthetic seeds and
all cyclic shifts counted the fraction of shifted outcomes within 0.05
of the observed numerical Y:

- Precision: 0.160
- Operator: 0.496
- Range shift: 0.064
- Sign: 0.458

Hence nearly half the nominal negatives in operator/sign experiments
are ambiguous or contradictory. This is a serious confound: improving
attention alone cannot clean the labels. A small offline test that
excluded exact embedding duplicates did not consistently improve AUROC;
do not treat this as a solved issue.

## Conclusion / next acceptance gate

Keep masked mean and the numeric baseline; retain gated attention as an
experimental option. Do NOT claim the feedback loop is better until it
outperforms KAN-only updates with a clean, non-leaky contrastive objective
under multi-seed/Qwen testing.

Next test: address contradictory negative pairs through repeated
interventions or a multi-positive conditional objective that does not
label alternative valid Y values as false. Evaluate heldout outcomes
prequentially and report both representation- and objective-level
ablations. Avoid introducing oracle-generated counterfactuals into
training; their use is confined to benchmark-only evaluation.
