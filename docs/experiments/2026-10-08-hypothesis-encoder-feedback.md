# Hypothesis-conditioned encoder negative feedback — ECSA, 2026-10-08

## Question

Can several competing implicit spline-KAN relation hypotheses provide a useful error signal to a *trainable adapter* on top of a **frozen text encoder**, while preserving previously acquired categorical and numerical distinctions? The motivating negative result was short LoRA training that improved standard relation scores but damaged a categorical counterfactual test.

The architecture remains `X → frozen text E_X → adapter X → KAN hypothesis population ← adapter Y ← frozen text E_Y ← Y`. Only `do(X)`-like factual transitions contribute positive training pairs; the system learns contrastive relation compatibility, not a known physical equation or a conditional world model. The X input contains only the state before the action and the action itself. Y is available only after observation.

## Three controlled ablations (plus no-feedback reference)

- **Frozen**: warmup adapters + KAN frozen after bootstrap.
- **KAN-only**: adapt all three KAN heads; freeze adapter weights.
- **Adapter-only**: adapt X/Y projection towers through KAN gradients; freeze KAN weights.
- **Alternating**: alternate KAN and adapter optimizer steps. All adaptive arms have *the same total number* of optimizer steps (KAN-only and adapter-only receive twice as many updates to their respective parameter family as alternating does; this is explicitly reported).

All arms start from the **same initialized/warmup-trained weights**, use the **same embedded X/Y observations** and identical heldout transitions. The frozen pretrained model is never updated. The use of the existing two-sided KAN relation is deliberate: the model scores an X,Y pair, not a direct Y=f(X) regression.

## Feedback mechanism and protection

1. Pretrain all heads and separate X/Y adapters on bootstrap matched pairs versus roll-shuffled Y.
2. Score each candidate KAN hypothesis on a **separate calibration partition**; form clipped softmax empirical validation weights. These are *not* Bayesian posteriors, causal beliefs, or calibrated Y probabilities.
3. In the adapter steps backpropagate the weighted contrastive relation loss through the frozen KAN operators to the X/Y adapters. Add variance anti-collapse loss, pointwise distillation from initial adapter outputs and relational-geometry preservation (adjacent-pair distances).
4. Every checkpoint recompute empirical hypothesis weights in the **current** latent coordinate system; accept only when calibration NLL has not degraded beyond a predefined tolerance and latent spread has not collapsed, otherwise rollback the entire checkpoint.
5. Increment separate adapter/hypothesis versions only for accepted checkpoints. Never reuse weights from an old representation version. Heldout examples do not enter optimizer or calibration gates.

## Benchmarks and metrics

Reuse existing raw-world `FullTextTransitionProjector` samples from categorical, numerical and category×number interaction fixtures. All train/eval object IDs are disjoint; test data includes shifted numerical support and a paraphrased text domain.

Report AUROC, within-X paired ranking and binary Brier on heldout *paired compatibility* discrimination, plus separate tests: (a) opposite categorical state with **identical numeric X**, (b) materially perturbed observed Y while retaining X, and (c) paraphrase robustness. They are not a calibrated p(Y | do(X)) or a proof of causal law recovery.

Frozen/positive-vs-shuffled training has unavoidable false negatives for intrinsically many-to-many mechanisms. Calibration pairs may also be noisy. The model uses small 2D adapters and a simplified fixed-grid spline KAN, not full KAN 2.0. The calibration gate is a heuristic with no formal statistical guarantee.

## Reproduce

    pip install -e '.[test,pretrained-encoder]'
    python -m pytest -q tests/test_feedback_encoder.py
    python -m ecsa.experimental.feedback_encoder \
      --model sentence-transformers/all-MiniLM-L6-v2 \
      --laws categorical numeric interaction --seeds 0 1 2 \
      --bootstrap 48 --adaptation 32 --calibration 16 --heldout 32 \
      --dimension 64 --warmup-steps 65 --feedback-steps 40 \
      --output results/feedback_minilm.json

An additional CPU GitHub Actions job tests **actual pretrained** `Qwen/Qwen3-Embedding-0.6B` pinned at `3d106eabb5535a84de3ae88f45887a78259b52de`, on one interaction seed due resource limits. It updates **only** the adapters and/or KAN heads, not the pretrained backbone or LoRA weights.

**Acceptance criterion:** It is a functional closed feedback loop if nonzero gradients flow through a frozen KAN into adapters, the version/rollback gates work and no heldout labels enter updates. Any claim that feedback is *useful* must be based on paired real-encoder metrics vs KAN-only, preferably on more seeds. No performance win is assumed.
