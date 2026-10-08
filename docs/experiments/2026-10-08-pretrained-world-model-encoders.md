# ECSA: using a pretrained language world model as a frozen state encoder

Date: 2026-10-08. Experimental integration of published model **weights**, not
an assertion that their hidden states are already good state representations.

## Scientific question

Does a pretrained *text environment simulator* retain more useful causal/dynamic
information in its frozen hidden states than the sentence embeddings previously
tested by ECSA? Can a fixed ensemble of spline-KAN relation hypotheses discover
and generalize unknown X/Y relations more reliably? A generative world model
must also be evaluated using its **native next-observation likelihood**, because
the training objective is not sentence embedding similarity.

The upstream BehR release is
[BehR-WorldModel-Textworld-Qwen2.5-7B](https://huggingface.co/Ricardo-H/BehR-WorldModel-Textworld-Qwen2.5-7B).
It is a `Qwen2ForCausalLM` trained for next-observation simulation in TextWorld,
**not an embedding model**. In its native evaluation the input is interaction
history + next action. Our canonical JSON X/Y inputs are an important domain and
prompt-format shift; untested transfer to ECSA worlds must not be claimed.

Paper: [Beyond State Consistency: Behavior Consistency in Text-Based World
Models](https://arxiv.org/abs/2604.13824).
[Authors' evaluation/serving repo](https://github.com/Ricardo-H/behr-wm).
The public project does not include its full BehR training pipeline.

## Candidates and resource reality

| Candidate | HF ID | Role |
|---|---|---|
| BehR TextWorld 7B | `Ricardo-H/BehR-WorldModel-Textworld-Qwen2.5-7B` | first real test: pretrained world-model states + next-token likelihood |
| Qwen2.5 7B base | `Qwen/Qwen2.5-7B` | same-family/size control; separate run |
| Qwen2.5 0.5B | `Qwen/Qwen2.5-0.5B` | CPU CI interface smoke only; **not** a BehR control |
| Qwen-AgentWorld 35B-A3B | `Qwen/Qwen-AgentWorld-35B-A3B` | later remote generative simulator; **not** silently passed to Qwen2-specific hidden state extractor |

Qwen-AgentWorld can be served with vLLM or Transformers per
[official documentation](https://github.com/QwenLM/Qwen-AgentWorld).
Because it uses a different Qwen3.5 MoE architecture, it must not be treated
as interchangeable with a Qwen2 hidden-state encoder; external generation
quality and latent probing are distinct investigations.

The BehR model card lists an inherited license. It is derived from
`X1AOX1A/WorldModel-Textworld-Qwen2.5-7B` (base revision
`b052a201ae867c3058efba17c9af9cb1635d1f09`).
Verify the relevant upstream licensing terms before redistribution.

## Architecture (no OOC / no encoder training)

    Canonical raw X = observation BEFORE + action
          |
    frozen causal-LM contextual hidden states
          |
    fixed X-side pooling + projection ----------+
                                                  | -> KAN hypothesis ensemble
    fixed Y-side pooling + projection ----------+
          |
    frozen contextual states of AFTER + outcome

The pretrained causal backbone has `requires_grad=False`. Every X/Y
pooling, gating, and projection parameter is frozen throughout KAN warmup and
additional KAN head updates. The existing `TokenKANSystem` is reused;
`fit_kan_only` can update only spline-KAN heads.

For a fair experimental comparison use the same heldout transitions,
counterfactual controls, KAN population, warmup budget, random seeds,
calibration splits, and evaluation for:
`sentence_64`, `sentence_full`, `token_mean`,
`numeric_control`, `gated_attention`, `gated_kan_update`.

The 7B hidden state width and projection parameters are larger than prior
0.6B embedding experiments. They are **not** overall FLOP/parameter-matched;
compare base Qwen2.5-7B with BehR-7B first for a backbone-family control.
Even the same KAN head can inherit different fixed input projections.
Report head and projector sizes separately.

## An independent generative measurement

The model's own next-token logits are evaluated via teacher-forced

    NLL( Y | "Observation and action:" + X + "Next observation:" )

The target Y is scored only as next tokens after the entire X/action prefix,
never fed back into X. All labels used for scoring are heldout. The target's
per-token NLL is length-normalized; this is not a calibrated probability for a
physical intervention and tokenization-length artifacts remain possible.
A separate evaluation-only negative Y comes from the synthetic benchmark
oracle. Reject cases with the same original and counterfactual Y. Report
`pair_accuracy`, `mean_true_nll`, `mean_wrong_nll`, `mean_nll_margin`,
and number of skipped ambiguous outcomes.

The prompt is explicit **canonical JSON**, NOT claimed to be identical to the
TextWorld rollout template on which BehR was optimized. If this test fails,
retest using its native TextWorld conversation formatting before attributing
the failure to representation capacity.

## Offline and CI reproduction

Core tests with a tiny injected model (not pretrained):

```bash
pip install -e '.[test,neural-relations]'
pytest -q tests/test_world_model_encoders.py tests/test_token_state_relations.py
python -m ecsa.experimental.world_model_encoder_benchmark --list-candidates
```

CI additionally loads real, small pretrained `Qwen/Qwen2.5-0.5B` weights on
CPU and tests hidden-state extraction plus teacher-forced likelihood. Passing
that job establishes only integration and nonleakage, **not BehR accuracy**.

## Run actual BehR on a CUDA machine (with sufficient VRAM)

```bash
pip install -e '.[test,pretrained-encoder]'
pip install bitsandbytes
python -m ecsa.experimental.world_model_encoder_benchmark \
  --candidate behr_textworld --device cuda --4bit \
  --laws precision operator sign range_shift --seeds 0 1 2 \
  --arms sentence_64 sentence_full token_mean numeric_control \
         gated_attention gated_kan_update \
  --bootstrap 24 --adaptation 16 --calibration 12 --heldout 24 \
  --warmup-steps 45 --extra-kan-steps 24 --checkpoint-every 8 \
  --max-length 1024 --likelihood-pairs 16 \
  --output results/behr_textworld.json
```

Run the **same** command with `--candidate base_qwen25` and a distinct output.
These are separate model downloads/runs. Specify `--revision <immutable HF
commit>` for each actual result; when absent the loader records the resolved
config commit if the Transformers backend exposes it. Do not copy the upstream
base revision onto the BehR fine-tuned checkpoint.

7B BF16 full-precision weights require substantial memory; 4-bit quantized
inference may fit a consumer 12GB CUDA card at small batch/sequence length but
is not guaranteed. This code intentionally blocks accidental 7B CPU loads
unless explicitly overridden. For CPU use the small smoke candidate instead.

## Approval gates

1. **Identity and leakage**: source object IDs disjoint; no Y in model prefix
   for KAN representation; hidden-law counterfactual only after train/test split.
2. **Alignment**: compare true vs genuinely incompatible Y; exclude identical
   outcomes and report ambiguous/stochastic actions separately.
3. **Numeracy/operator probes**: precision, comparisons, signs, range shifts,
   plus paraphrase and small-number counterfactuals.
4. **Meaningful transfer**: better paired ranking than the same-family base
   checkpoint, not just better generation fluency or lower ordinary LM NLL.
5. **Cost**: accuracy versus compute, VRAM and latency; judge if BehR inference
   is viable in ECSA's candidate-selection loop.

No discoveries of causal mechanisms, universal transfer, or replacement of the
numeric channel are asserted before actual pretrained BehR scores exist.

## Scope and boundaries

No changes to ECSA's core ScienceKernel, autonomous experiment selection, or
observation contracts. The integration belongs in the optional
`src/ecsa/experimental/` layer and retains the chosen **KAN-only, frozen
encoder** architecture. The first step is instrumented evidence, not rollout
planning or causal-identification claims.
