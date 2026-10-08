# ECSA text-first transition relations — first integration gate

Date: 2026-10-08. Status: experimental neural backend with production-safe typed acquisition.

The autonomous researcher already has a domain-neutral `RawObservation`, `GroundAction` and `InteractionTransition` contract. This change adds a **whole-state** relation path in parallel with the existing `NumericRelationAcquisition`:

    X = canonical_json({"before": raw_payload_before,
                        "action": {"schema_id": id, "arguments": [...]}})
    Y = canonical_json({"after": raw_payload_after,
                        "outcome": {"success": ..., "payload": ...}})

The agent never receives Y while choosing the action. The full raw payload is retained; no manual `motor`/`reactor`/`temperature` feature names are required. The `TextRelationAcquisition` archive is bounded, deduplicated and keyed by the opaque action schema. An optional typed `TextRelationSink` can consume samples. This does **not** replace existing perception, argument grounding, action candidate discovery, LOCM, applicability learners or numerical measurements.

## Text feature backend

The reproducible first adapter uses a deterministic signed hash of Unicode tokens, token bigrams and character n-grams. It is deliberately **not** described as a semantic pretrained encoder. Two independent trainable towers project X and Y into latent spaces; a spline-KAN relation head scores compatibility `R(E_x(X), E_y(Y))`. There is also an MLP-head control. Their outputs are discrimination scores, not calibrated probability distributions over future states.

A feature-port interface permits replacing hashing with a pretrained text encoder after separate validation. Models receive the **whole** serialization, not a developer-picked numeric vector. A numeric-only ablation and a hybrid ablation use automatic numeric scanning as a control, not as the primary acquisition representation.

## Test contract and risks

- No future observation or outcome is accessible in X; Y is produced only after the environment acts.
- JSON key ordering is stable without discarding nested text, numbers, booleans or arbitrary keys. The observation envelope (observation_id and step) is metadata; payload itself is serialized in full.
- Hash n-grams preserve text/number distinctions but provide no true compositional language understanding. Identity fields **in the payload** are retained; models can still exploit spurious IDs. The benchmark blocks the easiest cross-state identity shortcut (Y contains no shared ID) and tests new X identity values; real-world entity-renaming invariance remains unproven.
- Pairwise AUROC on heldout synthetic pairs is a representation diagnostic only. It does not establish autonomous action selection, calibrated conditional Y prediction, symbolic equation discovery, or causal laws.
- Matched-vs-shuffled Y training can create false negatives, especially in many-to-one mechanisms. The evaluation negatives are selected by distant observed numeric Y values, **using heldout Y only during evaluation**. The model does not query a membership oracle during training.
- Numeric, text and hybrid feature vectors share the same dimensionality; each KAN has the same number of trained parameters and optimizer steps. Replacing the fixed n-gram encoder with a pretrained transformer would invalidate that parameter/compute comparison.

## First benchmark

Three synthetic transition families: categorical contextual rule, numerical relationship, context-numeric interaction. Train and held-out objects have disjoint opaque IDs; held-out numeric controls cover a mildly wider range. Compare text-only, numeric-only and hybrid for both KAN and MLP heads, across seeds 0/1/2, with identical data and fit budgets.

    python -m pytest -q tests/test_world_model_text_relations.py tests/test_text_first_kan.py
    python -m ecsa.experimental.text_first_kan --seeds 0 1 2 --samples 144 \
      --holdout 96 --feature-size 64 --steps 130 --output results.json

**Next gate after this experiment:** a pretrained or task-trained language encoder via `TextFeaturePort`, calibrated Y candidate generation, and closed-loop neural influence on experiment selection. Until that gate succeeds, KAN text hypotheses are observational diagnostics rather than active scientific policies.
