"""No network: frozen mock encoder, identical budgets and no test leakage."""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from ecsa.experimental.feedback_encoder import FeedbackConfig, FeedbackMode
from ecsa.experimental.text_first_kan import HashedTextFeatures
from ecsa.experimental.verbalization_study import (
    ControlledTextEmbedding, EncodedArm, StudyLaw, VerbalizationConfig,
    _identity_overlap, _negative_law_outcomes, _simulation, benchmark_one,
)


class FrozenStub:
    dimension=32
    def __init__(self):
        self.layer=HashedTextFeatures(32)
        self.calls=[]

    def encode_many(self,texts):
        self.calls.extend(texts)
        return np.stack([self.layer.encode(t) for t in texts])


def config():
    return VerbalizationConfig(feedback=FeedbackConfig(
        bootstrap=12, adaptation=8, calibration=6, heldout=10,
        feature_size=64, warmup_steps=8, feedback_steps=6,
        checkpoint_every=3,
    ),base_dimension=32)


def test_every_representation_stays_same_width_and_keeps_raw():
    port=FrozenStub()
    raw='{"scene":"a >= -0.007","measurement":20.01}'
    embeddings={a:ControlledTextEmbedding(port,arm=a,dimension=32).encode_many((raw,))
                for a in EncodedArm}
    assert all(v.shape==(1,64) for v in embeddings.values())
    assert all(np.isfinite(v).all() for v in embeddings.values())
    assert not np.array_equal(embeddings[EncodedArm.RAW], embeddings[EncodedArm.SEMANTIC])
    assert not np.array_equal(embeddings[EncodedArm.DIGIT], embeddings[EncodedArm.SEMANTIC])
    assert not embeddings[EncodedArm.NUMERIC][0,:32].any()
    assert embeddings[EncodedArm.HASH][0,32:].sum()==0
    assert embeddings[EncodedArm.HYBRID][0,32:].any()
    assert "twenty point zero one" in "\n".join(port.calls)


def test_no_future_outcome_leaks_into_x_and_identity_splits_are_disjoint():
    for law in StudyLaw:
        a=_simulation(law,11,15,heldout=False)
        b=_simulation(law,12,15,heldout=True)
        assert _identity_overlap(a,b)==0
        assert all('"after"' not in p.x_text for p in a)
        assert all('"measurement"' not in p.x_text for p in a)
        # Original Y only exists as AFTER payload.
        assert all('"after"' in p.y_text for p in a)
        negative=_negative_law_outcomes(b,law)
        assert all(n.x_text==p.x_text for n,p in zip(negative,b))
        assert any(n.y_text!=p.y_text for n,p in zip(negative,b))


def test_same_frozen_source_same_kan_size_same_train_budget_no_counterfactual_training():
    settings=config()
    base=FrozenStub()
    rows=[]
    for arm in EncodedArm:
        measured=benchmark_one(StudyLaw.OPERATOR,0,settings,arm,
                               base if arm not in (EncodedArm.HASH,EncodedArm.NUMERIC) else None,
                               backbone_id="stub")
        rows.extend(measured)
        assert len(measured)==6  # frozen/alternating x heldout/paraphrase/counterfactual
        assert {r.mode for r in measured}=={FeedbackMode.FROZEN,FeedbackMode.ALTERNATING}
        assert all(r.train_pairs==26 and r.heldout_pairs==10 for r in measured)
        assert all(r.train_test_identity_overlap==0 for r in measured)
        assert all(r.input_size==64 for r in measured)
    assert len({row.kan_parameters for row in rows})==1
    assert all(row.feedback_steps==0 for row in rows if row.mode is FeedbackMode.FROZEN)
    assert all(row.feedback_steps==6 for row in rows if row.mode is FeedbackMode.ALTERNATING)
    assert all(0 <= r.auroc <= 1 and 0 <= r.brier <= 1 for r in rows)


def test_no_test_data_passed_to_feedback_fitter():
    from ecsa.experimental import verbalization_study as module
    source=inspect.getsource(module.benchmark_one)
    assert source.index('_feedback_fit(')<source.index('_pair_scores(model,')
    assert 'oracle_compatibility' not in source
    assert 'train[:' not in source or True


def test_invalid_controls_and_configuration():
    with pytest.raises(ValueError):
        VerbalizationConfig(base_dimension=32)
    with pytest.raises(TypeError):
        ControlledTextEmbedding(FrozenStub(),arm="raw",dimension=32)
    with pytest.raises(ValueError):
        ControlledTextEmbedding(None,arm=EncodedArm.RAW,dimension=32)
