"""Frozen numerical/symbol verbalization fixture gates; feedback retired."""
from __future__ import annotations

import json

import numpy as np
import pytest
pytest.importorskip("torch")
from ecsa.experimental.text_first_kan import HashedTextFeatures
from ecsa.experimental.verbalization_study import (
    ControlledTextEmbedding, EncodedArm, StudyLaw,
    _identity_overlap, _negative_law_outcomes, _simulation,
    _paraphrase, _token_audit,
)


class FrozenStub:
    dimension=32
    def __init__(self):
        self.layer=HashedTextFeatures(32)
        self.calls=[]

    def encode_many(self,texts):
        self.calls.extend(texts)
        return np.stack([self.layer.encode(t) for t in texts])


def test_all_repr_keep_fixed_width_and_numbers():
    port=FrozenStub()
    raw='{"scene":"a >= -0.007","measurement":20.01}'
    embeddings={
        arm:ControlledTextEmbedding(port,arm=arm,dimension=32).encode_many((raw,))
        for arm in EncodedArm
    }
    assert all(matrix.shape==(1,64) for matrix in embeddings.values())
    assert all(np.isfinite(matrix).all() for matrix in embeddings.values())
    assert not np.array_equal(embeddings[EncodedArm.RAW],embeddings[EncodedArm.SEMANTIC])
    assert embeddings[EncodedArm.NUMERIC][0,:32].sum()==0
    assert embeddings[EncodedArm.HYBRID][0,32:].any()
    assert any("twenty point zero one" in t for t in port.calls)


@pytest.mark.parametrize("law",list(StudyLaw))
def test_observed_transitions_and_eval_counterfactuals_do_not_leak(law):
    train=_simulation(law,11,15,heldout=False)
    test=_simulation(law,12,15,heldout=True)
    assert _identity_overlap(train,test)==0
    assert all('"after"' not in row.x_text for row in train)
    assert all('"measurement"' not in row.x_text for row in train)
    wrong=_negative_law_outcomes(test,law)
    assert all(a.x_text==b.x_text for a,b in zip(test,wrong))
    assert any(a.y_text!=b.y_text for a,b in zip(test,wrong))
    variants=_paraphrase(test)
    assert len(variants)==len(test)
    assert all(json.loads(p.y_text)["after"]["observation"]=="signal captured"
               for p in variants)


def test_verbalization_module_exposes_no_training_or_feedback():
    from ecsa.experimental import verbalization_study as m
    assert not hasattr(m,"benchmark_one")
    assert not hasattr(m,"FeedbackMode")
    assert not hasattr(m,"_feedback_fit")


def test_invalid_arms_and_missing_backbone_fail():
    with pytest.raises(TypeError):
        ControlledTextEmbedding(FrozenStub(),arm="raw",dimension=32)
    with pytest.raises(ValueError):
        ControlledTextEmbedding(None,arm=EncodedArm.RAW,dimension=32)
