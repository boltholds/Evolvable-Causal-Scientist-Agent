"""TDD controls: witnessed negatives, train-only PCA and frozen KAN inputs."""
from __future__ import annotations

import inspect
import json
from hashlib import blake2b

import numpy as np
import pytest

torch=pytest.importorskip("torch")

from ecsa.experimental.token_state_relations import TokenStates
from ecsa.experimental.verified_relation_kan import (
    PoolKind, ProjectionKind, ReplayWitness, StudyConfig,
    ContextStatePort, extract_frozen_vectors,
    fixture_replay_witnesses, mine_repeat_supported_pairs,
    fit_train_only_projection, WideSplineKAN, RelationPopulation,
    _vectorize, _project_pairs, fit_kan_with_witnesses,
    benchmark_one,
)
from ecsa.experimental.verbalization_study import StudyLaw


class FixedEncoderStub:
    """Stable fake tensor outputs; NOT pretrained and no hidden-law oracle."""
    dimension=96
    def __init__(self):
        self.calls=0

    def encode_many(self,texts:tuple[str,...])->TokenStates:
        self.calls+=1
        token=np.zeros((len(texts),4,self.dimension),dtype=np.float32)
        mask=np.ones((len(texts),4),dtype=bool)
        sentence=np.zeros((len(texts),self.dimension),dtype=np.float32)
        for i,text in enumerate(texts):
            for j in range(4):
                digest=blake2b((str(j)+text).encode(),digest_size=16).digest()
                rng=np.random.default_rng(int.from_bytes(digest[:8],"little"))
                token[i,j]=rng.normal(0,.1,self.dimension)
            sentence[i]=token[i,-1]
            sentence[i]/=max(np.linalg.norm(sentence[i]),1e-12)
        return TokenStates(token,mask,sentence)


def contract_witness(experiment_id,y,*,contract=True,replays=3,
                     epoch="v1",schema="act"):
    return ReplayWitness(experiment_id,epoch,schema,
                         '{"before":{"id":"'+experiment_id+'"},"action":"go"}',
                         tuple([y]*replays),contract)


def test_repeated_deterministic_alternatives_exclude_false_binary_negatives():
    replay=(
        contract_witness("a",'{"measurement":1}'),
        contract_witness("b",'{"measurement":1}'),
        contract_witness("c",'{"measurement":-1}'),
        contract_witness("d",'{"measurement":-1}'),
    )
    mined=mine_repeat_supported_pairs(replay,seed=4)
    assert mined.audit.validated_pairs==4
    assert mined.audit.shuffled_false_negative_fraction==.5
    assert all(x.y_true!=x.y_alternative for x in mined.samples)
    assert len({x.positive_experiment_id for x in mined.samples})==4
    assert all(x.positive_experiment_id!=x.alternative_experiment_id
               for x in mined.samples)


def test_unknown_or_stochastic_world_abstains():
    records=(
        contract_witness("a","positive",contract=False),
        ReplayWitness("b","v1","act","B",("positive","negative","positive"),True),
        contract_witness("c","negative",replays=1),
    )
    mined=mine_repeat_supported_pairs(records,seed=0,minimum_replays=3)
    assert mined.samples==()
    assert mined.audit.unstable_or_unsupported_witnesses==3
    assert mined.audit.shuffled_false_negative_fraction is None


def test_same_outcome_or_schema_epoch_never_yields_negative_label():
    same=tuple(contract_witness(s,"identical") for s in ("a","b","c"))
    mined=mine_repeat_supported_pairs(same,seed=0)
    assert mined.samples==()
    assert mined.audit.no_disjoint_alternative==3
    different=(
        contract_witness("a","positive",schema="one"),
        contract_witness("b","negative",schema="other"),
        contract_witness("c","negative",epoch="other"),
    )
    assert mine_repeat_supported_pairs(different,seed=2).samples==()


def test_conflicting_duplicate_replay_groups_rejected():
    one=contract_witness("a","positive")
    other=ReplayWitness("b","v1","act",one.x_text,("negative",)*3,True)
    with pytest.raises(ValueError,match="duplicate X"):
        mine_repeat_supported_pairs((one,other),seed=0)


def test_fixture_only_generates_true_repeated_world_steps_and_disjoint_splits():
    for law in (StudyLaw.PRECISION,StudyLaw.OPERATOR):
        examples=fixture_replay_witnesses(law,51,40,heldout=False,repetitions=3)
        held=fixture_replay_witnesses(law,151,24,heldout=True,repetitions=3)
        assert len(examples)==40
        assert all(len(set(row.observed_y_replays))==1 for row in examples)
        assert all(row.deterministic_contract for row in examples)
        assert not {x.experiment_id for x in examples}&{x.experiment_id for x in held}
        mined=mine_repeat_supported_pairs(examples,seed=5)
        assert mined.audit.validated_pairs>10
        assert all(r.y_true!=r.y_alternative for r in mined.samples)
        assert all('"after"' not in r.x_text for r in mined.samples)


def test_pair_miner_never_calls_hidden_law_or_counterfactual_generator():
    import ecsa.experimental.verified_relation_kan as m
    src=inspect.getsource(m.mine_repeat_supported_pairs)
    assert "_law_response" not in src
    assert "_negative_law_outcomes" not in src
    assert "oracle" not in src
    assert "random permutation" in m.mine_repeat_supported_pairs.__doc__
    # In contrast, the fixture generator is the isolated environment backend.
    assert "_law_response" in inspect.getsource(m.fixture_replay_witnesses)


@pytest.mark.parametrize("projection",list(ProjectionKind))
@pytest.mark.parametrize("latent",[2,8,16,32])
def test_fitted_projection_has_train_only_moments_and_variable_width(projection,latent):
    rng=np.random.default_rng(102)
    x=rng.normal(0,1,(48,96)).astype(np.float32)
    y=rng.normal(0,1,(48,96)).astype(np.float32)
    fitted=fit_train_only_projection(x,y,kind=projection,width=latent,seed=2)
    assert fitted.x_basis.shape==(96,latent)
    assert fitted.y_basis.shape==(96,latent)
    assert fitted.fitted_sample_count==48
    xx,yy=fitted.apply(x,y)
    assert xx.shape==yy.shape==(48,latent)
    assert np.isfinite(xx).all() and np.isfinite(yy).all()
    before=fitted.x_center.copy()
    fitted.apply(rng.normal(999,1,(12,96)).astype(np.float32),
                 rng.normal(-999,1,(12,96)).astype(np.float32))
    assert np.array_equal(before,fitted.x_center)
    with pytest.raises(ValueError,match="encoder width"):
        fitted.apply(np.ones((5,97),np.float32),np.ones((5,96),np.float32))


def test_wide_kan_input_and_parameter_counts_change_with_latent_width():
    count={}
    for width in (2,8,16,32):
        model=RelationPopulation(width)
        score=model.logits(torch.randn(7,width),torch.randn(7,width))
        assert score.shape==(3,7)
        count[width]=sum(p.numel() for p in model.parameters())
    assert all(count[a]<count[b] for a,b in zip((2,8,16),(8,16,32)))


def test_real_contrastive_training_updates_kan_not_port():
    port=FixedEncoderStub()
    train=fixture_replay_witnesses(StudyLaw.OPERATOR,12,32,heldout=False,repetitions=3)
    cal=fixture_replay_witnesses(StudyLaw.OPERATOR,50,16,heldout=False,repetitions=3)
    positive=mine_repeat_supported_pairs(train,seed=0)
    control=mine_repeat_supported_pairs(cal,seed=1)
    tr=_vectorize(port,positive.samples,PoolKind.SENTENCE_FULL)
    cv=_vectorize(port,control.samples,PoolKind.SENTENCE_FULL)
    fitted=fit_train_only_projection(tr[0],tr[1],kind=ProjectionKind.PCA,
                                     width=8,seed=3)
    model,loss=fit_kan_with_witnesses(
        _project_pairs(fitted,tr),_project_pairs(fitted,cv),
        dimension=8,seed=0,
        config=StudyConfig(32,16,12,3,12,0.009,3))
    assert np.isfinite(loss)
    assert port.calls==2
    assert sum(p.numel() for p in model.parameters())>0
    assert len(model.hypotheses)==3
    assert all(p.grad is None or np.isfinite(p.grad.detach().numpy()).all()
               for p in model.parameters())


def test_two_laws_two_projections_with_same_heldout_samples():
    fake=FixedEncoderStub()
    cfg=StudyConfig(train_count=32,calibration_count=16,
                    heldout_count=12,replays_per_x=3,warmup_steps=10,
                    checkpoint_every=5)
    results=[]
    for law in (StudyLaw.PRECISION,StudyLaw.OPERATOR):
        rows=benchmark_one(
            port=fake,law=law,seed=0,config=cfg,
            pools=(PoolKind.SENTENCE_FULL,PoolKind.TOKEN_MEAN),
            projections=(ProjectionKind.RANDOM,ProjectionKind.PCA),
            dimensions=(2,8),
        )
        assert len(rows)==8
        assert all(r.train_pairs>0 and r.heldout_pairs>0 for r in rows)
        assert all(0<=r.heldout_auroc<=1 for r in rows)
        assert all(r.identity_overlap==0 for r in rows)
        assert {r.dimension for r in rows}=={2,8}
        assert {r.projection for r in rows}==set(ProjectionKind)
        assert all(r.frozen_encoder_width==96 for r in rows)
        results.extend(rows)
    assert len(results)==16


def test_unsupported_latent_dimension_raises():
    a=np.zeros((8,16),np.float32)
    with pytest.raises(ValueError):
        fit_train_only_projection(a,a,kind=ProjectionKind.PCA,width=32,seed=1)
    with pytest.raises(ValueError):
        fit_train_only_projection(a,a,kind=ProjectionKind.PCA,width=1,seed=1)
    with pytest.raises(TypeError):
        extract_frozen_vectors(FixedEncoderStub(),("X",),"sentence_full")
