"""Frozen role-relative prediction targets, independent of JEPA coordinates."""
from ecsa.world_model.contracts import (
    GroundAction,InteractionTransition,RawObservation,RawActionOutcome,freeze_raw_value
)
from ecsa.world_model.perception.base import (
    EntityObservation,ObservedFeature,PerceptualObservation
)
from ecsa.world_model.shared_evaluation import (
    PublicOutcomeTransition,SharedOutcomeFrame,TargetKind,
)


def sample(n, before=1, after=2, *, uuid=10, missing=False,
           categorical=None, categorical_after=None, extra=None):
    x=RawObservation(f"x-{n}",n,freeze_raw_value({}))
    y=RawObservation(f"y-{n}",n+1,freeze_raw_value({}))
    action=GroundAction("operation",(freeze_raw_value(uuid),))
    transition=InteractionTransition(f"sample-{n}",x,action,
                        RawActionOutcome(True,freeze_raw_value({})),y)
    def entity(value,post):
        if post and missing:
            return ()
        features=[ObservedFeature("number",freeze_raw_value(value),f"entity-evidence-{n}-{post}")]
        if categorical is not None:
            features.append(ObservedFeature("mode",
                 freeze_raw_value(categorical_after if post else categorical),
                 f"mode-{n}-{post}"))
        return (EntityObservation(f"local-{n}-{post}",str(uuid),tuple(features),
                                  f"entity-{n}-{post}"),)
    before_obs=PerceptualObservation(x.observation_id,entity(before,False),())
    post_globals=(ObservedFeature("new-field",freeze_raw_value(extra),f"extra-{n}"),) if extra is not None else ()
    after_obs=PerceptualObservation(y.observation_id,entity(after,True),post_globals)
    return PublicOutcomeTransition(transition,before_obs,after_obs)


def test_training_only_frame_keeps_identical_fingerprint_on_heldout():
    frame=SharedOutcomeFrame.fit((sample(1),sample(2,1,3),sample(3,1,2)))
    orig=frame.fingerprint
    frame.project(sample(4,1,123))
    assert frame.fingerprint==orig
    assert frame.coordinates


def test_test_only_feature_does_not_extend_vocabulary():
    frame=SharedOutcomeFrame.fit((sample(1),sample(2)))
    old_ids=tuple(c.key for c in frame.coordinates)
    test=frame.project(sample(4,extra=123))
    assert test.unseen_feature_count>=1
    assert tuple(c.key for c in frame.coordinates)==old_ids


def test_numeric_scalers_frozen_and_constant_safe():
    frame=SharedOutcomeFrame.fit((sample(1,1,2),sample(2,1,2)))
    number=next(c for c in frame.coordinates if c.key=="arg[0]:number")
    assert number.kind is TargetKind.NUMERIC_DELTA
    assert number.train_scale>0
    assert number.train_center==1.0
    result=frame.project(sample(5,1,1000))
    assert result.observed_mask[result.coordinate_ids.index(number.key)]
    assert frame.fingerprint==frame.fingerprint


def test_missing_is_not_persistence():
    frame=SharedOutcomeFrame.fit((sample(1),sample(2)))
    test=frame.project(sample(3,missing=True))
    idx=test.coordinate_ids.index("arg[0]:number")
    assert test.observed_mask[idx] is False
    assert test.values[idx]==0.0


def test_object_renaming_does_not_change_role_target():
    frame=SharedOutcomeFrame.fit((sample(1,uuid=11),sample(2,uuid=12)))
    a=frame.project(sample(3,uuid=900))
    b=frame.project(sample(4,uuid=901))
    assert a.coordinate_ids==b.coordinate_ids
    assert a.values==b.values


def test_categorical_and_numeric_targets_have_distinct_kinds():
    frame=SharedOutcomeFrame.fit((
        sample(1,categorical="off",categorical_after="on"),
        sample(2,categorical="off",categorical_after="off"),
    ))
    kinds={c.key:c.kind for c in frame.coordinates}
    assert kinds["arg[0]:number"] is TargetKind.NUMERIC_DELTA
    assert kinds["arg[0]:mode"] is TargetKind.CATEGORICAL_CHANGE
    assert kinds["action:success"] is TargetKind.ACTION_SUCCESS


def test_public_frame_order_and_duplicate_training_ids():
    import pytest
    with pytest.raises(ValueError,match="duplicate"):
        SharedOutcomeFrame.fit((sample(1),sample(1)))
