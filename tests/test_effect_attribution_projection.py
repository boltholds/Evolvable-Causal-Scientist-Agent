"""Generic object/action trial projection without benchmark-specific names."""
from ecsa.world_model.contracts import (
    GroundAction, RawObservation, RawActionOutcome, InteractionTransition,
    freeze_raw_value,
)
from ecsa.world_model.grounding import InteractionGrounder
from ecsa.world_model.perception.base import EntityObservation, ObservedFeature, PerceptualObservation
from ecsa.world_model.effect_attribution import EffectTrialProjector


def sample(index, *, uuid=11, before=1, after=2, schema="Z9",
           omit_after=False, duplicated=False):
    act=GroundAction(schema,(freeze_raw_value(uuid),))
    x=RawObservation(f"raw-x-{index}",0,freeze_raw_value({}))
    y=RawObservation(f"raw-y-{index}",1,freeze_raw_value({}))
    transition=InteractionTransition(f"evidence-{index}",x,act,RawActionOutcome(True,freeze_raw_value({})),y)
    def feature(value,identity,mark):
        return EntityObservation(
            f"slot-{mark}-{identity}",str(identity),
            (ObservedFeature("charge",freeze_raw_value(value),f"prov-{mark}-{identity}"),),
            f"prov-{mark}-{identity}"
        )
    p=PerceptualObservation(x.observation_id,(feature(before,uuid,"before"),),())
    right=[] if omit_after else [feature(after,uuid,"after")]
    if duplicated and not omit_after:right.append(feature(after,uuid,"another"))
    q=PerceptualObservation(y.observation_id,tuple(right),())
    return transition,InteractionGrounder().observe(transition,p,q),p,q


def test_argument_binding_is_role_relative_across_uuid_renaming():
    projector=EffectTrialProjector()
    a=projector.project(*sample(1,uuid=11))
    b=projector.project(*sample(2,uuid=999))
    assert len(a)==len(b)==1
    assert a[0].effect_key==b[0].effect_key=="arg[0]:charge"
    assert a[0].argument_role=="arg[0]"
    assert a[0].changed is True and a[0].observed
    assert "999" not in a[0].effect_key+b[0].effect_key


def test_missing_post_feature_is_not_no_change():
    trials=EffectTrialProjector().project(*sample(3,omit_after=True))
    assert len(trials)==1
    assert not trials[0].observed
    assert trials[0].changed is None


def test_unchanged_observed_feature_produces_explicit_negative_trial():
    trials=EffectTrialProjector().project(*sample(4,before=2,after=2))
    assert len(trials)==1
    assert trials[0].observed
    assert trials[0].changed is False


def test_ambiguous_identity_does_not_fabricate_match():
    trials=EffectTrialProjector().project(*sample(5,duplicated=True))
    assert len(trials)==1
    assert trials[0].changed is None


def test_schema_name_has_no_domain_semantics():
    a=EffectTrialProjector().project(*sample(6,schema="!α?"))[0]
    assert a.action_schema_id=="!α?"
    assert a.effect_key=="arg[0]:charge"


def test_observable_global_change_always_has_trial():
    tr,g,p,q=sample(7)
    p=PerceptualObservation(p.observation_id,p.entities,(
        ObservedFeature("ambient",freeze_raw_value(1),"p-global"),))
    q=PerceptualObservation(q.observation_id,q.entities,(
        ObservedFeature("ambient",freeze_raw_value(2),"q-global"),))
    grounded=InteractionGrounder().observe(tr,p,q)
    events=EffectTrialProjector().project(tr,grounded,p,q)
    assert {v.effect_key for v in events}=={"arg[0]:charge","global:ambient"}
