"""Identical public-target protocol in unrelated wire domains."""
from ecsa.world_model.contracts import (
    RawObservation,GroundAction,RawActionOutcome,InteractionTransition,freeze_raw_value,
)
from ecsa.world_model.perception.base import (
    PerceptualObservation,ObservedFeature,
)
from ecsa.world_model.shared_evaluation import PublicOutcomeTransition,SharedOutcomeFrame
from ecsa.world_model.predictive_adapters import PersistencePort,RawRidgePort


def sample(index,name,feature,x,y):
    old=RawObservation(f"{name}-before-{index}",index,freeze_raw_value({"x":x}))
    new=RawObservation(f"{name}-after-{index}",index+1,freeze_raw_value({"x":y}))
    step=InteractionTransition(f"{name}-transition-{index}",old,GroundAction(name,()),
                               RawActionOutcome(True,freeze_raw_value({})),new)
    def perceive(item,val):
        return PerceptualObservation(item.observation_id,(),(
            ObservedFeature(feature,freeze_raw_value(val),item.observation_id),))
    return PublicOutcomeTransition(step,perceive(old,x),perceive(new,y))


def test_two_unrelated_raw_worlds_share_core_evaluation_contract():
    for name,feature in (("opaque:0","volume"),("nova!","reactivity")):
        data=tuple(sample(i,name,feature,float(i),float(i+1)) for i in range(5))
        frame=SharedOutcomeFrame.fit(data[:3])
        ridge=RawRidgePort.fit(frame,data[:3])
        persistence=PersistencePort(frame)
        assert ridge.frame_fingerprint==persistence.frame_fingerprint==frame.fingerprint
        assert len(frame.project(data[-1]).coordinate_ids)==len(frame.coordinates)


def test_test_seed_not_used_for_feature_discovery():
    source=tuple(sample(i,"A","public",i,i+1) for i in range(3))
    frame=SharedOutcomeFrame.fit(source)
    other=sample(10,"B","test-only",1,2)
    projection=frame.project(other)
    assert projection.unseen_feature_count==1
    assert "global:test-only" not in projection.coordinate_ids
