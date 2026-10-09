"""Shared prediction adapters must never define their own test targets."""
import pytest
from ecsa.world_model.shared_evaluation import SharedOutcomeFrame,SharedForecast
from ecsa.world_model.predictive_adapters import (
    PersistencePort,RawRidgePort,LatentReadoutPort,
)
from test_shared_outcome_frame import sample


class SyntheticLatent:
    def predict_latent(self,before,action):
        return (float(len(before.entities)), float(len(action.arguments)))


def train():
    return (sample(1,1,2),sample(2,2,4),sample(3,4,5),sample(4,2,2))


def test_neural_and_linear_backend_share_target_fingerprint():
    data=train()
    frame=SharedOutcomeFrame.fit(data)
    ridge=RawRidgePort.fit(frame,data)
    latent=LatentReadoutPort.fit(frame,data,SyntheticLatent())
    persistence=PersistencePort(frame)
    assert {ridge.frame_fingerprint,latent.frame_fingerprint,
            persistence.frame_fingerprint}=={frame.fingerprint}


def test_latent_readout_uses_train_only():
    data=train()
    frame=SharedOutcomeFrame.fit(data)
    backend=LatentReadoutPort.fit(frame,data,SyntheticLatent())
    old=backend.predict_public(data[0].before,data[0].transition.action)
    unknown=sample(10,1,9999)
    assert backend.predict_public(unknown.before,unknown.transition.action)==old
    assert backend.frame_fingerprint==frame.fingerprint


def test_persistence_outputs_exact_common_no_change():
    data=train()
    frame=SharedOutcomeFrame.fit(data)
    p=PersistencePort(frame)
    output=p.predict_public(data[0].before,data[0].transition.action)
    idx=output.coordinate_ids.index("arg[0]:number")
    coordinate=next(c for c in frame.coordinates if c.key=="arg[0]:number")
    assert output.values[idx] == -coordinate.train_center/coordinate.train_scale


def test_raw_ridge_predicts_same_scale_as_latent_port():
    data=train()
    frame=SharedOutcomeFrame.fit(data)
    a=RawRidgePort.fit(frame,data).predict_public(data[0].before,data[0].transition.action)
    b=LatentReadoutPort.fit(frame,data,SyntheticLatent()).predict_public(data[0].before,data[0].transition.action)
    assert len(a.values)==len(b.values)==len(frame.coordinates)
    assert a.coordinate_ids==b.coordinate_ids


def test_forecast_wrong_frame_or_invalid_probabilities_rejected():
    with pytest.raises(ValueError):
        SharedForecast((1.0,),(True,),("x",),"bad")
    with pytest.raises(ValueError):
        SharedForecast((float("nan"),),(True,),("x",),"frame:1")
    with pytest.raises(ValueError):
        SharedForecast((1.4,),(True,),("action:success",),"frame:1")
