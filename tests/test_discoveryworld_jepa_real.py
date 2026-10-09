"""Optional real DiscoveryWorld adapter smoke: no oracle or scorecard accesses."""
from __future__ import annotations

import importlib.util

import pytest
import numpy as np

from ecsa.benchmarks.discoveryworld.jepa_replay import encode_public_json


@pytest.mark.skipif(importlib.util.find_spec('discoveryworld') is None,
                    reason='requires the pinned optional DiscoveryWorld dependency')
def test_public_real_discoveryworld_observation_featurizes_without_oracle() -> None:
    from ecsa.benchmarks.discoveryworld.environment import DiscoveryWorldEnvironmentAdapter
    env=DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(0, max_steps=2)
    pre=env.observe()
    vector=encode_public_json(pre,width=256,kind='observation')
    assert vector.shape==(256,)
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector)>0
    # Only normal action API. The benchmark scorecard is never accessed.
    available=env.available_actions()
    assert isinstance(available,dict) and available
