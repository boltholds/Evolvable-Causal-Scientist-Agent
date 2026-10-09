"""Evaluator-only third dynamics fixture; never imported by MechanismRouter.

Rules are held by the simulator and scoring layer; model receives only
observations, one-hot actions and subsequent observations, with a common
sensor geometry and independent observation nuisance across epochs.
"""
from __future__ import annotations

from enum import StrEnum

import numpy as np

from .jepa_world import Action, Mechanism, OpaqueSwitchWorld, TransitionDataset, _ACTIONS, action_vector


class RouterFixtureLaw(StrEnum):
    ADDITIVE='additive'
    GATED='gated'
    REVERSE_GATED='reverse_gated'


class _ThreeLawWorld(OpaqueSwitchWorld):
    def __init__(self,*,law:RouterFixtureLaw,seed:int,noise_std:float):
        if not isinstance(law,RouterFixtureLaw):
            raise TypeError('evaluator law enum required')
        super().__init__(mechanism=Mechanism.ADDITIVE,seed=seed,noise_std=noise_std)
        self._fixture_law=law

    def _set_fixture_law(self,law:RouterFixtureLaw)->None:
        if not isinstance(law,RouterFixtureLaw):raise TypeError('fixture mechanism required')
        self._fixture_law=law

    def step(self,action:Action)->np.ndarray:
        if not isinstance(action,Action):raise TypeError('typed action required')
        if self._fixture_law is RouterFixtureLaw.REVERSE_GATED:
            if action is Action.PULSE and self._gate:
                self._level-=.36
            elif action is Action.FLIP:
                self._gate=not self._gate
            return self._observe()
        self._mechanism=(Mechanism.ADDITIVE if self._fixture_law is RouterFixtureLaw.ADDITIVE
                         else Mechanism.GATED)
        return super().step(action)


def collect_router_transitions(*,law:RouterFixtureLaw,count:int,seed:int,noise_std:float=.16,
                               action_bias:Action|None=None) -> TransitionDataset:
    if not isinstance(law,RouterFixtureLaw):raise TypeError('fixture law required')
    if type(count) is not int or count<1 or type(seed) is not int or seed<0:
        raise ValueError('positive count and nonnegative seed required')
    if action_bias is not None and not isinstance(action_bias,Action):
        raise TypeError('typed action bias required')
    rng=np.random.default_rng(seed+100_019)
    world=_ThreeLawWorld(law=law,seed=seed,noise_std=noise_std)
    before,actions,after=[],[],[]
    for i in range(count):
        observed=world.reset(trial_seed=seed*1_000_003+i)
        action=_ACTIONS[int(rng.integers(len(_ACTIONS)))]
        if action_bias is not None and i%2==0:
            action=action_bias
        before.append(observed)
        actions.append(action_vector(action))
        after.append(world.step(action))
    return TransitionDataset(np.asarray(before,dtype=np.float32),
                             np.asarray(actions,dtype=np.float32),
                             np.asarray(after,dtype=np.float32))


def generate_blinded_stream(*,laws:tuple[RouterFixtureLaw,...],phase_length:int,
                            seed:int,noise_std:float=.16)->TransitionDataset:
    if not isinstance(laws,tuple) or not laws or not all(isinstance(l,RouterFixtureLaw) for l in laws):
        raise TypeError('fixture-only phases must be immutable typed laws')
    if type(phase_length) is not int or phase_length<8 or type(seed) is not int or seed<0:
        raise ValueError('invalid fixture length or seed')
    world=_ThreeLawWorld(law=laws[0],seed=50_000+seed,noise_std=noise_std)
    total=phase_length*len(laws)
    # Phase-local RNG preserves the past when future phases are appended.
    schedule=np.concatenate(tuple(
        np.random.default_rng(17_000+seed*101+phase).permutation(
            np.tile(np.arange(len(_ACTIONS)),
                    (phase_length+len(_ACTIONS)-1)//len(_ACTIONS))
        )[:phase_length]
        for phase in range(len(laws))
    ))
    before,actions,after=[],[],[]
    for index,i in enumerate(schedule):
        world._set_fixture_law(laws[index//phase_length])
        act=_ACTIONS[int(i)]
        before.append(world.reset(trial_seed=seed*2_000_003+index))
        actions.append(action_vector(act))
        after.append(world.step(act))
    return TransitionDataset(np.asarray(before,dtype=np.float32),
                             np.asarray(actions,dtype=np.float32),
                             np.asarray(after,dtype=np.float32))
