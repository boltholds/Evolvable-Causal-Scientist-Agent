"""Regression and causal experiment-design gates for mechanism shifts."""
from __future__ import annotations

import inspect
import math

import pytest
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from ecsa.experimental.mechanism_shift import (
    NewMechanism, ShiftArm, ShiftConfig, ShiftWorld, _train_and_calibration,
    benchmark, run_shift, _spawn_generation, _mutate_grid,
)
from ecsa.experimental.intervention_relations import (
    ObservedPair, StudyConfig, SyntheticInterventionWorld, _fit_hypotheses,
)
from ecsa.experimental.relation_discovery import Law


def small_config():
    return ShiftConfig(
        study=StudyConfig(
            bootstrap=28,calibration=8,train_steps=18,
            interventions=12,candidates=6,bins=19,heldout=12,
        ),
        generation_interval=6,steps_per_offspring=5,offspring=3,
        warmup_replay=5,
    )


def test_novel_environment_port_only_exposes_intervention():
    world=ShiftWorld(NewMechanism.TWO_BRANCH,seed=17,noise=0.0)
    assert not hasattr(world,'check_pair')
    assert not hasattr(world,'membership')
    samples=[world.intervene((0.3,-0.4)) for _ in range(35)]
    assert any(v < 0 for v in samples) and any(v > 0 for v in samples)
    assert len(world.calls)==35
    assert not any(word in inspect.getsource(run_shift) for word in ('_physical_y','heldout','_mechanism'))


def test_calibration_never_reenters_training():
    warmup=tuple(ObservedPair((float(i),0.),float(i)) for i in range(28))
    recent=tuple(ObservedPair((float(i+100),0.),float(i+100)) for i in range(12))
    training,calibration=_train_and_calibration(warmup,recent,small_config())
    assert calibration==tuple(recent[::4])
    assert set(training).isdisjoint(set(calibration))
    assert not any(item in training for item in warmup[-8:])
    assert len(training)==5+9


def test_compute_matched_structural_only_and_parameter_only_mutations():
    torch.set_num_threads(1)
    cfg=small_config()
    rows=benchmark(mechanism=NewMechanism.BILINEAR,seed=1,config=cfg)
    assert len(rows)==5
    by={r.arm:r for r in rows}
    assert set(by)=={arm.value for arm in ShiftArm if arm is not ShiftArm.STRUCTURE_ADAPTIVE_EIG}
    assert all(r.world_interventions==12 for r in rows)
    assert all(r.gradient_steps==3*cfg.steps_per_offspring for r in rows)
    assert all(r.proposals==3 for r in rows)
    assert len({r.initial_nll for r in rows})==1
    assert rows[0].heldout_outside_grid==rows[1].heldout_outside_grid
    random_trajectories=[[(t.x,t.y) for t in row.trials] for row in rows]
    assert all(t==random_trajectories[0] for t in random_trajectories)
    assert by['retrain'].topology_changes_proposed==0
    assert by['parameters'].topology_changes_proposed==0
    assert by['structure'].topology_changes_proposed==3
    assert by['mixed'].topology_changes_proposed==1
    assert by['structure'].topology_changes_accepted==1
    assert by['grid_geometry'].basis_changes_proposed==3
    assert by['grid_geometry'].basis_changes_accepted==1
    assert by['grid_geometry'].topology_changes_proposed==0
    assert by['parameters'].topology_changes_accepted==0
    assert by['grid_geometry'].gradient_steps==by['retrain'].gradient_steps
    assert all(math.isfinite(r.final_nll) for r in rows)
    assert all(len(r.checkpoints_nll)==3 for r in rows)
    assert all(len(r.generations)==1 for r in rows)


def test_mutated_posterior_versions_do_not_replay_training_samples():
    cfg=small_config()
    warmup_world=SyntheticInterventionWorld(Law.ADDITIVE,seed=20)
    rng=np.random.default_rng(21)
    warmup=tuple(ObservedPair(tuple(map(float,x)),warmup_world.intervene(tuple(x)))
                 for x in rng.uniform(-.9,.9,(cfg.study.bootstrap,2)))
    bank,_,grid=_fit_hypotheses(warmup,cfg.study,seed=2)
    pool=tuple(np.asarray([[0.,0.],[.2,-.5],[.8,.8],[.4,.1],[-.6,.2],[-.2,-.2]],dtype=np.float32)
               for _ in range(12))
    trace=run_shift(world=ShiftWorld(NewMechanism.THRESHOLD,seed=31),arm=ShiftArm.MIXED,
                    initial_bank=bank,warmup=warmup,grid=grid,candidates=pool,
                    config=cfg,rng_seed=28)
    assert len(trace.trials)==12
    assert trace.generations[0].epoch==1
    assert trace.generations[0].evidence_available==6
    assert len(set(trace.trials[5].posterior_ids)&set(trace.trials[6].prior_ids))==0
    assert trace.trials[6].prior!=trace.trials[5].posterior
    assert trace.trials[7].prior==trace.trials[6].posterior


def test_no_evaluation_oracle_or_law_inside_policy_signature():
    assert 'heldout' not in inspect.signature(run_shift).parameters
    assert 'mechanism' not in inspect.signature(run_shift).parameters
    assert 'heldout' not in inspect.signature(_spawn_generation).parameters
    assert 'mechanism' not in inspect.signature(_spawn_generation).parameters
    assert 'heldout' not in inspect.getsource(run_shift)


def test_invalid_budget_and_mechanism_rejected():
    with pytest.raises(ValueError):
        ShiftConfig(generation_interval=0)
    with pytest.raises(ValueError):
        ShiftConfig(warmup_replay=80)
    with pytest.raises(ValueError):
        ShiftWorld(NewMechanism.BILINEAR,seed=1).intervene((float('nan'), 0.0))


def test_grid_relocation_preserves_parameter_count_weights_and_parent():
    cfg=small_config()
    warmup_world=SyntheticInterventionWorld(Law.ADDITIVE,seed=20)
    rng=np.random.default_rng(21)
    warmup=tuple(ObservedPair(tuple(map(float,x)),warmup_world.intervene(tuple(x)))
                 for x in rng.uniform(-.9,.9,(cfg.study.bootstrap,2)))
    bank,_,_grid=_fit_hypotheses(warmup,cfg.study,seed=2)
    parent=bank[0]
    child=_mutate_grid(parent,rng=np.random.default_rng(14),sigma=.35,child_id='grid-child')
    assert parent.model is not child.model
    assert sum(p.numel()for p in parent.model.parameters())==sum(p.numel()for p in child.model.parameters())
    assert all(torch.equal(a,b) for a,b in zip(parent.model.parameters(),child.model.parameters()))
    assert not torch.equal(parent.model.relation.first.centers,child.model.relation.first.centers)
    assert not torch.equal(parent.model.relation.last.centers,child.model.relation.last.centers)
    assert torch.all(torch.diff(child.model.relation.first.centers)>=0)
    assert child.model.relation.first.centers.shape==parent.model.relation.first.centers.shape


def test_stationary_negative_control_matches_warmup_law():
    world=ShiftWorld(NewMechanism.UNCHANGED_CONTROL, seed=17,noise=0)
    x=(0.2,-0.4)
    result=world.intervene(x)
    expected=0.7*x[0]-0.45*x[1]+0.2*np.sin(2*x[0])
    assert result==pytest.approx(expected)


def test_same_seed_reproducible_on_hidden_new_mechanism():
    cfg=small_config()
    first=benchmark(mechanism=NewMechanism.TWO_BRANCH,seed=3,config=cfg,
                    arms=(ShiftArm.RETRAIN,ShiftArm.GRID_GEOMETRY))
    repeat=benchmark(mechanism=NewMechanism.TWO_BRANCH,seed=3,config=cfg,
                    arms=(ShiftArm.RETRAIN,ShiftArm.GRID_GEOMETRY))
    assert [(r.initial_nll,r.final_nll,r.parameter_steps) for r in first] == [
        (r.initial_nll,r.final_nll,r.parameter_steps) for r in repeat
    ]


def test_adaptive_eig_arm_keeps_world_budget_and_versioned_posteriors():
    cfg=small_config()
    (adaptive,)=benchmark(mechanism=NewMechanism.BILINEAR,seed=5,config=cfg,
                         arms=(ShiftArm.STRUCTURE_ADAPTIVE_EIG,))
    assert len(adaptive.trials)==cfg.study.interventions
    assert adaptive.eig_choices+adaptive.random_choices==cfg.study.interventions
    assert adaptive.gradient_steps==cfg.offspring*cfg.steps_per_offspring
    assert adaptive.topology_changes_proposed==cfg.offspring
    assert adaptive.generations[0].accepted_changed_topology
    assert all(math.isfinite(t.predictive_nll) for t in adaptive.trials)
