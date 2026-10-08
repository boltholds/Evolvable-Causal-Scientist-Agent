"""Gates for evolutionary KAN mutation and fair closed-loop intervention study."""
from __future__ import annotations

import inspect
import math
from dataclasses import asdict
import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from ecsa.contracts import PredictiveDistribution, TheoryPosterior
from ecsa.experimental.evolutionary_relations import (
    ChoiceReason,
    EvolutionArm,
    EvolutionConfig,
    MutationKind,
    _choose,
    _evolve,
    _structure,
    benchmark,
    mutate_kan,
    run_interventions,
)
from ecsa.experimental.intervention_relations import (
    ObservedPair,
    StudyConfig,
    SyntheticInterventionWorld,
    _fit_hypotheses,
)
from ecsa.experimental.relation_discovery import HeadKind, Law, RelationModel
from ecsa.experimental.intervention_relations import RelationHypothesis


def example_bank():
    torch.manual_seed(1)
    bank = tuple(
        RelationHypothesis(
            f"h-{i}", RelationModel(HeadKind.SPLINE_KAN),
            0.8, np.ones(21) / 21,
        )
        for i in range(3)
    )
    return bank


def small_config() -> EvolutionConfig:
    return EvolutionConfig(
        study=StudyConfig(
            bootstrap=28, calibration=8, train_steps=24,
            interventions=6, candidates=8, bins=21, heldout=20,
        ),
        mutation_interval=3,
        mutation_steps=8,
        random_floor=0.0,
    )


def test_random_structural_and_parameter_mutations_preserve_parent():
    bank = example_bank()
    parent = bank[0]
    original = {k: v.clone() for k, v in parent.model.state_dict().items()}
    parameter_child = mutate_kan(
        parent, kind=MutationKind.PARAMETER,
        rng=np.random.default_rng(3), sigma=0.1, child_id="child-param",
    )
    structure_child = mutate_kan(
        parent, kind=MutationKind.STRUCTURE,
        rng=np.random.default_rng(3), sigma=0.1, child_id="child-structure",
    )
    assert parent.model is not parameter_child.model
    assert _structure(structure_child.model) != _structure(parent.model)
    assert _structure(parameter_child.model) == _structure(parent.model)
    assert all(torch.equal(v, parent.model.state_dict()[k]) for k, v in original.items())
    assert any(
        not torch.equal(v, parameter_child.model.state_dict()[k])
        for k, v in original.items() if k.startswith("relation.")
    )
    for key, tensor in original.items():
        if key.startswith("encoder."):
            assert torch.equal(tensor, structure_child.model.state_dict()[key])


def test_versioned_population_resets_posterior_without_replaying_training_outcomes():
    config = small_config()
    torch.set_num_threads(1)
    world = SyntheticInterventionWorld(Law.ADDITIVE, seed=11)
    rng = np.random.default_rng(12)
    warmup = tuple(
        ObservedPair(tuple(float(v) for v in x), world.intervene(tuple(x)))
        for x in rng.uniform(-0.9, 0.9, (config.study.bootstrap, 2))
    )
    bank, _, grid = _fit_hypotheses(warmup, config.study, seed=0)
    pools = tuple(np.asarray([[.1, .2], [-.1, .2], [.5, .4]], dtype=np.float32) for _ in range(6))
    trial_world = SyntheticInterventionWorld(Law.ADDITIVE, seed=21)
    trace = run_interventions(
        world=trial_world, arm=EvolutionArm.EVOLVING_RANDOM,
        initial_bank=bank, grid=grid, warmup=warmup,
        candidate_pools=pools, config=config, rng_seed=8,
    )
    assert len(trial_world.calls) == 6
    assert len(trace.events) == 6
    assert sum(e.accepted for e in trace.events) == 2
    assert {e.kind for e in trace.events} == {MutationKind.STRUCTURE, MutationKind.PARAMETER}
    assert trace.mutation_steps == len(trace.events) * config.mutation_steps
    assert trace.trials[2].epoch == 0
    assert trace.trials[3].epoch == 1
    # A new version may use old observations for fitting but old posterior
    # weights are never carried across a mutation epoch.
    old_ids = {ident for ident, _ in trace.trials[2].posterior_updated}
    new_ids = {ident for ident, _ in trace.trials[3].posterior_prior}
    assert old_ids.isdisjoint(new_ids)
    for event in trace.events:
        assert event.child_id not in old_ids or event.epoch == 0
    assert abs(sum(p for _, p in trace.final.posterior.probabilities) - 1.0) < 1e-9


def test_adaptive_gate_requires_reliable_calibration():
    config = small_config()
    posterior = TheoryPosterior((("a", 0.5), ("b", 0.5)))
    distributions = (
        tuple(PredictiveDistribution(h, "uninformative", (((0,), p), ((1,), 1-p)))
              for h, p in (("a", .5), ("b", .5))),
        tuple(PredictiveDistribution(h, "informative", (((0,), p), ((1,), 1-p)))
              for h, p in (("a", .95), ("b", .05))),
    )
    bank = example_bank()[:2]
    index, gain, reason, passed = _choose(
        bank, posterior, distributions,
        arm=EvolutionArm.EVOLVING_ADAPTIVE_EIG, rng=np.random.default_rng(0),
        calibration_nll=4.0, calibration_reference=3.0, config=config,
    )
    assert reason is ChoiceReason.LOW_RELIABILITY
    assert not passed
    assert gain > 0
    index, gain, reason, passed = _choose(
        bank, posterior, distributions,
        arm=EvolutionArm.EVOLVING_ADAPTIVE_EIG, rng=np.random.default_rng(0),
        calibration_nll=1.0, calibration_reference=3.0, config=config,
    )
    assert index == 1 and passed and reason is ChoiceReason.EIG


def test_random_floor_is_enforced_after_calibration_gate():
    posterior = TheoryPosterior((("a", .5), ("b", .5)))
    predictions = (
        tuple(PredictiveDistribution(h, "c0", (((0,), p), ((1,), 1-p)))
              for h, p in (("a", .5), ("b", .5))),
        tuple(PredictiveDistribution(h, "c1", (((0,), p), ((1,), 1-p)))
              for h, p in (("a", .99), ("b", .01))),
    )
    config = EvolutionConfig(
        study=small_config().study,
        random_floor=.99,
    )
    _, _, reason, passed = _choose(
        example_bank(), posterior, predictions,
        arm=EvolutionArm.EVOLVING_ADAPTIVE_EIG, rng=np.random.default_rng(1),
        calibration_nll=1., calibration_reference=3., config=config,
    )
    assert passed and reason is ChoiceReason.RANDOM_FLOOR


def test_intervention_policy_signature_prevents_heldout_leakage():
    assert "heldout" not in inspect.signature(run_interventions).parameters
    assert "heldout" not in inspect.signature(_evolve).parameters
    assert "oracle_compatibility" not in inspect.getsource(run_interventions)
    assert "_physical_y" not in inspect.getsource(run_interventions)


def test_four_arm_study_has_equal_world_budget_and_new_generation_ids():
    cfg = small_config()
    rows = benchmark(Law.ADDITIVE, seed=2, config=cfg)
    assert {r.arm for r in rows} == set(EvolutionArm) - {EvolutionArm.RETRAIN_ONLY_RANDOM}
    assert len(rows) == 4
    assert all(r.interventions == 6 for r in rows)
    assert all(r.warmup_observations == cfg.study.bootstrap for r in rows)
    assert all(r.heldout_observations == cfg.study.heldout for r in rows)
    assert len({r.initial_nll for r in rows}) == 1
    assert len({r.heldout_unconditional_nll for r in rows}) == 1
    assert len({r.heldout_knn_nll for r in rows}) == 1
    fixed, fixed_eig, evolving, adaptive = rows
    assert fixed.mutation_proposals == fixed_eig.mutation_proposals == 0
    assert fixed.mutation_training_steps == fixed_eig.mutation_training_steps == 0
    assert fixed.eig_choices == evolving.eig_choices == 0
    assert evolving.mutation_proposals == adaptive.mutation_proposals == 6
    assert evolving.mutation_training_steps == adaptive.mutation_training_steps
    assert evolving.mutation_accepted == adaptive.mutation_accepted == 2
    assert all(len(r.trials) == 6 for r in rows)
    assert all(r.random_choices + r.eig_choices == 6 for r in rows)
    assert all(len(r.checkpoints_nll) == 2 for r in rows)
    assert all(math.isfinite(r.final_nll) for r in rows)
    # JSON artifact must preserve each independent mutation and outcome.
    assert len(json.loads(json.dumps(asdict(adaptive)))["trials"]) == 6


def test_malformed_mutation_and_budget_rejected():
    with pytest.raises(ValueError):
        EvolutionConfig(population_size=1)
    with pytest.raises(ValueError):
        EvolutionConfig(mutation_interval=0)
    with pytest.raises(ValueError):
        EvolutionConfig(random_floor=1.)
    with pytest.raises(TypeError):
        mutate_kan(example_bank()[0], kind="structural", rng=np.random.default_rng(4),
                   sigma=.05, child_id="bad")


def test_retraining_only_control_spends_equal_fitting_budget_without_mutations():
    cfg = small_config()
    five = benchmark(Law.ADDITIVE, seed=6, config=cfg, include_retrain_control=True)
    assert len(five) == 5
    control = five[-1]
    evolved = five[2]
    assert control.arm is EvolutionArm.RETRAIN_ONLY_RANDOM
    assert control.interventions == evolved.interventions == cfg.study.interventions
    assert control.mutation_training_steps == evolved.mutation_training_steps
    assert control.mutation_proposals == control.mutation_accepted == 0
    assert control.mutation_events and all(
        event.kind is MutationKind.CLONE for event in control.mutation_events
    )
    assert control.initial_nll == evolved.initial_nll
