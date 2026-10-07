import json
from pathlib import Path

import inspect

from ecsa.benchmarks.discoveryworld.contracts import (
    DiscoveryWorldEpisodeConfig,
    GenericScalarEvidence,
    ScientificContext,
)
import ecsa.autonomy as autonomy
from ecsa.adapters.mlmd import MLMDMechanismRepository
from ecsa.benchmarks.discoveryworld.policies import autonomous_scientist
from ecsa.benchmarks.discoveryworld.policies.autonomous_scientist import (
    AutonomousScientistPolicy,
    _transfer_views,
)
from ecsa.benchmarks.discoveryworld.arena import (
    _context_for_policy,
    run_episode,
)
from ecsa.mechanisms import (
    EpistemicStatus,
    MechanismKind,
    MechanismRecord,
    MechanismScope,
    TransferStatus,
)


def _observation(entities, *, task="Investigate sample station") -> dict:
    return {
        "ui": {
            "taskProgress": [{"description": task}],
            "dialog_box": {"dialogIn": "", "dialogOptions": {}},
            "inventoryObjects": [],
            "accessibleEnvironmentObjects": entities,
            "nearbyObjects": {"objects": {}},
            "lastActionMessage": "",
        }
    }


def _evidence(
    evidence_id,
    step,
    entity_ids,
    feature_key,
    value,
    text,
    action_name="USE",
):
    return GenericScalarEvidence(
        evidence_id=evidence_id,
        step=step,
        action_name=action_name,
        entity_ids=entity_ids,
        feature_key=feature_key,
        value=value,
        raw_text=text,
    )


def test_autonomous_policy_has_no_scenario_spoilers() -> None:
    source = (
        inspect.getsource(autonomous_scientist)
        + inspect.getsource(autonomy)
    ).lower()
    forbidden = (
        "reactor",
        "crystal",
        "density",
        "densitometer",
        "thermometer",
        "spectrometer",
        "radiation",
        "frequency",
        "linear",
    )
    assert not any(token in source for token in forbidden)
    core_source = inspect.getsource(autonomy).lower()
    assert "benchmarks" not in core_source
    assert "discoveryworld" not in core_source
    assert "MechanismRecord" not in core_source


def test_autonomous_policy_selects_location_from_public_goal_words() -> None:
    policy = AutonomousScientistPolicy({})
    decision = policy.decide(
        _observation([], task="Investigate the alpha station"),
        {
            "TELEPORT_TO_LOCATION": {
                "args": ["arg1"],
                "desc": "teleport",
            }
        },
        {
            "plaza": [0, 0],
            "alpha station": [1, 1],
        },
        ScientificContext("ctx", (), (), ()),
    )

    assert decision.action == {
        "action": "TELEPORT_TO_LOCATION",
        "arg1": "alpha station",
    }


def test_autonomous_policy_discovers_numeric_program_from_neutral_evidence() -> None:
    policy = AutonomousScientistPolicy(
        {"max_polynomial_degree": 2}
    )
    entities = [
        {"uuid": 101, "name": "controller (activated)", "description": "controller"},
        {"uuid": 102, "name": "controller (activated)", "description": "controller"},
        {"uuid": 103, "name": "controller (idle)", "description": "controller"},
        {"uuid": 201, "name": "sample 1", "description": "sample 1"},
        {"uuid": 202, "name": "sample 2", "description": "sample 2"},
        {"uuid": 203, "name": "sample 3", "description": "sample 3"},
    ]
    evidence = (
        _evidence("i1", 1, (301, 201), "input", 1.0, "probe sample 1"),
        _evidence("i2", 2, (301, 202), "input", 2.0, "probe sample 2"),
        _evidence("i3", 3, (301, 203), "input", 3.0, "probe sample 3"),
        _evidence("o1", 4, (101,), "output", 12.0, "controller #1 current 12", "TALK"),
        _evidence("o2", 5, (102,), "output", 22.0, "controller #2 current 22", "TALK"),
        _evidence("a1", 4, (101,), "identity", 1.0, "controller identity 1", "TALK"),
        _evidence("a2", 5, (102,), "identity", 2.0, "controller identity 2", "TALK"),
        _evidence("a3", 6, (103,), "identity", 3.0, "controller identity 3", "TALK"),
    )
    context = ScientificContext(
        "ctx",
        (),
        (),
        (),
        generic_evidence=evidence,
    )

    decision = policy.decide(
        _observation(entities),
        {
            "ACTIVATE": {"args": ["arg1"]},
            "PICKUP": {"args": ["arg1"]},
            "PUT": {"args": ["arg1", "arg2"]},
            "TALK": {"args": ["arg1"]},
            "TELEPORT_TO_OBJECT": {"args": ["arg1"]},
        },
        {},
        context,
    )

    assert len(decision.generic_hypotheses) == 1
    hypothesis = decision.generic_hypotheses[0]
    assert hypothesis.polynomial_degree == 1
    assert hypothesis.predictions[0].control_entity_id == 103
    assert hypothesis.predictions[0].subject_entity_id == 203
    assert hypothesis.predictions[0].predicted_value == 32.0


def test_transfer_candidate_is_structural_prior_not_feature_answer() -> None:
    source = MechanismRecord(
        mechanism_id="source",
        version=1,
        kind=MechanismKind.SYMBOLIC_RULE,
        epistemic_status=EpistemicStatus.ADMITTED,
        representation_artifact="benchmark-private-artifact",
        scope=MechanismScope(
            context_ids=("old",),
            regime_ids=("normal",),
            domain_ids=("world",),
            task_ids=("task",),
            required_assumptions=(
                "public-observation-only",
                "generic-scalar-discovery",
            ),
        ),
        transfer_status=TransferStatus.CONTEXT_SPECIALIZED,
        parameters=(
            ("transfer_key", "scalar-interaction-to-control"),
            ("generic_degree", 1),
            ("generic_feature_key", "old-feature"),
            ("benchmark_private_parameter", 999.0),
        ),
    )
    context = ScientificContext(
        "new",
        (),
        (source,),
        (),
        generic_evidence=(),
    )

    [view] = _transfer_views(context)
    scientist = autonomy.AutonomousScientist()

    assert view.ref == source.ref
    assert view.transfer_key == "scalar-interaction-to-control"
    assert view.preferred_degree == 1
    assert not hasattr(view, "parameters")
    assert scientist._transfer_source((view,)) == view


def test_autonomous_policy_does_not_assert_function_family() -> None:
    policy = AutonomousScientistPolicy({})
    assert policy.scientific_assumptions == (
        "public-observation-only",
        "generic-scalar-discovery",
    )
    assert all(
        "linear" not in assumption
        for assumption in policy.scientific_assumptions
    )



def test_arena_uses_generic_context_for_autonomous_policy() -> None:
    policy = AutonomousScientistPolicy({})
    context = _context_for_policy(seed=0, policy=policy)

    assert context.assumptions == (
        "public-observation-only",
        "generic-scalar-discovery",
    )
    assert "linear-family" not in context.assumptions


def test_real_autonomous_policy_executes_public_loop(
    tmp_path: Path,
) -> None:
    policy = AutonomousScientistPolicy(
        {
            "max_polynomial_degree": 2,
            "max_pair_trials": 16,
        }
    )
    repository = MLMDMechanismRepository.sqlite(
        tmp_path / "mechanisms.sqlite"
    )

    result = run_episode(
        config=DiscoveryWorldEpisodeConfig(
            scenario="Reactor Lab",
            difficulty="Normal",
            seed=0,
            max_steps=5,
        ),
        policy=policy,
        repository=repository,
        output_dir=tmp_path / "run",
    )

    assert 1 <= result.evaluation.steps <= 5
    assert (tmp_path / "run" / "actions.jsonl").exists()
    assert (tmp_path / "run" / "observations.jsonl").exists()



def test_real_autonomous_seed0_reaches_generic_hypothesis(
    tmp_path: Path,
) -> None:
    policy = AutonomousScientistPolicy(
        {
            "max_polynomial_degree": 2,
            "max_pair_trials": 96,
            "control_tolerance": 1.5,
        }
    )
    repository = MLMDMechanismRepository.sqlite(
        tmp_path / "mechanisms.sqlite"
    )
    run_dir = tmp_path / "autonomous-discovery"

    result = run_episode(
        config=DiscoveryWorldEpisodeConfig(
            scenario="Reactor Lab",
            difficulty="Normal",
            seed=0,
            max_steps=250,
        ),
        policy=policy,
        repository=repository,
        output_dir=run_dir,
    )

    events = [
        json.loads(line)
        for line in (run_dir / "scientific_events.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    actions = [
        json.loads(line)
        for line in (run_dir / "actions.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]

    generic_evidence = [
        event
        for event in events
        if event.get("kind") == "generic_scalar_evidence"
    ]
    generic_hypotheses = [
        event
        for event in events
        if event.get("kind") == "generic_hypothesis_frozen"
    ]

    diagnostic = {
        "steps": result.evaluation.steps,
        "generic_evidence_count": len(generic_evidence),
        "event_kinds": sorted(
            {
                event.get("kind")
                for event in events
                if isinstance(event.get("kind"), str)
            }
        ),
        "last_actions": actions[-20:],
    }

    assert generic_evidence, diagnostic
    assert any(
        len(event["evidence"]["entity_ids"]) >= 2
        for event in generic_evidence
    ), diagnostic
    assert generic_hypotheses, diagnostic



def test_empty_terminal_dialog_is_not_actionable() -> None:
    observation = _observation([])
    observation["ui"]["dialog_box"] = {
        "dialogIn": "Done.",
        "dialogOptions": {},
        "is_in_dialog": True,
    }

    decision = AutonomousScientistPolicy({}).decide(
        observation,
        {
            "DISCOVERY_FEED_GET_UPDATES": {
                "args": [],
                "desc": "observe",
            }
        },
        {},
        ScientificContext("ctx", (), (), ()),
    )

    assert decision.action == {
        "action": "DISCOVERY_FEED_GET_UPDATES"
    }
