import json

from ecsa.benchmarks.discoveryworld.contracts import PolicyDecision
from ecsa.benchmarks.discoveryworld.environment import (
    DiscoveryWorldEnvironmentAdapter,
)


def test_real_reactor_lab_all_official_seeds_load() -> None:
    for seed in range(5):
        env = DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(seed)
        observation = env.observe()
        assert observation["ui"]["taskProgress"][0]["description"]
        assert env.steps == 0
        assert env.done is False


def test_action_call_ticks_exactly_once() -> None:
    env = DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(0)
    before = env.steps
    locations = env.teleport_locations()
    destination = next(iter(locations))
    env.act(
        {
            "action": "TELEPORT_TO_LOCATION",
            "arg1": destination,
        }
    )
    assert env.steps == before + 1


def test_failed_action_still_ticks_once(monkeypatch) -> None:
    env = DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(0)
    before = env.steps
    monkeypatch.setattr(
        env._api,
        "performAgentAction",
        lambda **_: {"success": False, "errors": ["invalid"]},
    )
    result = env.act({"action": "PICKUP"})
    assert result.success is False
    assert env.steps == before + 1


def test_real_reactor_observation_has_no_oracle_keys() -> None:
    observation = DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(0).observe()
    encoded = json.dumps(observation).lower()
    assert "criticalquestions" not in encoded
    assert "criticalhypotheses" not in encoded
    assert "scorecard" not in encoded


def test_policy_surface_never_reads_scorecard(monkeypatch) -> None:
    env = DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(0)
    monkeypatch.setattr(
        env._api,
        "getTaskScorecard",
        lambda: (_ for _ in ()).throw(AssertionError("oracle read")),
    )
    env.observe()
    env.available_actions()
    env.teleport_locations()



def test_policy_decision_accepts_discoveryworld_dialog_packet() -> None:
    decision = PolicyDecision(
        action={"chosen_dialog_option_int": 2},
    )
    assert decision.action == {"chosen_dialog_option_int": 2}


def test_dialog_action_packet_ticks_exactly_once(monkeypatch) -> None:
    env = DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(0)
    before = env.steps
    received = []

    def perform(**kwargs):
        received.append(kwargs["actionJSON"])
        return {"success": True, "errors": []}

    monkeypatch.setattr(env._api, "performAgentAction", perform)

    result = env.act({"chosen_dialog_option_int": 1})

    assert result.success is True
    assert received == [{"chosen_dialog_option_int": 1}]
    assert env.steps == before + 1



def test_upstream_string_errors_are_normalized(monkeypatch) -> None:
    env = DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(0)

    monkeypatch.setattr(
        env._api,
        "performAgentAction",
        lambda **_: {"success": False, "errors": "dialog parse error"},
    )

    result = env.act({"action": "PICKUP"})

    assert result.success is False
    assert result.errors == ("dialog parse error",)


def test_upstream_empty_string_errors_are_normalized(monkeypatch) -> None:
    env = DiscoveryWorldEnvironmentAdapter.reactor_lab_normal(0)

    monkeypatch.setattr(
        env._api,
        "performAgentAction",
        lambda **_: {"success": True, "errors": ""},
    )

    result = env.act({"action": "PICKUP"})

    assert result.success is True
    assert result.errors == ()
