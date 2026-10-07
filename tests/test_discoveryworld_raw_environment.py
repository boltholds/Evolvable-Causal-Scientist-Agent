import inspect
import json

from ecsa.benchmarks.discoveryworld.raw_environment import (
    DiscoveryWorldRawEnvironment,
)
from ecsa.world_model.contracts import (
    GroundAction,
    RawActionSchema,
    freeze_raw_value,
)


def test_raw_adapter_exposes_action_symbol_and_arguments_without_role_map() -> None:
    env = DiscoveryWorldRawEnvironment.reactor_lab_normal(
        0,
        max_steps=8,
    )

    schemas = env.list_raw_actions()

    assert schemas
    assert all(isinstance(schema, RawActionSchema) for schema in schemas)
    assert any(schema.parameters for schema in schemas)
    assert all(not hasattr(schema, "role") for schema in schemas)
    assert all(
        not hasattr(parameter, "role")
        for schema in schemas
        for parameter in schema.parameters
    )


def test_raw_adapter_records_failed_action_as_outcome(monkeypatch) -> None:
    env = DiscoveryWorldRawEnvironment.reactor_lab_normal(
        0,
        max_steps=4,
    )
    before = env.steps
    monkeypatch.setattr(
        env._environment._api,
        "performAgentAction",
        lambda **_: {
            "success": False,
            "errors": ["opaque rejection"],
        },
    )

    outcome = env.execute_raw_action(
        GroundAction(
            schema_id="PICKUP",
            arguments=(freeze_raw_value(-999),),
        )
    )

    assert outcome.success is False
    assert outcome.payload.thaw()["errors"] == [
        "opaque rejection"
    ]
    assert env.steps == before + 1


def test_raw_adapter_policy_surface_contains_no_oracle_keys() -> None:
    env = DiscoveryWorldRawEnvironment.reactor_lab_normal(0)

    raw = env.observe_raw()
    encoded = json.dumps(raw.payload.thaw()).lower()

    assert "criticalquestions" not in encoded
    assert "criticalhypotheses" not in encoded
    assert "scorecard" not in encoded


def test_dialog_options_are_exposed_as_opaque_zero_arg_raw_actions(
    monkeypatch,
) -> None:
    env = DiscoveryWorldRawEnvironment.reactor_lab_normal(0)
    monkeypatch.setattr(
        env._environment,
        "observe",
        lambda: {
            "ui": {
                "dialog_box": {
                    "dialogOptions": {
                        "2": "opaque public option",
                    }
                },
                "inventoryObjects": [],
                "accessibleEnvironmentObjects": [],
                "nearbyObjects": {"objects": {}},
            }
        },
    )

    schemas = env.list_raw_actions()
    dialog = [
        schema
        for schema in schemas
        if schema.schema_id.startswith("__dw_wire_option__:")
    ]

    assert len(dialog) == 1
    assert dialog[0].parameters == ()
    assert dialog[0].public_metadata.thaw()[
        "label"
    ] == "opaque public option"


def test_raw_environment_module_has_no_semantic_role_map() -> None:
    import ecsa.benchmarks.discoveryworld.raw_environment as module

    source = inspect.getsource(module)
    assert "_ROLE_MAP" not in source
    assert "ActionRole" not in source



def test_raw_adapter_loads_unfamiliar_scenario_without_semantic_changes() -> None:
    env = DiscoveryWorldRawEnvironment.load(
        scenario="Archaeology Dating",
        difficulty="Normal",
        seed=0,
        max_steps=4,
    )

    raw = env.observe_raw()
    schemas = env.list_raw_actions()

    assert raw.step == 0
    assert schemas
    assert all(not hasattr(schema, "role") for schema in schemas)
