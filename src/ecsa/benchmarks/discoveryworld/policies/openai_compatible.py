from __future__ import annotations

import json
import os
from typing import Any

from ecsa.mechanisms import MechanismVersionRef

from ..contracts import (
    JSONValue,
    MeasurementKind,
    PolicyDecision,
    ReactorFrequencyPrediction,
    ReactorMechanismHypothesis,
    ScientificContext,
    validate_action_packet,
)


_SYSTEM_PROMPT = """You are the action policy for an agent in DiscoveryWorld Reactor Lab.

You receive only agent-visible observations, public action metadata, teleport locations,
and ECSA scientific memory. Never assume access to evaluator scorecards, hidden object
fields, criticalQuestions, criticalHypotheses, scoringInfo, or correct hidden reactor
frequencies.

Your job is to solve the task while using as few environment steps and scientific
instrument USE actions as practical.

Return exactly one JSON object with this shape:
{
  "action": {"action": "ACTION_NAME", "...": "JSON scalar args"}
            OR {"chosen_dialog_option_int": 0},
  "reasoning": "short explanation or null",
  "memory": "compact persistent state for your next turn or null",
  "hypotheses": [
    {
      "hypothesis_id": "stable unique id",
      "measurement_kind": "density|temperature|quantum_size|radiation|spectrum",
      "slope": 0.0,
      "offset": 0.0,
      "source_evidence_ids": ["public evidence ids"],
      "predictions": [
        {
          "target_crystal_uuid": 0,
          "target_reactor_uuid": 0,
          "predicted_frequency": 0.0
        }
      ],
      "source_mechanism": {"mechanism_id": "...", "version": 1} OR null
    }
  ],
  "validation_hypothesis_ids": ["hypothesis ids deliberately tested by this action"]
}

Rules:
- Pick exactly one action that is actually available. Never invent UUIDs or action names.
- If a dialog is open, use chosen_dialog_option_int only for a visible dialog option.
- Teleporting to the reactor lab is preferable to wasting navigation steps when a
  suitable public teleport is available.
- Learn from ui.lastActionMessage, dialog text, visible object names/UUIDs, and the
  structured ECSA measurements supplied below.
- Transfer candidates came from earlier seeds. They are suggestions, not facts in the
  current seed. Use them to prioritize experiments, but require target-context
  prospective validation before treating them as valid.
- Emit a structured hypothesis only when public evidence supports a concrete linear
  measurement-to-frequency proposal and you can make a prospective reactor prediction.
  Do not guess slope/offset merely to fill the schema.
- source_evidence_ids must refer to evidence ids supplied in scientific_context, either
  current measurements or evidence already carried by a transfer candidate.
- source_mechanism must be null unless this exact hypothesis is derived from one of the
  supplied transfer candidates; if used, copy its mechanism_id and version exactly.
- validation_hypothesis_ids should contain only hypotheses whose prediction the current
  action is deliberately testing. Merely retrieving or considering a transfer
  candidate is not validation.
- Keep memory compact but sufficient to avoid repeating measurements and actions.
- Do not put markdown around the JSON.
"""


def _compact_observation(
    observation: dict[str, JSONValue],
) -> dict[str, JSONValue]:
    ui = observation.get("ui")
    if not isinstance(ui, dict):
        return observation

    keep_ui = (
        "taskProgress",
        "lastActionMessage",
        "dialog_box",
        "inventoryObjects",
        "accessibleEnvironmentObjects",
        "nearbyObjects",
    )
    compact_ui: dict[str, JSONValue] = {
        key: ui[key]
        for key in keep_ui
        if key in ui
    }
    compact: dict[str, JSONValue] = {"ui": compact_ui}
    for key in (
        "agent",
        "agentLocation",
        "location",
        "worldStep",
        "step",
    ):
        if key in observation:
            compact[key] = observation[key]
    return compact


def _measurement_wire(measurement) -> dict[str, JSONValue]:
    return {
        "evidence_id": measurement.evidence_id,
        "step": measurement.step,
        "context_id": measurement.context_id,
        "crystal_uuid": measurement.crystal_uuid,
        "crystal_name": measurement.crystal_name,
        "instrument_uuid": measurement.instrument_uuid,
        "instrument_name": measurement.instrument_name,
        "kind": measurement.kind.value,
        "values": list(measurement.values),
        "raw_message": measurement.raw_message,
    }


def _scientific_context_wire(
    context: ScientificContext,
) -> dict[str, JSONValue]:
    return {
        "context_id": context.context_id,
        "measurements": [
            _measurement_wire(value)
            for value in context.measurements
        ],
        "transfer_candidates": [
            value.to_wire()
            for value in context.transfer_candidates
        ],
        "admitted_mechanisms": [
            value.to_wire()
            for value in context.admitted_mechanisms
        ],
    }


def _require_string(
    value: object,
    *,
    name: str,
    allow_none: bool = False,
) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _require_float(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    return float(value)


class OpenAICompatiblePolicy:
    """Provider-neutral chat-completions policy for DiscoveryWorld.

    The endpoint is configured with base_url and an API-key environment variable,
    so OpenAI-compatible providers such as OpenRouter can be used without changing
    benchmark code.
    """

    def __init__(self, config: dict[str, JSONValue]) -> None:
        if not isinstance(config, dict):
            raise ValueError("policy config must be a JSON object")

        model = config.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("policy config requires nonempty model")
        self.model = model

        base_url = config.get("base_url")
        if base_url is not None and (
            not isinstance(base_url, str) or not base_url
        ):
            raise ValueError("base_url must be null or a nonempty string")
        self.base_url = base_url

        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        if not isinstance(api_key_env, str) or not api_key_env:
            raise ValueError("api_key_env must be a nonempty string")
        self.api_key_env = api_key_env

        temperature = config.get("temperature", 0.0)
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not 0.0 <= float(temperature) <= 2.0
        ):
            raise ValueError("temperature must be between 0 and 2")
        self.temperature = float(temperature)

        max_tokens = config.get("max_tokens", 1800)
        if type(max_tokens) is not int or max_tokens < 128:
            raise ValueError("max_tokens must be an integer >= 128")
        self.max_tokens = max_tokens

        timeout = config.get("timeout_seconds", 120.0)
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or float(timeout) <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        self.timeout = float(timeout)

        max_attempts = config.get("max_attempts", 2)
        if type(max_attempts) is not int or max_attempts not in (1, 2, 3):
            raise ValueError("max_attempts must be 1, 2, or 3")
        self.max_attempts = max_attempts

        json_mode = config.get("json_mode", True)
        if type(json_mode) is not bool:
            raise ValueError("json_mode must be bool")
        self.json_mode = json_mode

        extra = config.get("extra_instructions")
        if extra is not None and not isinstance(extra, str):
            raise ValueError("extra_instructions must be null or string")
        self.extra_instructions = extra or ""

        self._memory: str | None = None
        self._step = 0
        self._known_hypothesis_ids: set[str] = set()
        self._client: Any | None = None

    def decide(
        self,
        observation: dict[str, JSONValue],
        available_actions: dict[str, JSONValue],
        teleport_locations: dict[str, JSONValue],
        scientific_context: ScientificContext,
    ) -> PolicyDecision:
        payload: dict[str, Any] = {
            "decision_step": self._step,
            "persistent_memory": self._memory,
            "observation": _compact_observation(observation),
            "available_actions": available_actions,
            "teleport_locations": teleport_locations,
            "scientific_context": _scientific_context_wire(
                scientific_context
            ),
        }
        if self.extra_instructions:
            payload["extra_instructions"] = self.extra_instructions

        last_error: str | None = None
        for _ in range(self.max_attempts):
            request_payload = dict(payload)
            if last_error is not None:
                request_payload["previous_response_error"] = last_error
            wire = self._complete_json(request_payload)
            try:
                decision = self._decision_from_wire(
                    wire,
                    observation=observation,
                    available_actions=available_actions,
                    scientific_context=scientific_context,
                )
            except (TypeError, ValueError) as exc:
                last_error = str(exc)
                continue

            if decision.memory is not None:
                self._memory = decision.memory
            self._known_hypothesis_ids.update(
                hypothesis.hypothesis_id
                for hypothesis in decision.hypotheses
            )
            self._step += 1
            return decision

        raise RuntimeError(
            "model failed to produce a valid DiscoveryWorld decision"
            + (f": {last_error}" if last_error else "")
        )

    def _complete_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        completion = client.chat.completions.create(**kwargs)
        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise RuntimeError(
                "OpenAI-compatible endpoint returned no assistant message"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "OpenAI-compatible endpoint returned empty content"
            )
        try:
            wire = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("model response is not valid JSON") from exc
        if not isinstance(wire, dict):
            raise ValueError("model response must be a JSON object")
        return wire

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"missing API key environment variable: {self.api_key_env}"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI-compatible policy requires the optional 'llm' extra"
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.timeout,
        }
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def _decision_from_wire(
        self,
        wire: dict[str, Any],
        *,
        observation: dict[str, JSONValue],
        available_actions: dict[str, JSONValue],
        scientific_context: ScientificContext,
    ) -> PolicyDecision:
        action = wire.get("action")
        if not isinstance(action, dict):
            raise ValueError("response action must be an object")
        validate_action_packet(action)

        action_name = action.get("action")
        if isinstance(action_name, str) and action_name not in available_actions:
            raise ValueError(
                f"model selected unavailable action: {action_name}"
            )

        reasoning = wire.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("reasoning must be null or string")
        memory = wire.get("memory")
        if memory is not None and not isinstance(memory, str):
            raise ValueError("memory must be null or string")
        if memory is None:
            memory = self._memory

        current_evidence = {
            value.evidence_id
            for value in scientific_context.measurements
        }
        transferred_evidence = {
            evidence_id
            for mechanism in scientific_context.transfer_candidates
            for evidence_id in mechanism.supporting_evidence
        }
        allowed_evidence = current_evidence | transferred_evidence
        candidate_refs = {
            mechanism.ref
            for mechanism in scientific_context.transfer_candidates
        }

        visible_uuids = self._visible_uuids(observation)
        hypotheses_raw = wire.get("hypotheses", [])
        if not isinstance(hypotheses_raw, list):
            raise ValueError("hypotheses must be a list")

        hypotheses: list[ReactorMechanismHypothesis] = []
        response_ids: set[str] = set()
        for item in hypotheses_raw:
            if not isinstance(item, dict):
                raise ValueError("each hypothesis must be an object")
            hypothesis_id = _require_string(
                item.get("hypothesis_id"),
                name="hypothesis_id",
            )
            assert hypothesis_id is not None
            if (
                hypothesis_id in response_ids
                or hypothesis_id in self._known_hypothesis_ids
            ):
                raise ValueError(
                    f"hypothesis_id must be new and unique: {hypothesis_id}"
                )
            response_ids.add(hypothesis_id)

            kind_raw = _require_string(
                item.get("measurement_kind"),
                name="measurement_kind",
            )
            assert kind_raw is not None
            try:
                measurement_kind = MeasurementKind(kind_raw)
            except ValueError as exc:
                raise ValueError(
                    f"unsupported measurement_kind: {kind_raw}"
                ) from exc

            evidence_raw = item.get("source_evidence_ids", [])
            if not isinstance(evidence_raw, list) or not all(
                isinstance(value, str) and value
                for value in evidence_raw
            ):
                raise ValueError(
                    "source_evidence_ids must be a string list"
                )
            source_evidence_ids = tuple(evidence_raw)
            unknown_evidence = set(source_evidence_ids) - allowed_evidence
            if unknown_evidence:
                raise ValueError(
                    "hypothesis references unknown public evidence: "
                    + ", ".join(sorted(unknown_evidence))
                )

            source_mechanism = self._source_ref(
                item.get("source_mechanism")
            )
            if (
                source_mechanism is not None
                and source_mechanism not in candidate_refs
            ):
                raise ValueError(
                    "source_mechanism is not a current transfer candidate"
                )

            predictions_raw = item.get("predictions")
            if not isinstance(predictions_raw, list) or not predictions_raw:
                raise ValueError(
                    "hypothesis requires at least one prediction"
                )
            predictions: list[ReactorFrequencyPrediction] = []
            for prediction_raw in predictions_raw:
                if not isinstance(prediction_raw, dict):
                    raise ValueError("prediction must be an object")
                crystal_uuid = _require_int(
                    prediction_raw.get("target_crystal_uuid"),
                    name="target_crystal_uuid",
                )
                reactor_uuid = _require_int(
                    prediction_raw.get("target_reactor_uuid"),
                    name="target_reactor_uuid",
                )
                if visible_uuids and (
                    crystal_uuid not in visible_uuids
                    or reactor_uuid not in visible_uuids
                ):
                    raise ValueError(
                        "prediction UUIDs must be publicly visible"
                    )
                predictions.append(
                    ReactorFrequencyPrediction(
                        target_crystal_uuid=crystal_uuid,
                        target_reactor_uuid=reactor_uuid,
                        predicted_frequency=_require_float(
                            prediction_raw.get("predicted_frequency"),
                            name="predicted_frequency",
                        ),
                        frozen_step=self._step,
                    )
                )

            hypotheses.append(
                ReactorMechanismHypothesis(
                    hypothesis_id=hypothesis_id,
                    measurement_kind=measurement_kind,
                    slope=_require_float(
                        item.get("slope"),
                        name="slope",
                    ),
                    offset=_require_float(
                        item.get("offset"),
                        name="offset",
                    ),
                    source_evidence_ids=source_evidence_ids,
                    predictions=tuple(predictions),
                    source_mechanism=source_mechanism,
                )
            )

        validation_raw = wire.get("validation_hypothesis_ids", [])
        if not isinstance(validation_raw, list) or not all(
            isinstance(value, str) and value
            for value in validation_raw
        ):
            raise ValueError(
                "validation_hypothesis_ids must be a string list"
            )
        validation_ids = tuple(validation_raw)
        if len(set(validation_ids)) != len(validation_ids):
            raise ValueError(
                "validation_hypothesis_ids must be unique"
            )
        known_after_response = (
            self._known_hypothesis_ids | response_ids
        )
        unknown_validation = (
            set(validation_ids) - known_after_response
        )
        if unknown_validation:
            raise ValueError(
                "validation references unknown hypothesis: "
                + ", ".join(sorted(unknown_validation))
            )

        return PolicyDecision(
            action=dict(action),
            reasoning=reasoning,
            memory=memory,
            hypotheses=tuple(hypotheses),
            validation_hypothesis_ids=validation_ids,
        )

    @staticmethod
    def _source_ref(value: object) -> MechanismVersionRef | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(
                "source_mechanism must be null or an object"
            )
        mechanism_id = _require_string(
            value.get("mechanism_id"),
            name="source_mechanism.mechanism_id",
        )
        assert mechanism_id is not None
        return MechanismVersionRef(
            mechanism_id=mechanism_id,
            version=_require_int(
                value.get("version"),
                name="source_mechanism.version",
            ),
        )

    @staticmethod
    def _visible_uuids(
        observation: dict[str, JSONValue],
    ) -> set[int]:
        ui = observation.get("ui")
        if not isinstance(ui, dict):
            return set()
        found: set[int] = set()

        def add_many(value: object) -> None:
            if not isinstance(value, list):
                return
            for item in value:
                if isinstance(item, dict):
                    uuid = item.get("uuid")
                    if type(uuid) is int:
                        found.add(uuid)

        add_many(ui.get("inventoryObjects"))
        add_many(ui.get("accessibleEnvironmentObjects"))
        nearby = ui.get("nearbyObjects")
        if isinstance(nearby, dict):
            groups = nearby.get("objects")
            if isinstance(groups, dict):
                for values in groups.values():
                    add_many(values)
        return found


def create_policy(
    config: dict[str, JSONValue],
) -> OpenAICompatiblePolicy:
    return OpenAICompatiblePolicy(config)
