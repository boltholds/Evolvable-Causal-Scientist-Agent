from __future__ import annotations

import re
from hashlib import sha256

from ecsa.autonomy import GenericScalarEvidence

from .contracts import (
    ActionPacket,
    JSONValue,
)


_NUMBER = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d+)?"
)


def _dialog_text(observation: dict[str, JSONValue]) -> str:
    ui = observation.get("ui")
    if not isinstance(ui, dict):
        return ""
    dialog = ui.get("dialog_box")
    if not isinstance(dialog, dict):
        return ""
    value = dialog.get("dialogIn")
    return value if isinstance(value, str) else ""


def _public_text_sources(
    observation: dict[str, JSONValue],
) -> tuple[tuple[str, str], ...]:
    ui = observation.get("ui")
    if not isinstance(ui, dict):
        return ()

    found: list[tuple[str, str]] = []
    for key in ("lastActionMessage", "extended_action_message"):
        value = ui.get(key)
        if isinstance(value, str) and value.strip():
            found.append((key, value))

    dialog = _dialog_text(observation)
    if dialog.strip():
        found.append(("dialog", dialog))
    return tuple(found)


def _action_name(action: ActionPacket) -> str:
    ordinary = action.get("action")
    if isinstance(ordinary, str) and ordinary:
        return ordinary
    if type(action.get("chosen_dialog_option_int")) is int:
        return "DIALOG_OPTION"
    return "UNKNOWN"


def _action_entity_ids(action: ActionPacket) -> tuple[int, ...]:
    found: list[int] = []
    for key in ("arg1", "arg2"):
        value = action.get(key)
        if type(value) is int and value >= 0:
            found.append(value)
    return tuple(found)


def _feature_key(
    *,
    source_name: str,
    line: str,
    selected_index: int,
) -> str:
    matches = tuple(_NUMBER.finditer(line))
    parts: list[str] = []
    cursor = 0
    for index, match in enumerate(matches):
        parts.append(line[cursor:match.start()].lower())
        parts.append("<value>" if index == selected_index else "<number>")
        cursor = match.end()
    parts.append(line[cursor:].lower())
    normalized = " ".join("".join(parts).split())
    digest = sha256(
        f"{source_name}:{normalized}".encode()
    ).hexdigest()[:20]
    return f"public-scalar:{digest}"


def extract_public_scalar_evidence(
    *,
    step: int,
    context_id: str,
    action: ActionPacket,
    post_observation: dict[str, JSONValue],
    dialog_entity_ids: tuple[int, ...] = (),
    source_names: frozenset[str] | None = None,
) -> tuple[GenericScalarEvidence, ...]:
    if type(step) is not int or step < 0:
        raise ValueError("step must be nonnegative")
    if not context_id:
        raise ValueError("context_id is required")

    direct_ids = _action_entity_ids(action)
    entity_ids = direct_ids or dialog_entity_ids
    action_name = _action_name(action)
    found: list[GenericScalarEvidence] = []

    for source_name, text in _public_text_sources(post_observation):
        if source_names is not None and source_name not in source_names:
            continue
        for line_index, raw_line in enumerate(text.splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            matches = tuple(_NUMBER.finditer(line))
            for number_index, match in enumerate(matches):
                raw_value = match.group(0).replace(",", "")
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                feature_key = _feature_key(
                    source_name=source_name,
                    line=line,
                    selected_index=number_index,
                )
                identity = (
                    f"{context_id}:{step}:{action_name}:{entity_ids}:"
                    f"{source_name}:{line_index}:{number_index}:"
                    f"{feature_key}:{value}"
                )
                evidence_id = (
                    "dw-public-scalar:"
                    + sha256(identity.encode()).hexdigest()
                )
                found.append(
                    GenericScalarEvidence(
                        evidence_id=evidence_id,
                        step=step,
                        action_name=action_name,
                        entity_ids=entity_ids,
                        feature_key=feature_key,
                        value=value,
                        raw_text=line,
                    )
                )
    return tuple(found)


class GenericEvidenceLedger:
    def __init__(self) -> None:
        self._evidence: list[GenericScalarEvidence] = []
        self._evidence_ids: set[str] = set()
        self._dialog_entity_ids: tuple[int, ...] = ()

    @property
    def evidence(self) -> tuple[GenericScalarEvidence, ...]:
        return tuple(self._evidence)

    def record_transition(
        self,
        *,
        step: int,
        context_id: str,
        pre_observation: dict[str, JSONValue],
        action: ActionPacket,
        post_observation: dict[str, JSONValue],
    ) -> tuple[GenericScalarEvidence, ...]:
        pre_dialog = _dialog_text(pre_observation)
        post_dialog = _dialog_text(post_observation)
        direct_ids = _action_entity_ids(action)

        if post_dialog and not pre_dialog and direct_ids:
            self._dialog_entity_ids = direct_ids

        pre_sources = dict(_public_text_sources(pre_observation))
        post_sources = dict(_public_text_sources(post_observation))
        changed_sources = frozenset(
            source_name
            for source_name, text in post_sources.items()
            if pre_sources.get(source_name) != text
        )
        created = extract_public_scalar_evidence(
            step=step,
            context_id=context_id,
            action=action,
            post_observation=post_observation,
            dialog_entity_ids=self._dialog_entity_ids,
            source_names=changed_sources,
        )
        unique: list[GenericScalarEvidence] = []
        for item in created:
            if item.evidence_id in self._evidence_ids:
                continue
            self._evidence_ids.add(item.evidence_id)
            self._evidence.append(item)
            unique.append(item)

        if pre_dialog and not post_dialog:
            self._dialog_entity_ids = ()
        return tuple(unique)
