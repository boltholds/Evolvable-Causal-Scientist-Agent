from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from .hypotheses import (
    ActionSchemaHypothesis,
    WorldContractHypothesis,
)


@dataclass(frozen=True)
class CanonicalWorldContract:
    fingerprint: str
    node_signatures: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.fingerprint:
            raise ValueError("canonical fingerprint is required")
        if not isinstance(self.node_signatures, tuple):
            raise ValueError("node_signatures must be immutable")


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(encoded.encode()).hexdigest()


def _normalize_parameter_signatures(
    action: ActionSchemaHypothesis,
    signatures: tuple[str, ...],
) -> tuple[str, ...]:
    normalized = list(signatures)
    for group in action.symmetric_parameter_groups:
        values = sorted(signatures[index] for index in group)
        for index, value in zip(sorted(group), values):
            normalized[index] = value
    return tuple(normalized)


def _position_token(
    action: ActionSchemaHypothesis,
    index: int,
) -> str:
    for group in action.symmetric_parameter_groups:
        if index in group:
            return f"sym:{len(group)}"
    return f"pos:{index}"


def canonicalize_world_contract(
    contract: WorldContractHypothesis,
) -> CanonicalWorldContract:
    type_by_id = {
        value.hypothesis_id: value
        for value in contract.entity_types
    }
    pred_by_id = {
        value.hypothesis_id: value
        for value in contract.predicates
    }
    numeric_by_id = {
        value.hypothesis_id: value
        for value in contract.numeric_fluents
    }
    action_by_id = {
        value.hypothesis_id: value
        for value in contract.actions
    }

    type_sig = {
        key: _digest(("entity-type",))
        for key in type_by_id
    }
    pred_sig = {
        key: _digest(("predicate", len(value.argument_type_ids), value.symmetric))
        for key, value in pred_by_id.items()
    }
    numeric_sig = {
        key: _digest(("numeric", len(value.argument_type_ids)))
        for key, value in numeric_by_id.items()
    }
    action_sig = {
        key: _digest(
            (
                "action",
                len(value.parameter_type_ids),
                tuple(
                    sorted(len(group) for group in value.symmetric_parameter_groups)
                ),
                len(value.precondition_predicate_ids),
                len(value.add_effect_predicate_ids),
                len(value.delete_effect_predicate_ids),
                len(value.numeric_effect_ids),
            )
        )
        for key, value in action_by_id.items()
    }
    affordance_sig = {
        value.hypothesis_id: _digest(
            (
                "affordance",
                len(value.argument_type_ids),
                round(float(value.success_probability), 12),
            )
        )
        for value in contract.affordances
    }

    for _ in range(12):
        next_pred = {}
        for key, predicate in pred_by_id.items():
            args = tuple(
                type_sig[type_id]
                for type_id in predicate.argument_type_ids
            )
            if predicate.symmetric:
                args = tuple(sorted(args))
            next_pred[key] = _digest(
                ("predicate", predicate.symmetric, args)
            )

        next_numeric = {
            key: _digest(
                (
                    "numeric",
                    tuple(
                        type_sig[type_id]
                        for type_id in fluent.argument_type_ids
                    ),
                )
            )
            for key, fluent in numeric_by_id.items()
        }

        next_action = {}
        for key, action in action_by_id.items():
            params = _normalize_parameter_signatures(
                action,
                tuple(
                    type_sig[type_id]
                    for type_id in action.parameter_type_ids
                ),
            )
            next_action[key] = _digest(
                (
                    "action",
                    params,
                    tuple(
                        sorted(
                            next_pred[predicate_id]
                            for predicate_id
                            in action.precondition_predicate_ids
                        )
                    ),
                    tuple(
                        sorted(
                            next_pred[predicate_id]
                            for predicate_id
                            in action.add_effect_predicate_ids
                        )
                    ),
                    tuple(
                        sorted(
                            next_pred[predicate_id]
                            for predicate_id
                            in action.delete_effect_predicate_ids
                        )
                    ),
                    tuple(
                        sorted(
                            next_numeric[numeric_id]
                            for numeric_id in action.numeric_effect_ids
                        )
                    ),
                    tuple(
                        sorted(
                            tuple(sorted(group))
                            for group in action.symmetric_parameter_groups
                        )
                    ),
                )
            )

        next_affordance = {}
        for affordance in contract.affordances:
            action = action_by_id[affordance.action_schema_id]
            args = _normalize_parameter_signatures(
                action,
                tuple(
                    type_sig[type_id]
                    for type_id in affordance.argument_type_ids
                ),
            )
            next_affordance[affordance.hypothesis_id] = _digest(
                (
                    "affordance",
                    next_action[affordance.action_schema_id],
                    args,
                    round(float(affordance.success_probability), 12),
                )
            )

        next_type = {}
        for type_id in type_by_id:
            incidence: list[object] = []
            for predicate_id, predicate in pred_by_id.items():
                for index, argument_type_id in enumerate(
                    predicate.argument_type_ids
                ):
                    if argument_type_id == type_id:
                        incidence.append(
                            (
                                "predicate",
                                next_pred[predicate_id],
                                "sym" if predicate.symmetric else index,
                            )
                        )
            for numeric_id, fluent in numeric_by_id.items():
                for index, argument_type_id in enumerate(
                    fluent.argument_type_ids
                ):
                    if argument_type_id == type_id:
                        incidence.append(
                            (
                                "numeric",
                                next_numeric[numeric_id],
                                index,
                            )
                        )
            for action_id, action in action_by_id.items():
                for index, argument_type_id in enumerate(
                    action.parameter_type_ids
                ):
                    if argument_type_id == type_id:
                        incidence.append(
                            (
                                "action",
                                next_action[action_id],
                                _position_token(action, index),
                            )
                        )
            for affordance in contract.affordances:
                action = action_by_id[affordance.action_schema_id]
                for index, argument_type_id in enumerate(
                    affordance.argument_type_ids
                ):
                    if argument_type_id == type_id:
                        incidence.append(
                            (
                                "affordance",
                                next_affordance[affordance.hypothesis_id],
                                _position_token(action, index),
                            )
                        )
            next_type[type_id] = _digest(
                (
                    "entity-type",
                    tuple(
                        sorted(
                            incidence,
                            key=lambda item: json.dumps(
                                item,
                                sort_keys=True,
                            ),
                        )
                    ),
                )
            )

        if (
            next_type == type_sig
            and next_pred == pred_sig
            and next_numeric == numeric_sig
            and next_action == action_sig
            and next_affordance == affordance_sig
        ):
            break
        type_sig = next_type
        pred_sig = next_pred
        numeric_sig = next_numeric
        action_sig = next_action
        affordance_sig = next_affordance

    role_signatures = tuple(
        _digest(
            (
                "argument-role",
                action_sig[role.action_schema_id],
                _position_token(
                    action_by_id[role.action_schema_id],
                    role.parameter_index,
                ),
                type_sig[role.entity_type_id],
            )
        )
        for role in contract.argument_roles
    )
    node_signatures = tuple(
        sorted(
            (
                *type_sig.values(),
                *pred_sig.values(),
                *numeric_sig.values(),
                *action_sig.values(),
                *affordance_sig.values(),
                *role_signatures,
            )
        )
    )
    fingerprint = _digest(
        (
            "world-contract",
            node_signatures,
            (
                len(type_sig),
                len(pred_sig),
                len(numeric_sig),
                len(action_sig),
                len(affordance_sig),
                len(role_signatures),
            ),
        )
    )
    return CanonicalWorldContract(
        fingerprint=fingerprint,
        node_signatures=node_signatures,
    )
