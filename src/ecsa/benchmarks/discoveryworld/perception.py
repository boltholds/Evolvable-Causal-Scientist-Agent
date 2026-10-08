from __future__ import annotations

from collections import defaultdict
from math import isfinite

from ecsa.world_model.contracts import (
    RawObservation,
    freeze_raw_value,
)
from ecsa.world_model.perception.base import ObservedFeature
from ecsa.world_model.perception.structured import (
    StructuredEntityRecord,
)


def _object_groups(payload: dict) -> tuple[tuple[str, list], ...]:
    ui = payload.get("ui")
    if not isinstance(ui, dict):
        return ()

    groups: list[tuple[str, list]] = []
    for name in (
        "inventoryObjects",
        "accessibleEnvironmentObjects",
    ):
        value = ui.get(name)
        if isinstance(value, list):
            groups.append((name, value))

    nearby = ui.get("nearbyObjects")
    if isinstance(nearby, dict):
        values = nearby.get("objects")
        if isinstance(values, dict):
            for name, items in sorted(values.items()):
                if isinstance(items, list):
                    groups.append(
                        (f"nearby:{name}", items)
                    )
    return tuple(groups)


class DiscoveryWorldStructuredDecoder:
    def decode_entities(
        self,
        raw_observation: RawObservation,
    ) -> tuple[StructuredEntityRecord, ...]:
        root = raw_observation.payload.thaw()
        if not isinstance(root, dict):
            return ()
        payload = root.get("observation")
        if not isinstance(payload, dict):
            return ()

        records: dict[int, dict[str, object]] = {}
        memberships: dict[int, set[str]] = defaultdict(set)

        for group_name, items in _object_groups(payload):
            for item in items:
                if not isinstance(item, dict):
                    continue
                entity_id = item.get("uuid")
                if type(entity_id) is not int:
                    continue
                records[entity_id] = dict(item)
                memberships[entity_id].add(group_name)

        result: list[StructuredEntityRecord] = []
        for entity_id in sorted(records):
            item = records[entity_id]
            features: list[ObservedFeature] = [
                ObservedFeature(
                    feature_id="public:identity",
                    value=freeze_raw_value(entity_id),
                    provenance_id=raw_observation.observation_id,
                )
            ]
            for key in ("name", "description"):
                value = item.get(key)
                if isinstance(value, str):
                    features.append(
                        ObservedFeature(
                            feature_id=f"public:{key}",
                            value=freeze_raw_value(value),
                            provenance_id=(
                                raw_observation.observation_id
                            ),
                        )
                    )
            # Transport all exposed scalar object features without assigning
            # domain-specific meanings. Identity is already source_identity.
            for key, value in sorted(item.items()):
                if key in ("uuid", "name", "description"):
                    continue
                if type(value) in (bool, int, float) and (
                    type(value) is bool or isfinite(float(value))
                ):
                    features.append(
                        ObservedFeature(
                            feature_id=f"public:{key}",
                            value=freeze_raw_value(value),
                            provenance_id=raw_observation.observation_id,
                        )
                    )
            for membership in sorted(
                memberships.get(entity_id, ())
            ):
                features.append(
                    ObservedFeature(
                        feature_id=(
                            "public:membership:"
                            + membership
                        ),
                        value=freeze_raw_value(True),
                        provenance_id=(
                            raw_observation.observation_id
                        ),
                    )
                )

            result.append(
                StructuredEntityRecord(
                    local_ref=f"dw:{entity_id}",
                    source_identity=str(entity_id),
                    features=tuple(features),
                )
            )
        return tuple(result)
