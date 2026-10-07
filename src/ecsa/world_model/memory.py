from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .canonical import (
    CanonicalWorldContract,
    canonicalize_world_contract,
)
from .hypotheses import (
    ActionSchemaHypothesis,
    AffordanceHypothesis,
    ArgumentRoleHypothesis,
    EntityTypeHypothesis,
    HypothesisStatus,
    NumericFluentHypothesis,
    PredicateHypothesis,
    WorldContractHypothesis,
)


@dataclass(frozen=True)
class WorldContractRef:
    contract_id: str
    version: int
    canonical_fingerprint: str

    def __post_init__(self) -> None:
        if not self.contract_id or not self.canonical_fingerprint:
            raise ValueError("world contract reference identifiers are required")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("world contract version must be positive")


@dataclass(frozen=True)
class WorldContractCandidate:
    ref: WorldContractRef
    contract: WorldContractHypothesis


class WorldContractStore(Protocol):
    def admit(
        self,
        contract: WorldContractHypothesis,
    ) -> WorldContractRef: ...

    def get(
        self,
        ref: WorldContractRef,
    ) -> WorldContractHypothesis: ...

    def find_transfer_candidates(
        self,
        canonical_context: CanonicalWorldContract,
    ) -> tuple[WorldContractCandidate, ...]: ...


def _status(value: str) -> HypothesisStatus:
    return HypothesisStatus(value)


def _common(wire: dict) -> dict:
    return {
        "supporting_evidence_ids": tuple(
            wire["supporting_evidence_ids"]
        ),
        "contradicting_evidence_ids": tuple(
            wire["contradicting_evidence_ids"]
        ),
        "confidence": float(wire["confidence"]),
        "status": _status(wire["status"]),
    }


def _contract_from_wire(wire: dict) -> WorldContractHypothesis:
    entity_types = tuple(
        EntityTypeHypothesis(
            hypothesis_id=item["hypothesis_id"],
            member_refs=tuple(item["member_refs"]),
            **_common(item),
        )
        for item in wire["entity_types"]
    )
    predicates = tuple(
        PredicateHypothesis(
            hypothesis_id=item["hypothesis_id"],
            argument_type_ids=tuple(item["argument_type_ids"]),
            symmetric=bool(item["symmetric"]),
            **_common(item),
        )
        for item in wire["predicates"]
    )
    numeric_fluents = tuple(
        NumericFluentHypothesis(
            hypothesis_id=item["hypothesis_id"],
            argument_type_ids=tuple(item["argument_type_ids"]),
            **_common(item),
        )
        for item in wire["numeric_fluents"]
    )
    argument_roles = tuple(
        ArgumentRoleHypothesis(
            hypothesis_id=item["hypothesis_id"],
            action_schema_id=item["action_schema_id"],
            parameter_index=int(item["parameter_index"]),
            entity_type_id=item["entity_type_id"],
            **_common(item),
        )
        for item in wire["argument_roles"]
    )
    actions = tuple(
        ActionSchemaHypothesis(
            hypothesis_id=item["hypothesis_id"],
            source_schema_id=item["source_schema_id"],
            parameter_type_ids=tuple(item["parameter_type_ids"]),
            precondition_predicate_ids=tuple(
                item["precondition_predicate_ids"]
            ),
            add_effect_predicate_ids=tuple(
                item["add_effect_predicate_ids"]
            ),
            delete_effect_predicate_ids=tuple(
                item["delete_effect_predicate_ids"]
            ),
            numeric_effect_ids=tuple(item["numeric_effect_ids"]),
            symmetric_parameter_groups=tuple(
                tuple(int(index) for index in group)
                for group in item["symmetric_parameter_groups"]
            ),
            **_common(item),
        )
        for item in wire["actions"]
    )
    affordances = tuple(
        AffordanceHypothesis(
            hypothesis_id=item["hypothesis_id"],
            action_schema_id=item["action_schema_id"],
            argument_type_ids=tuple(item["argument_type_ids"]),
            success_probability=float(item["success_probability"]),
            **_common(item),
        )
        for item in wire["affordances"]
    )
    return WorldContractHypothesis(
        contract_id=wire["contract_id"],
        entity_types=entity_types,
        predicates=predicates,
        numeric_fluents=numeric_fluents,
        argument_roles=argument_roles,
        actions=actions,
        affordances=affordances,
        supporting_evidence_ids=tuple(
            wire["supporting_evidence_ids"]
        ),
        contradicting_evidence_ids=tuple(
            wire["contradicting_evidence_ids"]
        ),
        confidence=float(wire["confidence"]),
        status=_status(wire["status"]),
    )


def _contract_to_json(contract: WorldContractHypothesis) -> str:
    return json.dumps(
        asdict(contract),
        sort_keys=True,
        separators=(",", ":"),
    )


class SQLiteWorldContractStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS world_contracts (
                    contract_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (contract_id, version)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_world_contracts_fingerprint
                ON world_contracts(fingerprint)
                """
            )

    def admit(
        self,
        contract: WorldContractHypothesis,
    ) -> WorldContractRef:
        if not isinstance(contract, WorldContractHypothesis):
            raise TypeError("admit requires WorldContractHypothesis")
        canonical = canonicalize_world_contract(contract)
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(version), 0)
                FROM world_contracts
                WHERE contract_id = ?
                """,
                (contract.contract_id,),
            ).fetchone()
            version = int(row[0]) + 1
            connection.execute(
                """
                INSERT INTO world_contracts(
                    contract_id,
                    version,
                    fingerprint,
                    payload
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    contract.contract_id,
                    version,
                    canonical.fingerprint,
                    _contract_to_json(contract),
                ),
            )
        return WorldContractRef(
            contract_id=contract.contract_id,
            version=version,
            canonical_fingerprint=canonical.fingerprint,
        )

    def get(
        self,
        ref: WorldContractRef,
    ) -> WorldContractHypothesis:
        if not isinstance(ref, WorldContractRef):
            raise TypeError("get requires WorldContractRef")
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                """
                SELECT fingerprint, payload
                FROM world_contracts
                WHERE contract_id = ? AND version = ?
                """,
                (ref.contract_id, ref.version),
            ).fetchone()
        if row is None:
            raise KeyError(
                f"unknown world contract version: "
                f"{ref.contract_id}@{ref.version}"
            )
        fingerprint, payload = row
        if fingerprint != ref.canonical_fingerprint:
            raise ValueError("world contract reference fingerprint mismatch")
        return _contract_from_wire(json.loads(payload))

    def find_transfer_candidates(
        self,
        canonical_context: CanonicalWorldContract,
    ) -> tuple[WorldContractCandidate, ...]:
        if not isinstance(canonical_context, CanonicalWorldContract):
            raise TypeError(
                "canonical_context must be CanonicalWorldContract"
            )
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                """
                SELECT c.contract_id, c.version, c.fingerprint, c.payload
                FROM world_contracts AS c
                JOIN (
                    SELECT contract_id, MAX(version) AS version
                    FROM world_contracts
                    WHERE fingerprint = ?
                    GROUP BY contract_id
                ) AS latest
                  ON latest.contract_id = c.contract_id
                 AND latest.version = c.version
                WHERE c.fingerprint = ?
                ORDER BY c.contract_id
                """,
                (
                    canonical_context.fingerprint,
                    canonical_context.fingerprint,
                ),
            ).fetchall()
        return tuple(
            WorldContractCandidate(
                ref=WorldContractRef(
                    contract_id=contract_id,
                    version=int(version),
                    canonical_fingerprint=fingerprint,
                ),
                contract=_contract_from_wire(json.loads(payload)),
            )
            for contract_id, version, fingerprint, payload in rows
        )
