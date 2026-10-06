from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def _wire(value: Any) -> Any:
    if is_dataclass(value):
        return _wire(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_wire(item) for item in value]
    if isinstance(value, list):
        return [_wire(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _wire(item) for key, item in value.items()}
    return value


class ArenaRunWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "actions.jsonl",
            "observations.jsonl",
            "scientific_events.jsonl",
            "mechanism_events.jsonl",
        ):
            (self.output_dir / name).write_text("")

    def write_json(self, name: str, value: Any) -> None:
        (self.output_dir / name).write_text(
            json.dumps(_wire(value), sort_keys=True, indent=2) + "\n"
        )

    def append_jsonl(self, name: str, value: Any) -> None:
        with (self.output_dir / name).open("a") as handle:
            handle.write(
                json.dumps(_wire(value), sort_keys=True, separators=(",", ":"))
                + "\n"
            )
