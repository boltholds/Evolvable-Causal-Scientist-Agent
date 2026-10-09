"""Public-log replay boundary for action-conditioned JEPA on DiscoveryWorld.

Only run metadata, actions.jsonl and observations.jsonl are read. No scorecard,
world history, hidden scenario attributes or task solution ever reaches features.
Feature hashing is a deliberately simple baseline, NOT learned perception.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from math import asinh, isfinite, sqrt
from pathlib import Path
import json
import re

import numpy as np

_ORACLE_KEYS = frozenset({
    'scoringinfo', 'criticalquestions', 'criticalhypotheses', 'scorecard',
    'final_scorecard', 'resonancefreq', 'isactivated', 'worldhistory',
    'exportworldhistoryjson',
})
_TRANSIENT_KEYS = frozenset({'worldstep', 'step'})
_WORD = re.compile(r'(?u)[\w]+', re.UNICODE)
_MAX_FEATURE_TOKENS = 6000


@dataclass(frozen=True)
class ReplayConfig:
    observation_dim: int = 256
    action_dim: int = 128

    def __post_init__(self) -> None:
        if type(self.observation_dim) is not int or self.observation_dim < 8:
            raise ValueError('observation_dim must be >= 8')
        if type(self.action_dim) is not int or self.action_dim < 8:
            raise ValueError('action_dim must be >= 8')


def _walk_public(value: object, path: str, *, kind: str):
    if isinstance(value, dict):
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError('JSON object keys must be strings')
            normalized = key.lower().replace('_', '')
            if normalized in _ORACLE_KEYS:
                raise ValueError(f'oracle data is prohibited in public {kind}: {key}')
            if normalized in _TRANSIENT_KEYS or normalized == 'uuid':
                # Neither episode clocks nor unstable object UUIDs should be
                # predictive shortcuts. Action identifiers remain public tokens.
                continue
            yield from _walk_public(value[key], f'{path}/{key}', kind=kind)
        return
    if isinstance(value, list):
        yield f'{path}:length:{min(len(value),32)}', 0.4
        for element in value:
            yield from _walk_public(element, f'{path}/[]', kind=kind)
        return
    if isinstance(value, str):
        yield f'{path}:present', 0.3
        for word in _WORD.findall(value.lower())[:100]:
            yield f'{path}:word:{word[:48]}', 1.0
        return
    if type(value) is bool:
        yield f'{path}:bool:{int(value)}', 1.0
        return
    if type(value) in (float, int):
        if not isfinite(float(value)):
            raise ValueError('nonfinite public numeric field')
        scaled = asinh(float(value))
        yield f'{path}:number', min(max(scaled / 5.0, -2.0), 2.0)
        yield f'{path}:bin:{min(max(round(scaled * 2),-60),60)}', 0.5
        return
    if value is None:
        yield f'{path}:null', 0.2
        return
    raise ValueError('unsupported public JSON value')


def encode_public_json(value: object, *, width: int, kind: str) -> np.ndarray:
    """Deterministic permutation-invariant JSON token sketch (no Python hash)."""
    if type(width) is not int or width < 8:
        raise ValueError('feature width must be >= 8')
    if kind not in ('observation', 'action'):
        raise ValueError('kind must be observation/action')
    vector = np.zeros(width, dtype=np.float32)
    count = 0
    for key, weight in _walk_public(value, kind, kind=kind):
        if count >= _MAX_FEATURE_TOKENS:
            raise ValueError('public observation exceeds fixed feature budget')
        digest = blake2b(key.encode('utf-8'), digest_size=8, person=b'ecsa-dw1').digest()
        idx = int.from_bytes(digest[:4], 'little') % width
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[idx] += sign * weight
        count += 1
    if count:
        vector /= sqrt(count)
    np.clip(vector, -4., 4., out=vector)
    return vector


@dataclass(frozen=True)
class ReplayTransitions:
    before: np.ndarray
    actions: np.ndarray
    after: np.ndarray
    action_names: tuple[str, ...]
    steps: tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.before)
        if (self.before.ndim != 2 or self.after.shape != self.before.shape
            or self.actions.ndim != 2 or len(self.actions) != n
            or len(self.action_names) != n or len(self.steps) != n or not n
            or not all(np.isfinite(a).all() for a in (self.before, self.actions, self.after))):
            raise ValueError('replay transitions must have finite aligned matrices')


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f'required public replay log missing: {path}')
    result = []
    with path.open('r', encoding='utf-8') as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path.name}:{number} must be an object')
            result.append(row)
    return result


def _action_name(value: dict) -> str:
    name = value.get('action')
    if isinstance(name, str) and name:
        return name
    option = value.get('chosen_dialog_option_int')
    if type(option) is int and option >= 0:
        return 'DIALOG_OPTION'
    raise ValueError('invalid public action packet')


def load_episode(directory: Path, config: ReplayConfig) -> ReplayTransitions:
    directory = Path(directory)
    metadata = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    if (not isinstance(metadata, dict) or metadata.get('scenario') != 'Reactor Lab'
        or metadata.get('difficulty') != 'Normal'
        or type(metadata.get('seed')) is not int):
        raise ValueError('only identified Reactor Lab / Normal public logs are supported')
    if directory.parent.name.startswith('seed-'):
        if metadata['seed'] != int(directory.parent.name[5:]):
            raise ValueError('run seed metadata does not match folder')
    actions = _jsonl(directory / 'actions.jsonl')
    observations = _jsonl(directory / 'observations.jsonl')
    pre: dict[int, dict] = {}
    post: dict[int, dict] = {}
    for row in observations:
        step, phase, value = row.get('step'), row.get('phase'), row.get('observation')
        if type(step) is not int or step < 0 or phase not in ('pre', 'post') or not isinstance(value, dict):
            raise ValueError('invalid public observation record')
        by_phase = pre if phase == 'pre' else post
        if step in by_phase:
            raise ValueError('duplicate observation phase and step')
        by_phase[step] = value
    if len(actions) != len(pre) or len(actions) != len(post):
        raise ValueError('pre/action/post alignment count mismatch')
    before, after, packets, names, steps = [], [], [], [], []
    seen_steps: set[int] = set()
    for row in actions:
        step, action = row.get('step'), row.get('action')
        if type(step) is not int or step < 0 or not isinstance(action, dict):
            raise ValueError('invalid public action log row')
        if step in seen_steps or step not in pre or step + 1 not in post:
            raise ValueError('pre/action/post alignment step mismatch')
        seen_steps.add(step)
        before.append(encode_public_json(pre[step], width=config.observation_dim, kind='observation'))
        after.append(encode_public_json(post[step + 1], width=config.observation_dim, kind='observation'))
        packets.append(encode_public_json(action, width=config.action_dim, kind='action'))
        names.append(_action_name(action))
        steps.append(step)
    if not steps or steps != sorted(steps):
        raise ValueError('nonempty monotonically ordered replay required')
    return ReplayTransitions(np.stack(before), np.stack(packets), np.stack(after),tuple(names),tuple(steps))


def _slice(rows: ReplayTransitions, indices: slice) -> ReplayTransitions:
    return ReplayTransitions(rows.before[indices], rows.actions[indices],
                             rows.after[indices], rows.action_names[indices], rows.steps[indices])


def _combine(rows: tuple[ReplayTransitions, ...]) -> ReplayTransitions:
    return ReplayTransitions(np.concatenate([r.before for r in rows]),
                             np.concatenate([r.actions for r in rows]),
                             np.concatenate([r.after for r in rows]),
                             tuple(n for r in rows for n in r.action_names),
                             tuple(s for r in rows for s in r.steps))


@dataclass(frozen=True)
class ReplaySplits:
    train: ReplayTransitions
    validation: ReplayTransitions
    test: ReplayTransitions
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    split_kind: str


def load_splits(root: Path, config: ReplayConfig, *, mode: str, arm: str = 'cold') -> ReplaySplits:
    if mode not in ('smoke', 'study'):
        raise ValueError('mode must be smoke/study')
    if arm not in ('cold', 'reuse'):
        raise ValueError('arm must be cold/reuse')
    root = Path(root)
    folders = sorted(root.glob(f'seed-*/{arm}'), key=lambda p: int(p.parent.name[5:]))
    if not folders and root.name == arm and (root / 'actions.jsonl').is_file():
        folders = [root]
    if not folders:
        raise ValueError('no DiscoveryWorld public episode logs found under seed-N/arm')
    by_seed: dict[int, ReplayTransitions] = {}
    for folder in folders:
        metadata = json.loads((folder / 'run.json').read_text())
        seed = metadata.get('seed')
        if type(seed) is not int or seed in by_seed:
            raise ValueError('each run must have a distinct numeric seed')
        by_seed[seed] = load_episode(folder, config)
    seeds = tuple(sorted(by_seed))
    if mode == 'study':
        if len(seeds) < 3:
            raise ValueError('study requires at least three distinct seeds for independent train/validation/test')
        train_seeds, validation_seeds, test_seeds = seeds[:-2], (seeds[-2],), (seeds[-1],)
        return ReplaySplits(_combine(tuple(by_seed[s] for s in train_seeds)),
                            by_seed[validation_seeds[0]], by_seed[test_seeds[0]],
                            train_seeds, validation_seeds, test_seeds, 'independent_seed_holdout')
    if len(seeds) != 1:
        raise ValueError('smoke requires exactly one seed: use study for multiple seeds')
    rows=by_seed[seeds[0]]
    n=len(rows.before)
    if n < 15:
        raise ValueError('smoke needs at least 15 logged actions')
    cut1,cut2=int(n*.6),int(n*.8)
    return ReplaySplits(_slice(rows,slice(0,cut1)), _slice(rows,slice(cut1,cut2)),
                        _slice(rows,slice(cut2,n)),seeds,seeds,seeds,
                        'single_episode_temporal_probe_only')
