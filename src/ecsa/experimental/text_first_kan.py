"""Reproducible text-first vs numeric vs hybrid X/Y KAN ablation.

Only the benchmark fixture knows its hidden laws. Model inputs are generated
through FullTextTransitionProjector from ordinary raw before/action/after.
A deterministic n-gram hash is the *text feature front-end*, followed by
learned independent X/Y encoder towers and a spline-KAN relation head.
It is not a pretrained semantic embedding model or proof of causal discovery.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import blake2b
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import Tensor, nn

from ecsa.world_model.contracts import (
    GroundAction, InteractionTransition, RawActionOutcome, RawObservation,
    freeze_raw_value,
)
from ecsa.world_model.text_relations import FullTextTransitionProjector, TextRelationSample
from .relation_discovery import (
    Dataset, HeadKind, MLPHead, SplineKANHead, fit, probability,
)


class InputRepresentation(StrEnum):
    TEXT = "text"
    NUMERIC = "numeric"
    HYBRID = "hybrid"


class FixtureLaw(StrEnum):
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    INTERACTION = "interaction"


@dataclass(frozen=True)
class TextKANConfig:
    samples: int = 144
    holdout: int = 96
    feature_size: int = 64
    training_steps: int = 130
    lr: float = 0.012

    def __post_init__(self) -> None:
        for key in ("samples", "holdout", "feature_size", "training_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 4:
                raise ValueError(f"{key} must be a positive integer >=4")
        if self.feature_size < 16:
            raise ValueError("feature_size must be >=16")
        if not 0.0 < self.lr < 0.1:
            raise ValueError("invalid learning rate")


_TOKEN = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)


def _hash_slot(text: str, dimension: int) -> tuple[int, int]:
    digest = blake2b(text.encode("utf-8"), digest_size=8, person=b"ECSA-TXT").digest()
    number = int.from_bytes(digest, "little")
    return (number % dimension, 1 if (number >> 8) & 1 else -1)


class HashedTextFeatures:
    """Fixed lexical n-gram feature layer; trainable semantics live in towers.

    Keeps all characters, numbers, signs and punctuation. Hash collisions are
    possible; this layer should eventually be replaced by a pretrained model
    through an independent TextFeaturePort, not sold as language understanding.
    """

    def __init__(self, dimension: int) -> None:
        if type(dimension) is not int or dimension < 16:
            raise ValueError("hash dimension must be >=16")
        self.dimension = dimension

    def encode(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError("text required")
        tokens = _TOKEN.findall(text.lower())
        features = np.zeros(self.dimension, dtype=np.float32)
        for token in tokens:
            slot, sign = _hash_slot("token:" + token, self.dimension)
            features[slot] += sign
        for left, right in zip(tokens, tokens[1:]):
            slot, sign = _hash_slot("bigram:" + left + "/" + right, self.dimension)
            features[slot] += sign
        # Character n-grams preserve small changes and the exact spelling of
        # unknown entities, including negation, minus signs and decimals.
        for width in (3, 4):
            for i in range(max(0, len(text) - width + 1)):
                slot, sign = _hash_slot("char:" + text[i:i+width], self.dimension)
                features[slot] += sign * .25
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
        return features


class TextFeaturePort(Protocol):
    """Feature-port interface; swap hash features for pretrained E()."""

    def encode(self, text: str) -> np.ndarray: ...


def _numeric_values(data: object, prefix: str = "") -> tuple[tuple[str, float], ...]:
    """Generic JSON numeric scanner: no named-feature or benchmark law hints."""
    items: list[tuple[str, float]] = []
    if isinstance(data, dict):
        for key, child in sorted(data.items()):
            items.extend(_numeric_values(child, f"{prefix}/{key}"))
    elif isinstance(data, (tuple, list)):
        for i, child in enumerate(data):
            items.extend(_numeric_values(child, f"{prefix}[{i}]"))
    elif type(data) in (int, float) and np.isfinite(float(data)):
        items.append((prefix, float(data)))
    return tuple(items)


def _numeric_hash(text: str, dimension: int) -> np.ndarray:
    parsed = json.loads(text)
    result = np.zeros(dimension, dtype=np.float32)
    for key, number in _numeric_values(parsed):
        slot, sign = _hash_slot("number:" + key, dimension)
        # Preserve magnitude and sign without fixed semantic unit mapping.
        result[slot] += sign * np.tanh(number / 2.0)
    return result


class FullTextFeatureProjector:
    def __init__(self, *, dimension: int = 64, encoder: HashedTextFeatures | None = None):
        self.dimension = dimension
        self.encoder = encoder if encoder is not None else HashedTextFeatures(dimension)

    def encode(self, text: str, representation: InputRepresentation) -> np.ndarray:
        if not isinstance(representation, InputRepresentation):
            raise TypeError("typed representation required")
        lexical = self.encoder.encode(text) if representation != InputRepresentation.NUMERIC else np.zeros(self.dimension, np.float32)
        numeric = _numeric_hash(text, self.dimension) if representation != InputRepresentation.TEXT else np.zeros(self.dimension, np.float32)
        # Two fixed-width feature channels in every arm: parameter counts match.
        return np.concatenate((lexical, numeric)).astype(np.float32)


class _TwinEncoders(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        def tower():
            return nn.Sequential(
                nn.Linear(input_size, 12), nn.Tanh(),
                nn.Linear(12, 2), nn.Tanh(),
            )
        self.x = tower()
        self.y = tower()

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        return self.x(x), self.y(y)


class DualEncoderRelation(nn.Module):
    """E_X(X), E_Y(Y) -> KAN/MLP compatibility score, never direct Y=f(X)."""

    def __init__(self, kind: HeadKind, input_size: int) -> None:
        super().__init__()
        self.encoder = _TwinEncoders(input_size)
        self.relation = SplineKANHead() if kind is HeadKind.SPLINE_KAN else MLPHead()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        zx, zy = self.encoder(x, y)
        return self.relation(torch.cat((zx, zy), dim=1)).flatten()


@dataclass(frozen=True)
class BenchmarkRow:
    law: str
    seed: int
    representation: InputRepresentation
    relation_head: HeadKind
    auroc: float
    paired_accuracy: float
    parameter_count: int
    train_observations: int
    heldout_observations: int
    train_object_id_overlap: int


def _simulation(law: FixtureLaw, seed: int, count: int, *, heldout: bool) -> tuple[TextRelationSample, ...]:
    """Benchmark world generates pairs, then exposes only typed raw transitions."""
    rng = np.random.default_rng(seed)
    projector = FullTextTransitionProjector()
    rows: list[TextRelationSample] = []
    for i in range(count):
        u, v = rng.uniform(-1.05 if heldout else -.9, 1.05 if heldout else .9, 2)
        mode = str(rng.choice(("amber", "violet")))
        mode_sign = 1 if mode == "amber" else -1
        if law is FixtureLaw.CATEGORICAL:
            y = mode_sign * 0.75
        elif law is FixtureLaw.NUMERIC:
            y = .8*u - .45*v + .12*u*v
        elif law is FixtureLaw.INTERACTION:
            y = mode_sign * (.72*u - .18*v) + .12*mode_sign
        else:
            raise ValueError(law)
        y += rng.normal(0.0, .025)
        # No shared identifier in Y: otherwise a model could learn the trivial
        # identity correspondence rather than the intended transition relation.
        entity_id = f"{('eval' if heldout else 'train')}-entity-{seed}-{i}"
        before = RawObservation(
            f"pre:{entity_id}", i,
            freeze_raw_value({
                "scene": f"The object is {mode}. The control reads {u:.4f}.",
                "object": {"identity": entity_id, "hue": mode},
                "controls": {"left": round(float(u), 4), "right": round(float(v), 4)},
            }),
        )
        after = RawObservation(
            f"post:{entity_id}", i+1,
            freeze_raw_value({
                "observation": "The measurement was recorded.",
                "measurement": round(float(y), 4),
                "description": "a high signal" if y > 0 else "a low signal",
            }),
        )
        action = GroundAction("opaque-A0", (freeze_raw_value("activate"),))
        raw = InteractionTransition(
            f"transition:{entity_id}", before, action,
            RawActionOutcome(True, freeze_raw_value({})), after,
        )
        rows.append(projector.project(raw))
    return tuple(rows)


def _matrix(samples: tuple[TextRelationSample, ...], mode: InputRepresentation,
            projector: FullTextFeatureProjector) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([projector.encode(s.x_text, mode) for s in samples]),
        np.stack([projector.encode(s.y_text, mode) for s in samples]),
    )


def _negatives(samples: tuple[TextRelationSample, ...]) -> np.ndarray:
    # Evaluation-only clean incompatibilities; nearest different measurement
    # selected from real observed Ys. NO membership oracle is given to learner.
    ys = np.asarray([json.loads(s.y_text)["after"]["measurement"] for s in samples])
    distance = np.abs(ys[:, None] - ys[None, :])
    return np.argmax(distance, axis=1)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    return float((pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean())


def benchmark_one(law: FixtureLaw, seed: int, config: TextKANConfig,
                  representation: InputRepresentation, kind: HeadKind) -> BenchmarkRow:
    torch.set_num_threads(1)
    train = _simulation(law, seed * 101 + 9, config.samples, heldout=False)
    test = _simulation(law, seed * 101 + 109, config.holdout, heldout=True)
    feature = FullTextFeatureProjector(dimension=config.feature_size)
    x_train, y_train = _matrix(train, representation, feature)
    x_test, y_test = _matrix(test, representation, feature)
    # Generic fit() uses only x_view and y_view. x/y below are placeholders,
    # NOT ground-truth physical features or labels available to the learner.
    ds = Dataset(np.empty((len(train), 0), np.float32),
                 np.empty(len(train), np.float32), x_train, y_train)
    torch.manual_seed(seed * 1000 + 43)
    model = DualEncoderRelation(kind, input_size=2 * config.feature_size)
    fit(model, ds, seed=seed * 1000 + 113, steps=config.training_steps, lr=config.lr)
    j = _negatives(test)
    p = probability(model, np.concatenate((x_test, x_test)),
                    np.concatenate((y_test, y_test[j])))
    labels = np.concatenate((np.ones(len(test)), np.zeros(len(test))))
    train_ids = {json.loads(row.x_text)["before"]["object"]["identity"] for row in train}
    eval_ids = {json.loads(row.x_text)["before"]["object"]["identity"] for row in test}
    return BenchmarkRow(
        law=law.value, seed=seed,
        representation=representation, relation_head=kind,
        auroc=_auc(p, labels),
        paired_accuracy=float((p[:len(test)] > p[len(test):]).mean()),
        parameter_count=sum(v.numel() for v in model.parameters()),
        train_observations=len(train), heldout_observations=len(test),
        train_object_id_overlap=len(train_ids.intersection(eval_ids)),
    )


def run_benchmark(*, seeds: tuple[int, ...] = (0, 1, 2),
                  config: TextKANConfig = TextKANConfig()) -> tuple[BenchmarkRow, ...]:
    return tuple(benchmark_one(law, seed, config, rep, kind)
                 for law in FixtureLaw for seed in seeds
                 for rep in InputRepresentation
                 for kind in (HeadKind.SPLINE_KAN, HeadKind.MLP))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--samples", type=int, default=144)
    parser.add_argument("--holdout", type=int, default=96)
    parser.add_argument("--steps", type=int, default=130)
    parser.add_argument("--feature-size", type=int, default=64)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    config = TextKANConfig(samples=args.samples, holdout=args.holdout,
                           training_steps=args.steps, feature_size=args.feature_size)
    rows = run_benchmark(seeds=tuple(args.seeds), config=config)
    result = {"config": asdict(config), "seeds": args.seeds,
              "results": [asdict(row) for row in rows],
              "warnings": (
                  "Small synthetic contrastive relation-recognition task. "
                  "Hash n-grams are NOT pretrained semantic embeddings; "
                  "evaluation negatives are selected using Y distance; "
                  "this does not demonstrate causal identification, learned "
                  "symbolic laws, or real-world planning."
              )}
    body = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + "\n", encoding="utf-8")
    print(json.dumps({
        "total": len(rows),
        "auroc_by_law_and_representation": {
            law.value: {
                rep.value: {
                    kind.value: round(float(np.mean([r.auroc for r in rows
                            if r.law == law.value and r.representation is rep and
                            r.relation_head is kind])), 4)
                    for kind in (HeadKind.SPLINE_KAN, HeadKind.MLP)
                } for rep in InputRepresentation
            } for law in FixtureLaw
        }
    }, indent=2))


if __name__ == "__main__":
    main()
