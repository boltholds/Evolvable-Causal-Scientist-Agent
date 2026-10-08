"""Benchmark-only verbalized-state fixtures and frozen embedding adapters.

These fixtures retain historical synthetic X/Y environments for comparison.
KAN-to-encoder feedback, encoder adaptation and standalone feedback ablations
are retired; earlier experiments remain accessible through Git history.
"""
from __future__ import annotations

import json
from dataclasses import replace
from enum import StrEnum

import numpy as np

from .pretrained_text_study import TextEmbeddingPort
from .text_first_kan import HashedTextFeatures, _numeric_hash
from .verbalized_state import VerbalizationMode, verbalize_json
from ecsa.world_model.contracts import (
    RawObservation, GroundAction, RawActionOutcome, InteractionTransition,
    freeze_raw_value,
)
from ecsa.world_model.text_relations import FullTextTransitionProjector, TextRelationSample


class StudyLaw(StrEnum):
    CONTINUOUS = "continuous"
    PRECISION = "precision"
    OPERATOR = "operator"
    SIGN = "sign"
    RANGE_SHIFT = "range_shift"


class EncodedArm(StrEnum):
    RAW = "raw"
    SEMANTIC = "semantic_words"
    DIGIT = "digit_words"
    RAW_PLUS = "raw_plus_words"
    HASH = "hash_control"
    NUMERIC = "numeric_control"
    HYBRID = "raw_plus_numeric_control"


class ControlledTextEmbedding:
    """Pure feature adapter over a frozen embedding port; no hidden-law access."""

    def __init__(self, base: TextEmbeddingPort | None, *, arm: EncodedArm,
                 dimension: int = 64) -> None:
        if not isinstance(arm, EncodedArm):
            raise TypeError("typed arm required")
        if type(dimension) is not int or dimension < 16:
            raise ValueError("feature dimension must be at least 16")
        if base is None and arm not in (EncodedArm.HASH, EncodedArm.NUMERIC):
            raise ValueError("pretrained encoder required")
        if base is not None and base.dimension != dimension:
            raise ValueError("backbone dimension mismatch")
        self._base = base
        self.arm = arm
        self._base_dim = dimension
        self._hash = HashedTextFeatures(dimension)

    @property
    def dimension(self) -> int:
        return 2*self._base_dim

    def encode_many(self, texts: tuple[str, ...]) -> np.ndarray:
        if not isinstance(texts, tuple) or any(not isinstance(t, str) for t in texts):
            raise TypeError("immutable string tuple required")
        n = len(texts)
        if not n:
            return np.zeros((0, self.dimension), dtype=np.float32)
        features = np.zeros((n, self._base_dim), dtype=np.float32)
        numeric = np.zeros_like(features)
        if self.arm is EncodedArm.HASH:
            features = np.stack([self._hash.encode(t) for t in texts])
        elif self.arm is not EncodedArm.NUMERIC:
            assert self._base is not None
            input_mode = {
                EncodedArm.RAW: VerbalizationMode.RAW,
                EncodedArm.SEMANTIC: VerbalizationMode.SEMANTIC,
                EncodedArm.DIGIT: VerbalizationMode.DIGIT,
                EncodedArm.RAW_PLUS: VerbalizationMode.RAW_PLUS,
                EncodedArm.HYBRID: VerbalizationMode.RAW,
            }[self.arm]
            payload = tuple(verbalize_json(t, input_mode) for t in texts)
            features = np.asarray(self._base.encode_many(payload), dtype=np.float32)
        if self.arm in (EncodedArm.NUMERIC, EncodedArm.HYBRID):
            numeric = np.stack([_numeric_hash(t, self._base_dim) for t in texts])
        result = np.concatenate((features, numeric), axis=1)
        if result.shape != (n, self.dimension) or not np.isfinite(result).all():
            raise ValueError("embedding port emitted invalid features")
        return result.astype(np.float32)


def _compare(op: str, left: float, right: float) -> bool:
    return {
        ">": lambda: left > right,
        ">=": lambda: left >= right,
        "<": lambda: left < right,
        "<=": lambda: left <= right,
        "==": lambda: left == right,
        "!=": lambda: left != right,
    }[op]()


def _law_response(law: StudyLaw, u: float, v: float, op: str) -> float:
    if law is StudyLaw.CONTINUOUS or law is StudyLaw.RANGE_SHIFT:
        return 0.72*u - 0.38*v + 0.16*u*v
    if law is StudyLaw.PRECISION:
        return float(np.tanh((u-v)*125))
    if law is StudyLaw.OPERATOR:
        return 0.95 if _compare(op, u, v) else -0.95
    if law is StudyLaw.SIGN:
        return (0.80 if u >= 0 else -0.80) + 0.025*u
    raise ValueError(law)


def _simulation(law: StudyLaw, seed: int, n: int, *, heldout: bool) -> tuple[TextRelationSample, ...]:
    """Opaque textual transitions; fixture-specific law remains outside model."""
    if not isinstance(law, StudyLaw):
        raise TypeError("typed law required")
    rng = np.random.default_rng(seed)
    projector = FullTextTransitionProjector()
    items: list[TextRelationSample] = []
    operations = (">", ">=", "<", "<=", "==", "!=")
    for index in range(n):
        op = operations[int(rng.integers(0, len(operations)))]
        if law is StudyLaw.PRECISION:
            v = float(rng.uniform(10, 20))
            shift = float(rng.choice(
                (-.013, -.007, -.004, .004, .007, .013) if heldout else
                (-.02, -.01, -.005, .005, .01, .02)
            ))
            u = v + shift
        elif law is StudyLaw.OPERATOR:
            v = float(rng.uniform(-.85, .85))
            delta = float(rng.choice((0.0, -.15, .15, -.004, .004)))
            u = v + delta
        elif law is StudyLaw.RANGE_SHIFT and heldout:
            u = float(rng.choice((-1., 1.))*rng.uniform(2.0, 4.0))
            v = float(rng.choice((-1., 1.))*rng.uniform(2.0, 4.0))
        elif law is StudyLaw.SIGN:
            u = float(rng.choice((-1., 1.))*rng.uniform(.005, .92))
            v = float(rng.uniform(-.9, .9))
        else:
            u, v = (float(x) for x in rng.uniform(-.90, .90, 2))
        # Integer ID and leading-zero text IDs must not become measurements.
        identity = f"{'eval' if heldout else 'train'}-unit-{seed}-{index:04d}"
        u_token, v_token = f"{u:.5f}", f"{v:.5f}"
        state = {
            "object": {"identity": identity, "name": "sensor_A17"},
            "readings": {"primary": float(u_token), "reference": float(v_token)},
            "predicate": {"operator": op},
            "scene": f"the current meter reads {u_token}; the comparator {op} reference {v_token}",
        }
        y = _law_response(law, float(u_token), float(v_token), op)
        if law not in (StudyLaw.OPERATOR, StudyLaw.PRECISION):
            y += float(rng.normal(0, .018))
        action = GroundAction("unknown-action-0", (freeze_raw_value("apply"),))
        before = RawObservation("before:"+identity, index, freeze_raw_value(state))
        after = RawObservation("after:"+identity, index+1,
            freeze_raw_value({"measurement": float(f"{y:.5f}"), "observation": "recorded"}))
        transition = InteractionTransition(
            "event:"+identity, before, action,
            RawActionOutcome(True, freeze_raw_value({"event": "accepted"})), after,
        )
        items.append(projector.project(transition))
    return tuple(items)


def _negative_law_outcomes(
    rows: tuple[TextRelationSample, ...], law: StudyLaw,
) -> tuple[TextRelationSample, ...]:
    """Evaluation-only intervention counterfactual, preserving X exactly."""
    negatives = []
    for row in rows:
        before = json.loads(row.x_text)["before"]
        u = float(before["readings"]["primary"])
        v = float(before["readings"]["reference"])
        op = before["predicate"]["operator"]
        if law is StudyLaw.OPERATOR:
            swapped = {
                ">": "<=", ">=": "<", "<": ">=", "<=": ">",
                "==": "!=", "!=": "==",
            }[op]
            counter_y = _law_response(law, u, v, swapped)
        elif law is StudyLaw.PRECISION:
            counter_y = _law_response(law, v - (u - v), v, op)
        elif law is StudyLaw.SIGN:
            counter_y = _law_response(law, -u, v, op)
        else:
            counter_y = _law_response(law, -u, -v, op)
        after = json.loads(row.y_text)
        after["after"]["measurement"] = float(f"{counter_y:.5f}")
        negatives.append(replace(row, y_text=json.dumps(
            after, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        )))
    return tuple(negatives)


def _paraphrase(rows: tuple[TextRelationSample, ...]) -> tuple[TextRelationSample, ...]:
    output = []
    for row in rows:
        doc = json.loads(row.x_text)
        sentence = doc["before"]["scene"]
        doc["before"]["scene"] = sentence.replace(
            "the current meter reads ", "meter displays ",
        ).replace("the comparator ", "comparison relation ")
        outcome = json.loads(row.y_text)
        outcome["after"]["observation"] = "signal captured"
        output.append(replace(
            row,
            x_text=json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
            y_text=json.dumps(outcome, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
        ))
    return tuple(output)


def _token_audit(
    base: TextEmbeddingPort | None, arm: EncodedArm,
    rows: tuple[TextRelationSample, ...], seq_length: int,
) -> tuple[float | None, int | None]:
    """Report raw pre-truncation token lengths, where HF tokenizer exists.

    This audit neither modifies encoder state nor looks at outcome labels.
    """
    if base is None or arm in (EncodedArm.NUMERIC, EncodedArm.HASH):
        return None, None
    model = getattr(base, "model", None)
    if model is None:
        return None, None
    try:
        tokenizer = model[0].tokenizer
    except (TypeError, IndexError, AttributeError):
        return None, None
    input_mode = {
        EncodedArm.RAW: VerbalizationMode.RAW,
        EncodedArm.SEMANTIC: VerbalizationMode.SEMANTIC,
        EncodedArm.DIGIT: VerbalizationMode.DIGIT,
        EncodedArm.RAW_PLUS: VerbalizationMode.RAW_PLUS,
        EncodedArm.HYBRID: VerbalizationMode.RAW,
    }[arm]
    texts = tuple(verbalize_json(t, input_mode) for row in rows
                  for t in (row.x_text, row.y_text))
    lengths: list[int] = []
    for text in texts:
        tokens = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
        lengths.append(len(tokens))
    return sum(v > seq_length for v in lengths)/len(lengths), max(lengths)


def _identity_overlap(train: tuple[TextRelationSample, ...], test: tuple[TextRelationSample, ...]) -> int:
    left = {json.loads(s.x_text)["before"]["object"]["identity"] for s in train}
    right = {json.loads(s.x_text)["before"]["object"]["identity"] for s in test}
    return len(left & right)
