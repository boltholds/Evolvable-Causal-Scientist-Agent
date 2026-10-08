"""KAN feedback ablation: raw vs verbalized numeric/symbolic state text.

The pretrained embedding backbone is frozen. Each arm uses the same opaque
observed X/Y transitions, warmup weights, KAN architecture and optimizer
steps. Counterfactual Y is generated only for evaluation by the benchmark
world, never used by fitting or calibration.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from math import isfinite
from pathlib import Path

import numpy as np
import torch

from .feedback_encoder import (
    FeedbackConfig, FeedbackMode, _encoded_samples, _feedback_fit,
    _fit_warmup, _pair_scores,
)
from .pretrained_text_study import (
    DEFAULT_MODEL, SentenceTransformerTextEncoder, TextEmbeddingPort,
)
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


@dataclass(frozen=True)
class VerbalizationConfig:
    feedback: FeedbackConfig = FeedbackConfig(
        bootstrap=24, adaptation=16, calibration=8, heldout=24,
        warmup_steps=65, feedback_steps=35, checkpoint_every=7,
        feature_size=128,
    )
    seq_length: int = 256
    base_dimension: int = 64
    include_feedback: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.feedback, FeedbackConfig):
            raise TypeError("typed FeedbackConfig required")
        if type(self.seq_length) is not int or self.seq_length < 32:
            raise ValueError("invalid token sequence length")
        if self.feedback.feature_size != 2*self.base_dimension:
            raise ValueError("feedback input width must be 2x pretrained embedding dim")
        if type(self.include_feedback) is not bool:
            raise TypeError("feedback selection must be boolean")


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


@dataclass(frozen=True)
class VerbalizationScore:
    law: StudyLaw
    seed: int
    arm: EncodedArm
    mode: FeedbackMode
    domain: str
    auroc: float
    paired_accuracy: float
    brier: float
    initial_auroc: float
    train_pairs: int
    heldout_pairs: int
    input_size: int
    kan_parameters: int
    backbone_id: str
    backbone_revision: str | None
    feedback_steps: int
    adapter_version: int
    hypothesis_version: int
    calibration_nll: float
    min_spread: float
    accepted_epochs: int
    rejected_epochs: int
    train_test_identity_overlap: int
    token_truncation_rate: float | None
    observed_max_tokens: int | None


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


def benchmark_one(
    law: StudyLaw, seed: int, config: VerbalizationConfig, arm: EncodedArm,
    base: TextEmbeddingPort | None, *, backbone_id: str = "pretrained",
    backbone_revision: str | None = None,
) -> tuple[VerbalizationScore, ...]:
    """Run paired KAN-feedback ablation on the same raw world transitions."""
    torch.set_num_threads(1)
    f = config.feedback
    total = f.bootstrap + f.adaptation + f.calibration
    train = _simulation(law, seed*101+9, total, heldout=False)
    test = _simulation(law, seed*101+109, f.heldout, heldout=True)
    overlap = _identity_overlap(train, test)
    truncation_rate, max_tokens = _token_audit(base, arm, train+test, config.seq_length)
    if overlap:
        raise ValueError("heldout object identities leak into training")
    port = ControlledTextEmbedding(base, arm=arm, dimension=config.base_dimension)
    x, y = _encoded_samples(port, train)
    split = f.bootstrap + f.adaptation
    warm = _fit_warmup(x[:f.bootstrap], y[:f.bootstrap], f, seed=seed)
    cx, cy = x[split:], y[split:]
    evaluations = (
        ("heldout", test, None),
        ("paraphrase", _paraphrase(test), None),
        ("counterfactual", test, _negative_law_outcomes(test, law)),
    )
    initial_scores = {domain: _pair_scores(warm, port, rows, negative)[0]
                      for domain, rows, negative in evaluations}
    result: list[VerbalizationScore] = []
    modes = (FeedbackMode.FROZEN, FeedbackMode.ALTERNATING) if config.include_feedback else (FeedbackMode.FROZEN,)
    for mode in modes:
        model, audit = _feedback_fit(
            warm, x[:split], y[:split], cx, cy, f, mode, seed=seed,
        )
        for domain, rows, negative in evaluations:
            auc, paired, brier = _pair_scores(model, port, rows, negative)
            result.append(VerbalizationScore(
                law, seed, arm, mode, domain, auc, paired, brier, initial_scores[domain],
                len(train), len(test), port.dimension,
                sum(p.numel() for p in model.parameters()),
                backbone_id if arm not in (EncodedArm.NUMERIC, EncodedArm.HASH) else arm.value,
                backbone_revision if arm not in (EncodedArm.NUMERIC, EncodedArm.HASH) else None,
                audit.optimizer_steps, audit.representation_version,
                audit.mechanism_version, audit.final_calibration_nll,
                audit.final_spread, audit.accepted_epochs, audit.rejected_epochs,
                overlap, truncation_rate, max_tokens,
            ))
    return tuple(result)


def run_benchmark(
    *, seeds: tuple[int, ...], laws: tuple[StudyLaw, ...],
    arms: tuple[EncodedArm, ...], config: VerbalizationConfig,
    base: TextEmbeddingPort | None, backbone_id: str,
    backbone_revision: str | None,
) -> tuple[VerbalizationScore, ...]:
    if not seeds or not laws or not arms or len(set(arms)) != len(arms):
        raise ValueError("nonempty unique benchmark axes required")
    if any(not isinstance(l, StudyLaw) for l in laws) or any(not isinstance(a, EncodedArm) for a in arms):
        raise TypeError("typed benchmark axes required")
    return tuple(row for law in laws for seed in seeds for arm in arms
                 for row in benchmark_one(
                     law, seed, config, arm, base,
                     backbone_id=backbone_id, backbone_revision=backbone_revision,
                 ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1])
    parser.add_argument("--laws", nargs="+", choices=[l.value for l in StudyLaw], default=[l.value for l in StudyLaw])
    parser.add_argument("--arms", nargs="+", choices=[a.value for a in EncodedArm], default=[a.value for a in EncodedArm])
    parser.add_argument("--bootstrap", type=int, default=24)
    parser.add_argument("--adaptation", type=int, default=16)
    parser.add_argument("--calibration", type=int, default=8)
    parser.add_argument("--heldout", type=int, default=24)
    parser.add_argument("--warmup-steps", type=int, default=65)
    parser.add_argument("--feedback-steps", type=int, default=35)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    f = FeedbackConfig(
        bootstrap=args.bootstrap, adaptation=args.adaptation,
        calibration=args.calibration, heldout=args.heldout,
        feature_size=2*args.dimension, warmup_steps=args.warmup_steps,
        feedback_steps=args.feedback_steps,
        checkpoint_every=min(7, args.feedback_steps),
    )
    config = VerbalizationConfig(feedback=f, seq_length=args.max_seq_length,
                                  base_dimension=args.dimension,
                                  include_feedback=not args.no_feedback)
    arms = tuple(EncodedArm(a) for a in args.arms)
    use_backbone = any(a not in (EncodedArm.NUMERIC, EncodedArm.HASH) for a in arms)
    base = SentenceTransformerTextEncoder(
        args.model, dimension=args.dimension, revision=args.revision,
        device=args.device, max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    ) if use_backbone else None
    rows = run_benchmark(
        seeds=tuple(args.seeds), laws=tuple(StudyLaw(s) for s in args.laws),
        arms=arms, config=config, base=base,
        backbone_id=args.model if use_backbone else "none",
        backbone_revision=args.revision,
    )
    result = {
        "config": asdict(config),
        "model": args.model if use_backbone else "none",
        "revision": args.revision if use_backbone else None,
        "rows": [asdict(row) for row in rows],
        "warnings": [
            "Synthetic hidden-law benchmark: no universal numeric-language claim.",
            "Classification compatibility AUROC is NOT calibrated world-transition likelihood.",
            "Evaluation counterfactuals are generated only for scoring, not training.",
            "Token-length audit flags inputs that exceed the configured sequence length; truncated arms cannot be compared as full-text representations.",
            "Numeric scanner and hash are diagnostic control representations, not full language models.",
        ],
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    summary = {}
    for domain in ("heldout", "paraphrase", "counterfactual"):
        summary[domain] = {}
        for arm in arms:
            rr = [r for r in rows if r.arm is arm and r.domain == domain]
            summary[domain][arm.value] = {
                mode.value: round(float(np.mean([r.auroc for r in rr if r.mode is mode])), 4)
                for mode in (FeedbackMode.FROZEN, FeedbackMode.ALTERNATING)
                if any(r.mode is mode for r in rr)
            }
    print(json.dumps({"rows": len(rows), "by_domain": summary}, indent=2))


if __name__ == "__main__":
    main()
