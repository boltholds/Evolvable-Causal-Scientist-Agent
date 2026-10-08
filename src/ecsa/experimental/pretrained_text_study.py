"""Qwen3/frozen vs hash vs LoRA vs Qwen+numeric: implicit X/Y relation study.

All held-out labels are hidden from training. A train-only contrastive LoRA
pass adapts the full-text encoder when explicitly requested. The same KAN
head, X/Y projection towers, downstream fit steps and train/test pairs are
used for every arm. Adapted LoRA adds extra backbone training compute.

Large-model downloads are opt-in. The offline contract tests use a stub port.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from ecsa.experimental.relation_discovery import Dataset, HeadKind, fit, probability
from ecsa.experimental.text_first_kan import (
    DualEncoderRelation, FixtureLaw, FullTextFeatureProjector, HashedTextFeatures,
    InputRepresentation, TextKANConfig, _auc, _matrix, _negatives, _simulation,
)
from ecsa.world_model.text_relations import TextRelationSample


# Pinning is possible through --revision; absent revision means the model's
# current default branch, and a result cannot be considered fully reproducible.
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


@runtime_checkable
class TextEmbeddingPort(Protocol):
    @property
    def dimension(self) -> int: ...
    def encode_many(self, texts: tuple[str, ...]) -> np.ndarray: ...


class EncoderArm(StrEnum):
    HASH = "hash_text"
    FROZEN = "frozen_pretrained_text"
    HYBRID = "frozen_pretrained_hybrid"
    NUMERIC = "numeric_only"
    LORA = "lora_pretrained_text"


class EvalDomain(StrEnum):
    HELDOUT = "heldout"
    PARAPHRASE = "paraphrase"
    CONTRASTED_MODE = "contrasted_mode"


@dataclass(frozen=True)
class PretrainedStudyConfig:
    train_samples: int = 64
    heldout_samples: int = 32
    feature_size: int = 64
    train_steps: int = 80
    max_seq_length: int = 256
    batch_size: int = 8
    lr: float = 0.012
    lora_steps: int = 12
    lora_batch_size: int = 4
    lora_lr: float = 0.0002

    def __post_init__(self) -> None:
        for field in (
            "train_samples", "heldout_samples", "feature_size", "train_steps",
            "max_seq_length", "batch_size", "lora_steps", "lora_batch_size",
        ):
            number = getattr(self, field)
            if type(number) is not int or number < (16 if field == "feature_size" else 1):
                raise ValueError(f"{field} must be a positive integer")
        if not (0 < self.lr < 0.1 and 0 < self.lora_lr < 0.01):
            raise ValueError("invalid learning rates")


class SentenceTransformerTextEncoder:
    """Frozen or adapter-tuned SentenceTransformer with batched, cached inference.

    If `model` is supplied, Hugging Face is never contacted. Production use
    requires the optional pretrained-encoder dependencies; they are NOT core.
    """

    def __init__(
        self, model_id: str = DEFAULT_MODEL, *, dimension: int = 64,
        revision: str | None = None, device: str = "cpu",
        max_seq_length: int = 256, batch_size: int = 8, model: object | None = None,
    ) -> None:
        if not model_id or type(dimension) is not int or dimension < 16:
            raise ValueError("model_id and dimension are required")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Install optional dependencies: pip install -e '.[pretrained-encoder]'"
                ) from exc
            self.model = SentenceTransformer(
                model_id, revision=revision, device=device, trust_remote_code=False,
            )
        else:
            self.model = model
        self.model_id = model_id
        self.revision = revision
        self._dimension = dimension
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.model.max_seq_length = max_seq_length
        self._cache: dict[str, np.ndarray] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def clear_cache(self) -> None:
        self._cache.clear()

    def encode_many(self, texts: tuple[str, ...]) -> np.ndarray:
        if not isinstance(texts, tuple) or any(not isinstance(t, str) for t in texts):
            raise TypeError("immutable tuple of text strings required")
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        new = tuple(dict.fromkeys(text for text in texts if text not in self._cache))
        if new:
            values = self.model.encode(
                list(new), batch_size=self.batch_size,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False, truncate_dim=self.dimension,
            )
            result = np.asarray(values, dtype=np.float32)
            if result.shape != (len(new), self.dimension) or not np.isfinite(result).all():
                raise ValueError("embedding dimensions/values did not match requested size")
            for text, embedding in zip(new, result):
                self._cache[text] = embedding.copy()
        return np.stack([self._cache[text] for text in texts]).astype(np.float32)


def _paraphrase(sample: TextRelationSample) -> TextRelationSample:
    """Hold out alternate wordings; preserve raw numbers, values and labels."""
    from dataclasses import replace
    x = json.loads(sample.x_text)
    scene = x.get("before", {}).get("scene", "")
    scene = scene.replace("The object is ", "An item has the appearance ")
    scene = scene.replace("The control reads ", "; instrument reading ")
    x["before"]["scene"] = scene
    y = json.loads(sample.y_text)
    if y.get("after", {}).get("observation") == "The measurement was recorded.":
        y["after"]["observation"] = "A reading was logged."
    return replace(
        sample,
        x_text=json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        y_text=json.dumps(y, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
    )


def _opposite_mode_outcomes(
    samples: tuple[TextRelationSample, ...], law: FixtureLaw,
) -> tuple[TextRelationSample, ...]:
    """Evaluation-only counterfactual: X unchanged, Y from opposite mode.

    This requires the benchmark's hidden law and is NEVER used in training.
    Keeping numeric X fixed removes the conditional-magnitude shortcut.
    """
    if law not in (FixtureLaw.CATEGORICAL, FixtureLaw.INTERACTION):
        raise ValueError("only laws that depend on the categorical mode qualify")
    counterfactual: list[TextRelationSample] = []
    for sample in samples:
        x = json.loads(sample.x_text)["before"]
        u = float(x["controls"]["left"])
        v = float(x["controls"]["right"])
        sign = 1 if x["object"]["hue"] == "amber" else -1
        if law is FixtureLaw.CATEGORICAL:
            y = -sign * .75
        else:
            y = -sign * (.72 * u - .18 * v + .12)
        outcome = json.loads(sample.y_text)
        outcome["after"]["measurement"] = round(float(y), 4)
        outcome["after"]["description"] = "a high signal" if y > 0 else "a low signal"
        counterfactual.append(replace(
            sample,
            y_text=json.dumps(outcome, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":")),
        ))
    return tuple(counterfactual)


class ComparableFeatures:
    """Same input width in every arm: text dim | numeric dim."""

    def __init__(
        self, dimension: int, *, hash_encoder: HashedTextFeatures | None = None,
        pretrained: TextEmbeddingPort | None = None,
    ) -> None:
        self.dimension = dimension
        self.hash_encoder = hash_encoder or HashedTextFeatures(dimension)
        self.pretrained = pretrained
        self._numeric = FullTextFeatureProjector(dimension=dimension)
        if pretrained is not None and pretrained.dimension != dimension:
            raise ValueError("text feature dimension must match")

    def make(
        self, samples: tuple[TextRelationSample, ...], arm: EncoderArm,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(arm, EncoderArm):
            raise TypeError("typed arm required")
        if not samples:
            raise ValueError("nonempty observed transition samples required")
        x_text = tuple(sample.x_text for sample in samples)
        y_text = tuple(sample.y_text for sample in samples)
        texts = x_text + y_text
        n = len(samples)
        if arm is EncoderArm.HASH:
            encoded = np.stack([self.hash_encoder.encode(s) for s in texts])
        elif arm is EncoderArm.NUMERIC:
            encoded = np.zeros((len(texts), self.dimension), dtype=np.float32)
        else:
            if self.pretrained is None:
                raise ValueError("pretrained encoder required for requested arm")
            encoded = self.pretrained.encode_many(texts)
        if arm in (EncoderArm.NUMERIC, EncoderArm.HYBRID):
            numeric = np.stack([
                self._numeric.encode(s, InputRepresentation.NUMERIC)[self.dimension:]
                for s in texts
            ])
        else:
            numeric = np.zeros((len(texts), self.dimension), dtype=np.float32)
        # Always expose the same input shape to identical KAN+projections.
        combined = np.concatenate((encoded, numeric), axis=1).astype(np.float32)
        return combined[:n], combined[n:]


@dataclass(frozen=True)
class Score:
    law: str
    seed: int
    arm: EncoderArm
    eval_domain: EvalDomain
    auroc: float
    paired_accuracy: float
    brier: float
    kan_parameters: int
    train_pairs: int
    eval_pairs: int
    train_test_identity_overlap: int
    model_id: str
    revision: str | None
    encoder_feature_dim: int
    lora_steps: int


def _hard_negatives(samples: tuple[TextRelationSample, ...]) -> np.ndarray:
    """Nearer contrast than maximally distant Y, with a fixed gap.

    Y is inspected solely while *building evaluation labels*, never to select
    training pairs/negative samples. The threshold avoids ambiguous near ties.
    """
    ys = np.asarray([
        float(json.loads(sample.y_text)["after"]["measurement"])
        for sample in samples
    ])
    gaps = np.abs(ys[:, None] - ys[None, :])
    np.fill_diagonal(gaps, np.inf)
    candidates = np.where(gaps >= .18, gaps, np.inf)
    nearest = np.argmin(candidates, axis=1)
    impossible = np.isinf(candidates[np.arange(len(ys)), nearest])
    if impossible.any():
        nearest[impossible] = _negatives(samples)[impossible]
    return nearest


def _score(
    model: DualEncoderRelation, samples: tuple[TextRelationSample, ...],
    arm: EncoderArm, features: ComparableFeatures, *, law: FixtureLaw,
    seed: int, train: tuple[TextRelationSample, ...], domain: EvalDomain,
    model_id: str, revision: str | None, lora_steps: int,
    negative_samples: tuple[TextRelationSample, ...] | None = None,
) -> Score:
    x, y = features.make(samples, arm)
    if negative_samples is None:
        negatives = _hard_negatives(samples)
        y_negative = y[negatives]
    else:
        if len(negative_samples) != len(samples):
            raise ValueError("counterfactual length mismatch")
        x_negative, y_negative = features.make(negative_samples, arm)
        if not np.allclose(x_negative, x):
            raise ValueError("counterfactual changed pre-action X")
    with torch.no_grad():
        p = probability(model, np.concatenate((x, x)), np.concatenate((y, y_negative)))
    truth = np.concatenate((np.ones(len(samples)), np.zeros(len(samples))))
    score = float(((p-truth)**2).mean())
    train_ids = {json.loads(sample.x_text)["before"]["object"]["identity"] for sample in train}
    test_ids = {json.loads(sample.x_text)["before"]["object"]["identity"] for sample in samples}
    return Score(
        law.value, seed, arm, domain, _auc(p, truth),
        float((p[:len(samples)] > p[len(samples):]).mean()), score,
        sum(t.numel() for t in model.parameters()), len(train), len(samples),
        len(train_ids & test_ids), model_id, revision, features.dimension,
        lora_steps,
    )


def _train_kan(
    train: tuple[TextRelationSample, ...], features: ComparableFeatures,
    arm: EncoderArm, *, seed: int, config: PretrainedStudyConfig,
) -> DualEncoderRelation:
    x, y = features.make(train, arm)
    data = Dataset(
        np.empty((len(train), 0), dtype=np.float32),
        np.empty((len(train),), dtype=np.float32), x, y,
    )
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed * 1000 + 43)
    model = DualEncoderRelation(HeadKind.SPLINE_KAN, 2 * config.feature_size)
    fit(model, data, seed=seed * 1000 + 113, steps=config.train_steps, lr=config.lr)
    return model


def _move_model_features(features: dict[str, object], device: torch.device) -> dict[str, object]:
    """SentenceTransformer may include non-Tensor tokenizer metadata."""
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in features.items()}


def fit_lora_on_observed_pairs(
    encoder: SentenceTransformerTextEncoder,
    pairs: tuple[TextRelationSample, ...], *,
    steps: int, batch_size: int, lr: float, seed: int,
    allow_cpu: bool = False,
) -> int:
    """Adapt embedding backbone using ONLY observed training transitions.

    Symmetric in-batch X/Y contrastive task with possible false negatives.
    This is an adaptation of the representation objective, not a KAN loss.
    Downstream KAN receives the same fitting budget across all arms.
    """
    from peft import LoraConfig, TaskType

    model = encoder.model
    device = next(model.parameters()).device
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError("LoRA training needs GPU; use --allow-cpu-lora for small smoke tests")
    if not pairs or type(steps) is not int or steps <= 0:
        raise ValueError("nonempty observed pairs and positive LoRA steps required")
    torch.manual_seed(seed * 1000 + 981)
    model.add_adapter(LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.FEATURE_EXTRACTION,
    ))
    trainable = tuple(p for p in model.parameters() if p.requires_grad)
    if not trainable:
        raise RuntimeError("LoRA did not create trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    rng = np.random.default_rng(seed * 1000 + 991)
    x_texts = tuple(pair.x_text for pair in pairs)
    y_texts = tuple(pair.y_text for pair in pairs)
    model.train()
    for _ in range(steps):
        ids = rng.choice(len(pairs), size=min(batch_size, len(pairs)), replace=False)
        left = model.tokenize([x_texts[i] for i in ids])
        right = model.tokenize([y_texts[i] for i in ids])
        left = _move_model_features(left, device)
        right = _move_model_features(right, device)
        zx = model(left)["sentence_embedding"][:, :encoder.dimension]
        zy = model(right)["sentence_embedding"][:, :encoder.dimension]
        logits = (torch.nn.functional.normalize(zx, dim=1) @
                  torch.nn.functional.normalize(zy, dim=1).T) / .08
        labels = torch.arange(len(ids), device=device)
        loss = (torch.nn.functional.cross_entropy(logits, labels)
                + torch.nn.functional.cross_entropy(logits.T, labels)) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    model.eval()
    encoder.clear_cache()
    return sum(p.numel() for p in trainable)


def benchmark(
    *, model_id: str = DEFAULT_MODEL, revision: str | None = None,
    config: PretrainedStudyConfig = PretrainedStudyConfig(),
    seeds: tuple[int, ...] = (0,), laws: tuple[FixtureLaw, ...] = tuple(FixtureLaw),
    arms: tuple[EncoderArm, ...] = (
        EncoderArm.HASH, EncoderArm.FROZEN, EncoderArm.HYBRID, EncoderArm.NUMERIC,
    ),
    device: str = "cpu", allow_cpu_lora: bool = False,
    sentence_model: object | None = None,
) -> tuple[Score, ...]:
    if not seeds or not laws or not arms or len(set(arms)) != len(arms):
        raise ValueError("nonempty unique benchmark dimensions required")
    if any(not isinstance(law, FixtureLaw) for law in laws) or any(
        not isinstance(arm, EncoderArm) for arm in arms
    ):
        raise TypeError("typed laws and arms required")
    wants_pretrained = any(arm in (EncoderArm.FROZEN, EncoderArm.HYBRID, EncoderArm.LORA) for arm in arms)
    needs_lora = EncoderArm.LORA in arms
    if needs_lora and len(seeds) * len(laws) > 1:
        raise ValueError("LoRA arm retrains encoder per dataset: run one law/seed per job")
    if needs_lora and arms[-1] is not EncoderArm.LORA:
        raise ValueError("LoRA must run last to preserve frozen-backbone comparisons")
    encoder = (
        SentenceTransformerTextEncoder(
            model_id, dimension=config.feature_size, revision=revision,
            device=device, batch_size=config.batch_size,
            max_seq_length=config.max_seq_length, model=sentence_model,
        ) if wants_pretrained else None
    )
    features = ComparableFeatures(config.feature_size, pretrained=encoder)
    rows: list[Score] = []
    for law in laws:
        for seed in seeds:
            train = _simulation(law, seed * 101 + 9, config.train_samples, heldout=False)
            test = _simulation(law, seed * 101 + 109, config.heldout_samples, heldout=True)
            # Precompute all text features in a single batched encoder call. The
            # cache belongs to a specified model revision and is train/eval agnostic.
            if encoder is not None and not needs_lora:
                encoder.encode_many(tuple(
                    s for sample in (train + test) for s in (sample.x_text, sample.y_text)
                ))
            for arm in arms:
                if arm is EncoderArm.LORA:
                    assert encoder is not None
                    fit_lora_on_observed_pairs(
                        encoder, train, steps=config.lora_steps,
                        batch_size=config.lora_batch_size, lr=config.lora_lr,
                        seed=seed, allow_cpu=allow_cpu_lora,
                    )
                if encoder is not None and arm is EncoderArm.LORA:
                    encoder.encode_many(tuple(
                        s for sample in (train + test) for s in (sample.x_text, sample.y_text)
                    ))
                model = _train_kan(train, features, arm, seed=seed, config=config)
                domains = [
                    (EvalDomain.HELDOUT, test, None),
                    (EvalDomain.PARAPHRASE, tuple(_paraphrase(sample) for sample in test), None),
                ]
                if law in (FixtureLaw.CATEGORICAL, FixtureLaw.INTERACTION):
                    domains.append((
                        EvalDomain.CONTRASTED_MODE, test,
                        _opposite_mode_outcomes(test, law),
                    ))
                for domain, observed, negative in domains:
                    rows.append(_score(
                        model, observed, arm, features,
                        law=law, seed=seed, train=train, domain=domain,
                        model_id=(model_id if arm in (
                            EncoderArm.FROZEN, EncoderArm.HYBRID, EncoderArm.LORA,
                        ) else "non-pretrained"),
                        revision=revision if arm in (
                            EncoderArm.FROZEN, EncoderArm.HYBRID, EncoderArm.LORA,
                        ) else None,
                        lora_steps=config.lora_steps if arm is EncoderArm.LORA else 0,
                        negative_samples=negative,
                    ))
    return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--laws", nargs="+", default=[law.value for law in FixtureLaw],
                        choices=[law.value for law in FixtureLaw])
    parser.add_argument("--arms", nargs="+", default=[
        EncoderArm.HASH.value, EncoderArm.FROZEN.value,
        EncoderArm.HYBRID.value, EncoderArm.NUMERIC.value,
    ], choices=[a.value for a in EncoderArm])
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--heldout-samples", type=int, default=32)
    parser.add_argument("--feature-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--lora-steps", type=int, default=12)
    parser.add_argument("--lora-batch-size", type=int, default=4)
    parser.add_argument("--allow-cpu-lora", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    config = PretrainedStudyConfig(
        train_samples=args.train_samples, heldout_samples=args.heldout_samples,
        feature_size=args.feature_size, train_steps=args.steps,
        max_seq_length=args.max_seq_length, batch_size=args.batch_size,
        lora_steps=args.lora_steps, lora_batch_size=args.lora_batch_size,
    )
    result = benchmark(
        model_id=args.model, revision=args.revision,
        seeds=tuple(args.seeds), laws=tuple(FixtureLaw(s) for s in args.laws),
        arms=tuple(EncoderArm(s) for s in args.arms),
        config=config, device=args.device, allow_cpu_lora=args.allow_cpu_lora,
    )
    data = {
        "model": args.model, "revision": args.revision, "config": asdict(config),
        "device": args.device, "seeds": args.seeds,
        "results": [asdict(r) for r in result],
        "limitations": (
            "Implicit pair-recognition on controlled synthetic transitions. "
            "Only the measured encoder arms were executed. "
            "The pretrained backbone is frozen except for explicit LoRA arm. "
            "Heldout paraphrase is a limited text shift, not proof of universality. "
            "Hard mismatched Y were selected from evaluation outcomes; "
            "this is neither causal identification nor calibrated p(Y|X). "
            "LoRA is trained on matched X/Y pairs using in-batch negatives "
            "and has additional compute relative to frozen models."
        ),
    }
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "runs": len(result),
        "mean_auc": {
            arm.value: {
                domain.value: round(float(np.mean([
                    r.auroc for r in result if r.arm is arm and r.eval_domain is domain
                ])), 4)
                for domain in EvalDomain if any(r.arm is arm and r.eval_domain is domain for r in result)
            }
            for arm in (EncoderArm(value) for value in args.arms)
        },
    }, indent=2))


if __name__ == "__main__":
    main()
