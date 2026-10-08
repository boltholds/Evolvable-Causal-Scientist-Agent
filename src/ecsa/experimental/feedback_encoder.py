"""Hypothesis-weighted negative feedback from implicit KAN relations to encoders.

X -> frozen text backbone -> trainable E_X adapter -> [KAN hypotheses]
Y -> frozen text backbone -> trainable E_Y adapter -> [KAN hypotheses]

All hypothesis weights are *empirical calibration weights*, NOT posterior
probabilities over causal mechanisms. They are rebuilt after an adapter update:
its latent coordinate system has changed. This experiment never calls a world
oracle or uses held-out outcomes during feedback training / checkpoint gates.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .pretrained_text_study import (
    DEFAULT_MODEL,
    SentenceTransformerTextEncoder,
    TextEmbeddingPort,
    _hard_negatives,
    _opposite_mode_outcomes,
    _paraphrase,
)
from .relation_discovery import SplineKANHead
from .text_first_kan import FixtureLaw, _auc, _simulation
from ecsa.world_model.text_relations import TextRelationSample


class FeedbackMode(StrEnum):
    FROZEN = "frozen"
    KAN_ONLY = "kan_only"
    ADAPTER_ONLY = "adapter_only"
    ALTERNATING = "alternating"


@dataclass(frozen=True)
class FeedbackConfig:
    bootstrap: int = 48
    adaptation: int = 32
    calibration: int = 16
    heldout: int = 32
    feature_size: int = 64
    warmup_steps: int = 80
    feedback_steps: int = 60
    checkpoint_every: int = 10
    warmup_lr: float = 0.012
    kan_lr: float = 0.006
    adapter_lr: float = 0.004
    preservation_weight: float = 0.3
    geometry_weight: float = 0.08
    variance_weight: float = 0.1
    min_spread: float = 0.005
    calibration_tolerance: float = 0.18

    def __post_init__(self) -> None:
        for key in (
            "bootstrap", "adaptation", "calibration", "heldout",
            "feature_size", "warmup_steps", "feedback_steps", "checkpoint_every",
        ):
            val = getattr(self, key)
            if type(val) is not int or val < (16 if key == "feature_size" else 1):
                raise ValueError(f"{key} must be a positive integer (feature_size>=16)")
        if self.checkpoint_every > self.feedback_steps:
            raise ValueError("at least one feedback checkpoint required")
        for key in ("warmup_lr", "kan_lr", "adapter_lr"):
            val = getattr(self, key)
            if not 0 < val < 0.1 or not np.isfinite(val):
                raise ValueError(f"invalid {key}")
        for key in (
            "preservation_weight", "geometry_weight", "variance_weight",
            "min_spread", "calibration_tolerance",
        ):
            val = getattr(self, key)
            if not np.isfinite(val) or val < 0:
                raise ValueError(f"invalid {key}")


class _Projection(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 12), nn.Tanh(),
            nn.Linear(12, 2), nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class FeedbackRelationSystem(nn.Module):
    """Both adapters are shared by an ensemble of structural KAN hypotheses."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if type(input_dim) is not int or input_dim < 16:
            raise TypeError("positive integer feature dimension >=16 required")
        self.x_adapter = _Projection(input_dim)
        self.y_adapter = _Projection(input_dim)
        self.hypotheses = nn.ModuleList([
            SplineKANHead(knots=5, hidden=12),
            SplineKANHead(knots=7, hidden=9),
            SplineKANHead(knots=9, hidden=7),
        ])
        self.register_buffer("weights", torch.full((3,), 1. / 3.))
        self.adapter_version = 0
        self.hypothesis_version = 0

    def hypothesis_logits(self, x: Tensor, y: Tensor) -> Tensor:
        zx = self.x_adapter(x)
        zy = self.y_adapter(y)
        joint = torch.cat((zx, zy), dim=1)
        return torch.stack([head(joint).flatten() for head in self.hypotheses], dim=0)

    def probabilities(self, x: Tensor, y: Tensor) -> Tensor:
        with torch.no_grad():
            return (self.weights[:, None] * torch.sigmoid(self.hypothesis_logits(x, y))).sum(0)

    def spread(self, x: Tensor, y: Tensor) -> float:
        with torch.no_grad():
            zx = self.x_adapter(x)
            zy = self.y_adapter(y)
            return float(min(zx.std(dim=0).mean(), zy.std(dim=0).mean()))


def _collapse_penalty(z: Tensor, *, min_std: float) -> Tensor:
    return F.relu(min_std - z.std(dim=0, unbiased=False)).square().mean()


def _loss_per_hypothesis(model: FeedbackRelationSystem, x: Tensor, y: Tensor,
                         *, negative_shift: int = 1) -> Tensor:
    if len(y) < 2:
        raise ValueError("at least two real observed X/Y pairs required")
    positive = model.hypothesis_logits(x, y)
    negative = model.hypothesis_logits(x, torch.roll(y, negative_shift, dims=0))
    return (F.softplus(-positive).mean(dim=1) + F.softplus(negative).mean(dim=1)) / 2


@torch.no_grad()
def _empirical_hypothesis_weights(model: FeedbackRelationSystem, x: Tensor,
                                  y: Tensor) -> Tensor:
    """Calibration-score mixture weights, not a Bayesian posterior."""
    loss = _loss_per_hypothesis(model, x, y)
    logits = -loss * 3.0
    probabilities = logits.softmax(dim=0)
    return 0.90 * probabilities + 0.10 / len(probabilities)


@torch.no_grad()
def _validation_nll(model: FeedbackRelationSystem, x: Tensor, y: Tensor) -> float:
    p_true = model.probabilities(x, y).clamp(1e-6, 1-1e-6)
    p_false = model.probabilities(x, torch.roll(y, 1, dims=0)).clamp(1e-6, 1-1e-6)
    return float(((-p_true.log()).mean() - torch.log1p(-p_false).mean()) / 2)


def _accept_checkpoint(*, before_nll: float, after_nll: float,
                       before_spread: float, after_spread: float,
                       tolerance: float, min_spread: float) -> bool:
    return (np.isfinite(after_nll) and np.isfinite(after_spread)
            and after_nll <= before_nll + tolerance
            and after_spread >= min(min_spread, before_spread * .5))


def _freeze(model: FeedbackRelationSystem, *, update_heads: bool,
            update_adapters: bool) -> None:
    for p in model.hypotheses.parameters():
        p.requires_grad_(update_heads)
        p.grad = None
    for p in (*model.x_adapter.parameters(), *model.y_adapter.parameters()):
        p.requires_grad_(update_adapters)
        p.grad = None


@dataclass(frozen=True)
class FeedbackAudit:
    optimizer_steps: int
    kan_optimizer_steps: int
    adapter_optimizer_steps: int
    epochs: int
    accepted_epochs: int
    rejected_epochs: int
    initial_calibration_nll: float
    final_calibration_nll: float
    baseline_spread: float
    final_spread: float
    hypothesis_weights: tuple[float, ...]
    representation_version: int
    mechanism_version: int


def _fit_warmup(x: np.ndarray, y: np.ndarray, config: FeedbackConfig,
                *, seed: int) -> FeedbackRelationSystem:
    """Identical warmup for every ablation arm, using only initial observations."""
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("matched two-dimensional embedded observation pairs required")
    if x.shape[1] != config.feature_size or len(x) < 2:
        raise ValueError("embedding width or pair count mismatch")
    torch.set_num_threads(1)
    torch.manual_seed(seed * 1000 + 71)
    model = FeedbackRelationSystem(config.feature_size)
    x_t = torch.as_tensor(x.copy(), dtype=torch.float32)
    y_t = torch.as_tensor(y.copy(), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=config.warmup_lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed * 1000 + 91)
    for _ in range(config.warmup_steps):
        losses = _loss_per_hypothesis(model, x_t, y_t,
                                      negative_shift=int(rng.integers(1,len(x_t))))
        loss = losses.mean()
        zx, zy = model.x_adapter(x_t), model.y_adapter(y_t)
        loss = loss + config.variance_weight * (
            _collapse_penalty(zx, min_std=.10) +
            _collapse_penalty(zy, min_std=.10))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    opt.zero_grad(set_to_none=True)
    model.eval()
    return model


def _feedback_fit(warmup: FeedbackRelationSystem,
                  observed_x: np.ndarray, observed_y: np.ndarray,
                  calib_x: np.ndarray, calib_y: np.ndarray,
                  config: FeedbackConfig, mode: FeedbackMode,
                  *, seed: int) -> tuple[FeedbackRelationSystem, FeedbackAudit]:
    """Only observed train X/Y and disjoint calibration are accepted.

    Heldout samples are not an argument. Backbone is never a torch module here.
    Accepted checkpoints reweight hypotheses on current calibration because
    latent coordinates may have changed; obsolete weights are never reused.
    """
    if not isinstance(mode, FeedbackMode):
        raise TypeError("FeedbackMode required")
    model = copy.deepcopy(warmup)
    _freeze(model, update_heads=True, update_adapters=True)
    model.eval()
    x = torch.as_tensor(np.asarray(observed_x).copy(), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(observed_y).copy(), dtype=torch.float32)
    cx = torch.as_tensor(np.asarray(calib_x).copy(), dtype=torch.float32)
    cy = torch.as_tensor(np.asarray(calib_y).copy(), dtype=torch.float32)
    if x.shape != y.shape or cx.shape != cy.shape or x.ndim != 2 or cx.ndim != 2:
        raise ValueError("matched observed/calibration arrays required")
    if x.shape[1] != config.feature_size or cx.shape[1] != config.feature_size:
        raise ValueError("encoder dimension mismatch")
    if len(x) < 2 or len(cx) < 2:
        raise ValueError("at least two samples in each partition")
    model.weights.copy_(_empirical_hypothesis_weights(model, cx, cy))
    initial_nll = _validation_nll(model, cx, cy)
    initial_spread = model.spread(cx, cy)
    if mode is FeedbackMode.FROZEN:
        return model, FeedbackAudit(
            0, 0, 0, 0, 0, 0, initial_nll, initial_nll,
            initial_spread, initial_spread,
            tuple(float(t) for t in model.weights), 0, 0,
        )
    reference = copy.deepcopy(warmup)
    _freeze(reference, update_heads=False, update_adapters=False)
    torch.manual_seed(seed * 1000 + 111)
    rng = np.random.default_rng(seed * 1000 + 117)
    n_kan = n_adapter = accepted = rejected = 0
    epochs = 0
    step = 0
    while step < config.feedback_steps:
        current_steps = min(config.checkpoint_every, config.feedback_steps - step)
        snapshot = copy.deepcopy(model.state_dict())
        previous_version = (model.adapter_version, model.hypothesis_version)
        previous_nll = _validation_nll(model, cx, cy)
        before_spread = model.spread(cx, cy)
        # Whole-block rollback includes the optimizer state because a rejected
        # update must not alter the next block's momentum.
        for local_step in range(current_steps):
            absolute = step + local_step
            update_heads = (mode is FeedbackMode.KAN_ONLY or
                            (mode is FeedbackMode.ALTERNATING and absolute % 2 == 0))
            update_adapters = (mode is FeedbackMode.ADAPTER_ONLY or
                               (mode is FeedbackMode.ALTERNATING and absolute % 2 == 1))
            _freeze(model, update_heads=update_heads, update_adapters=update_adapters)
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters,
                lr=config.kan_lr if update_heads else config.adapter_lr,
                weight_decay=1e-5)
            shift = int(rng.integers(1,len(x)))
            losses = _loss_per_hypothesis(model, x, y, negative_shift=shift)
            loss = (model.weights.detach() * losses).sum()
            if update_adapters:
                zx, zy = model.x_adapter(x), model.y_adapter(y)
                with torch.no_grad():
                    initial_zx = reference.x_adapter(x)
                    initial_zy = reference.y_adapter(y)
                preserve = (F.mse_loss(zx, initial_zx) +
                            F.mse_loss(zy, initial_zy)) / 2
                # Preserve relational geometry, not only pointwise embeddings.
                diff_zx = (zx - torch.roll(zx, 1, dims=0)).square().sum(1)
                diff_zy = (zy - torch.roll(zy, 1, dims=0)).square().sum(1)
                diff_ref_x = (initial_zx - torch.roll(initial_zx, 1, dims=0)).square().sum(1)
                diff_ref_y = (initial_zy - torch.roll(initial_zy, 1, dims=0)).square().sum(1)
                geometry = (F.mse_loss(diff_zx, diff_ref_x) +
                            F.mse_loss(diff_zy, diff_ref_y)) / 2
                loss = (loss + config.preservation_weight * preserve
                        + config.geometry_weight * geometry
                        + config.variance_weight * (
                            _collapse_penalty(zx, min_std=.10)
                            + _collapse_penalty(zy, min_std=.10)))
                n_adapter += 1
            else:
                n_kan += 1
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
        step += current_steps
        epochs += 1
        model.weights.copy_(_empirical_hypothesis_weights(model, cx, cy))
        after_nll = _validation_nll(model, cx, cy)
        after_spread = model.spread(cx, cy)
        good = _accept_checkpoint(
            before_nll=previous_nll, after_nll=after_nll,
            before_spread=before_spread, after_spread=after_spread,
            tolerance=config.calibration_tolerance,
            min_spread=config.min_spread,
        )
        if good:
            accepted += 1
            if mode is FeedbackMode.KAN_ONLY:
                model.hypothesis_version += 1
            elif mode is FeedbackMode.ADAPTER_ONLY:
                model.adapter_version += 1
            else:
                model.adapter_version += 1
                model.hypothesis_version += 1
        else:
            rejected += 1
            model.load_state_dict(snapshot)
            model.adapter_version, model.hypothesis_version = previous_version
    _freeze(model, update_heads=False, update_adapters=False)
    model.eval()
    return model, FeedbackAudit(
        config.feedback_steps, n_kan, n_adapter, epochs, accepted, rejected,
        initial_nll, _validation_nll(model, cx, cy), initial_spread,
        model.spread(cx, cy),
        tuple(float(t) for t in model.weights),
        model.adapter_version, model.hypothesis_version,
    )


@dataclass(frozen=True)
class FeedbackScore:
    law: str
    seed: int
    mode: FeedbackMode
    domain: str
    auroc: float
    paired_accuracy: float
    brier: float
    init_auc: float
    bootstrap: int
    adaptation: int
    calibration: int
    heldout: int
    identity_overlap: int
    frozen_encoder_id: str
    hypothesis_weights: tuple[float, ...]
    adapter_version: int
    hypothesis_version: int
    optimizer_steps: int
    kan_optimizer_steps: int
    adapter_optimizer_steps: int
    accepted_epochs: int
    rejected_epochs: int
    calibration_nll: float
    min_latent_spread: float


def _perturbed_y(samples: tuple[TextRelationSample, ...]) -> tuple[TextRelationSample, ...]:
    """Benchmark-only numeric counterfactual for the exact same X.

    It is NEVER passed to any training or calibration method.
    """
    from dataclasses import replace
    outputs = []
    for sample in samples:
        doc = json.loads(sample.y_text)
        original = float(doc["after"]["measurement"])
        shifted = original + (0.70 if original <= 0 else -0.70)
        doc["after"]["measurement"] = round(shifted, 4)
        doc["after"]["description"] = "a high signal" if shifted > 0 else "a low signal"
        outputs.append(replace(sample, y_text=json.dumps(
            doc, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )))
    return tuple(outputs)


def _encoded_samples(encoder: TextEmbeddingPort,
                     rows: tuple[TextRelationSample, ...]) -> tuple[np.ndarray, np.ndarray]:
    text = tuple(s.x_text for s in rows) + tuple(s.y_text for s in rows)
    vectors = np.asarray(encoder.encode_many(text), dtype=np.float32)
    if vectors.shape != (len(text), encoder.dimension) or not np.isfinite(vectors).all():
        raise ValueError("frozen encoder emitted wrong shapes or nonfinite embeddings")
    return vectors[:len(rows)].copy(), vectors[len(rows):].copy()


@torch.no_grad()
def _pair_scores(model: FeedbackRelationSystem,
                 encoder: TextEmbeddingPort,
                 rows: tuple[TextRelationSample, ...],
                 negative: tuple[TextRelationSample, ...] | None = None) -> tuple[float, float, float]:
    if negative is not None and len(negative) != len(rows):
        raise ValueError("counterfactual negatives must match positives")
    x, y = _encoded_samples(encoder, rows)
    if negative is None:
        # Evaluation-only pairs chosen by observed Y differences, NOT by model.
        j = _hard_negatives(rows)
        negative_y = y[j]
    else:
        xx, negative_y = _encoded_samples(encoder, negative)
        if not np.array_equal(x, xx):
            raise ValueError("the counterfactual changed pre-action X")
    tx = torch.tensor(x)
    p_yes = model.probabilities(tx, torch.tensor(y)).numpy()
    p_no = model.probabilities(tx, torch.tensor(negative_y)).numpy()
    predicted = np.r_[p_yes, p_no]
    truth = np.r_[np.ones(len(rows)), np.zeros(len(rows))]
    return (
        _auc(predicted, truth),
        float(np.mean(p_yes > p_no)),
        float(np.mean((predicted-truth)**2)),
    )


def _feedback_benchmark(*, law: FixtureLaw, seed: int,
                        config: FeedbackConfig, encoder: TextEmbeddingPort,
                        frozen_encoder_id: str = "provided-encoder") -> tuple[FeedbackScore, ...]:
    """All four modes share warmup weights, frozen embeddings and data split."""
    if not isinstance(law, FixtureLaw):
        raise TypeError("typed fixture required")
    if not isinstance(encoder, TextEmbeddingPort):
        raise TypeError("frozen embedding port required")
    if encoder.dimension != config.feature_size:
        raise ValueError("frozen embedding width does not match configuration")
    total = config.bootstrap + config.adaptation + config.calibration
    train = _simulation(law, seed * 101 + 9, total, heldout=False)
    test = _simulation(law, seed * 101 + 109, config.heldout, heldout=True)
    ids_train = {json.loads(s.x_text)["before"]["object"]["identity"] for s in train}
    ids_test = {json.loads(s.x_text)["before"]["object"]["identity"] for s in test}
    overlap = len(ids_train & ids_test)
    if overlap:
        raise ValueError("training and evaluation entity identities overlap")
    x_train, y_train = _encoded_samples(encoder, train)
    # No train/calibration/heldout mixing: recent observations are adaptation
    # only; calibration is a disjoint tail, test is created separately.
    train_cut = config.bootstrap + config.adaptation
    warm = _fit_warmup(x_train[:config.bootstrap], y_train[:config.bootstrap], config, seed=seed)
    calibration_x = x_train[train_cut:]
    calibration_y = y_train[train_cut:]
    output: list[FeedbackScore] = []
    evaluations = [("heldout", test, None),
                   ("paraphrase", tuple(_paraphrase(s) for s in test), None)]
    if law in (FixtureLaw.CATEGORICAL, FixtureLaw.INTERACTION):
        evaluations.append(("counterfactual_mode", test,
                            _opposite_mode_outcomes(test, law)))
    if law in (FixtureLaw.NUMERIC, FixtureLaw.INTERACTION):
        evaluations.append(("numeric_perturbation", test, _perturbed_y(test)))
    initial_scores = {
        domain: _pair_scores(warm, encoder, samples, contrast)[0]
        for domain, samples, contrast in evaluations
    }
    for mode in FeedbackMode:
        model, audit = _feedback_fit(
            warm, x_train[:train_cut], y_train[:train_cut],
            calibration_x, calibration_y, config, mode, seed=seed,
        )
        for domain, rows, contrast in evaluations:
            auc, paired, brier = _pair_scores(model, encoder, rows, contrast)
            output.append(FeedbackScore(
                law=law.value, seed=seed, mode=mode, domain=domain,
                auroc=auc, paired_accuracy=paired, brier=brier,
                init_auc=initial_scores[domain],
                bootstrap=config.bootstrap, adaptation=config.adaptation,
                calibration=config.calibration, heldout=config.heldout,
                identity_overlap=overlap, frozen_encoder_id=frozen_encoder_id,
                hypothesis_weights=audit.hypothesis_weights,
                adapter_version=audit.representation_version,
                hypothesis_version=audit.mechanism_version,
                optimizer_steps=audit.optimizer_steps,
                kan_optimizer_steps=audit.kan_optimizer_steps,
                adapter_optimizer_steps=audit.adapter_optimizer_steps,
                accepted_epochs=audit.accepted_epochs,
                rejected_epochs=audit.rejected_epochs,
                calibration_nll=audit.final_calibration_nll,
                min_latent_spread=audit.final_spread,
            ))
    return tuple(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--laws", nargs="+", choices=[l.value for l in FixtureLaw],
                        default=[law.value for law in FixtureLaw])
    parser.add_argument("--bootstrap", type=int, default=48)
    parser.add_argument("--adaptation", type=int, default=32)
    parser.add_argument("--calibration", type=int, default=16)
    parser.add_argument("--heldout", type=int, default=32)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--feedback-steps", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    config = FeedbackConfig(
        bootstrap=args.bootstrap, adaptation=args.adaptation,
        calibration=args.calibration, heldout=args.heldout,
        feature_size=args.dimension, warmup_steps=args.warmup_steps,
        feedback_steps=args.feedback_steps,
    )
    encoder = SentenceTransformerTextEncoder(
        args.model, dimension=args.dimension, revision=args.revision,
        device=args.device, max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    )
    rows = [asdict(result) for law in args.laws for seed in args.seeds
            for result in _feedback_benchmark(
                law=FixtureLaw(law), seed=seed, config=config,
                encoder=encoder, frozen_encoder_id=f"{args.model}@{args.revision or 'unpinned'}",
            )]
    payload = {
        "model": args.model, "revision": args.revision,
        "config": asdict(config), "results": rows,
        "limitations": [
            "KAN hypotheses are weighted by empirical train-only calibration scores, not a causal posterior.",
            "Embeddings are frozen; feedback updates only small X/Y adapters.",
            "All fixtures and evaluation counterfactuals are synthetic; no universal encoder claim.",
            "Intervention selection and symbolic law recovery are not tested by this ablation.",
        ],
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n")
    summary = {}
    for domain in sorted({r["domain"] for r in rows}):
        summary[domain] = {}
        for mode in FeedbackMode:
            rs = [r for r in rows if r["domain"] == domain and r["mode"] == mode.value]
            summary[domain][mode.value] = {
                "n": len(rs),
                "mean_auroc": round(float(np.mean([r["auroc"] for r in rs])), 5) if rs else None,
                "mean_brier": round(float(np.mean([r["brier"] for r in rs])), 5) if rs else None,
            }
    print(json.dumps({"total": len(rows), "by_domain": summary}, indent=2))


if __name__ == "__main__":
    main()
