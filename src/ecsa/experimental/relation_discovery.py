"""Experimental dual-encoder implicit-relation discovery.

Architecture: X -> E_x, Y -> E_y; KAN(E_x(X), E_y(Y)) measures pair
compatibility. This is not an X->Y regressor or proof of causality.
Optional dependency: torch and numpy. No DiscoveryWorld semantics.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Law(StrEnum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    IMPLICIT = "implicit"


class HeadKind(StrEnum):
    MLP = "mlp"
    SPLINE_KAN = "spline_kan"


@dataclass(frozen=True)
class ExperimentConfig:
    train_pairs: int = 96
    test_pairs: int = 192
    steps: int = 180
    active_queries: int = 12
    active_steps: int = 100
    ensemble_members: int = 3
    lr: float = 0.012
    heldout_radius: float = 1.05

    def __post_init__(self) -> None:
        for name, number in (
            ("train_pairs", self.train_pairs), ("test_pairs", self.test_pairs),
            ("steps", self.steps), ("active_queries", self.active_queries),
            ("active_steps", self.active_steps), ("ensemble_members", self.ensemble_members),
        ):
            if type(number) is not int or number < (2 if name == "ensemble_members" else 1):
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 < self.lr < 0.2 or not 0.8 <= self.heldout_radius <= 1.5:
            raise ValueError("invalid experiment configuration")


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray
    y: np.ndarray
    x_view: np.ndarray
    y_view: np.ndarray


def _physical_y(law: Law, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u, v = x[:, 0], x[:, 1]
    if law is Law.ADDITIVE:
        return (0.7*u - 0.45*v + 0.20*np.sin(2*u)).astype(np.float32)
    if law is Law.MULTIPLICATIVE:
        return (u*v + 0.13*u).astype(np.float32)
    if law is Law.IMPLICIT:
        sign = rng.choice(np.array([-1., 1.]), size=len(x))
        return (sign*np.sqrt(0.35 + 0.5*u*u + 0.3*v*v)).astype(np.float32)
    raise ValueError(law)


def _x_view(x: np.ndarray) -> np.ndarray:
    u, v = x[:, 0], x[:, 1]
    return np.stack((u+0.2*v, v-0.3*u, np.tanh(u+0.4*v)), axis=1).astype(np.float32)


def _y_view(y: np.ndarray) -> np.ndarray:
    return np.stack((y, np.tanh(0.9*y) + 0.07*y*y), axis=1).astype(np.float32)


def make_dataset(law: Law, n: int, seed: int, *, radius: float, noise: float = 0.015) -> Dataset:
    if n < 4:
        raise ValueError("at least four observations required")
    rng = np.random.default_rng(seed)
    x = rng.uniform(-radius, radius, (n, 2)).astype(np.float32)
    y = _physical_y(law, x, rng)
    observed = y + rng.normal(0, noise, n).astype(np.float32)
    return Dataset(x, y, _x_view(x), _y_view(observed))


def oracle_compatibility(law: Law, x: np.ndarray, y: np.ndarray, tolerance: float = .11) -> np.ndarray:
    """Benchmark-only membership oracle. Never called in passive training."""
    u, v = x[:, 0], x[:, 1]
    if law is Law.ADDITIVE:
        residual = np.abs(y - (0.7*u-0.45*v+0.2*np.sin(2*u)))
    elif law is Law.MULTIPLICATIVE:
        residual = np.abs(y - (u*v+0.13*u))
    elif law is Law.IMPLICIT:
        residual = np.abs(y*y - (0.35+0.5*u*u+0.3*v*v))
    else:
        raise ValueError(law)
    return (residual <= tolerance).astype(np.float32)


class SeparateEncoders(nn.Module):
    def __init__(self, latent: int = 2) -> None:
        super().__init__()
        self.x = nn.Sequential(nn.Linear(3, 12), nn.Tanh(), nn.Linear(12, latent), nn.Tanh())
        self.y = nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, latent), nn.Tanh())

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        return self.x(x), self.y(y)


class SplineEdgeLayer(nn.Module):
    """A first-degree B-spline on every (input, output) edge.

    This is a lightweight fixed-grid KAN, not full pykan/MultKAN.
    """
    def __init__(self, inputs: int, outputs: int, knots: int = 7) -> None:
        super().__init__()
        self.register_buffer("centers", torch.linspace(-1.5, 1.5, knots))
        self.spacing = 3. / (knots-1)
        self.coefficients = nn.Parameter(torch.randn(inputs, outputs, knots)*.07)
        self.residual = nn.Parameter(torch.randn(inputs, outputs)*.10)
        self.bias = nn.Parameter(torch.zeros(outputs))

    def forward(self, x: Tensor) -> Tensor:
        basis = F.relu(1. - torch.abs(x.unsqueeze(-1)-self.centers)/self.spacing)
        return (
            torch.einsum("bik,iok->bo", basis, self.coefficients)
            + x @ self.residual + self.bias
        )


class SplineKANHead(nn.Module):
    def __init__(self, *, knots: int = 7, hidden: int = 9) -> None:
        super().__init__()
        self.first = SplineEdgeLayer(4, hidden, knots)
        self.last = SplineEdgeLayer(hidden, 1, knots)

    def forward(self, xy: Tensor) -> Tensor:
        return self.last(torch.tanh(self.first(xy)))


class MLPHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1),
        )

    def forward(self, xy: Tensor) -> Tensor:
        return self.layers(xy)


class RelationModel(nn.Module):
    def __init__(self, kind: HeadKind, *, kan_knots: int = 7, kan_hidden: int = 9) -> None:
        super().__init__()
        if not isinstance(kind, HeadKind):
            raise TypeError("typed relation-head kind required")
        self.encoder = SeparateEncoders()
        self.relation = (
            MLPHead() if kind is HeadKind.MLP
            else SplineKANHead(knots=kan_knots, hidden=kan_hidden)
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        zx, zy = self.encoder(x, y)
        return self.relation(torch.cat((zx, zy), dim=1)).flatten()

    def encoder_spread(self, x: Tensor, y: Tensor) -> tuple[float, float]:
        with torch.no_grad():
            zx, zy = self.encoder(x, y)
            return float(zx.std(dim=0).mean()), float(zy.std(dim=0).mean())


def parameters(model: nn.Module) -> int:
    return sum(t.numel() for t in model.parameters())


def _tensor(array: np.ndarray) -> Tensor:
    return torch.tensor(array, dtype=torch.float32)


def fit(
    model: RelationModel,
    pairs: Dataset,
    *,
    seed: int,
    steps: int,
    lr: float,
    queried: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    """Observed positive pairs vs permuted Y; optional separately labelled queries.

    No oracle check on shuffled training negatives (some are genuinely valid).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    px, py = _tensor(pairs.x_view), _tensor(pairs.y_view)
    if queried is not None:
        qx, qy, qlabels = (_tensor(v) for v in queried)
    model.train()
    for _ in range(steps):
        shift = int(rng.integers(1, len(px)))
        xs = torch.cat((px, px), dim=0)
        ys = torch.cat((py, torch.roll(py, shifts=shift, dims=0)), dim=0)
        labels = torch.cat((torch.ones(len(px)), torch.zeros(len(px))))
        if queried is not None and len(qx):
            indices = torch.arange(len(qx)).repeat(max(1, len(px)//max(1, len(qx))))
            xs = torch.cat((xs, qx[indices]), dim=0)
            ys = torch.cat((ys, qy[indices]), dim=0)
            labels = torch.cat((labels, qlabels[indices]), dim=0)
        logits = model(xs, ys)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        zx, zy = model.encoder(px, py)
        spread_loss = (
            F.relu(.10 - zx.std(dim=0)).square().mean()
            + F.relu(.10 - zy.std(dim=0)).square().mean()
        )
        optimizer.zero_grad()
        (loss + .02*spread_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()


@torch.no_grad()
def probability(model: RelationModel, xv: np.ndarray, yv: np.ndarray, *, batch: int = 4096) -> np.ndarray:
    chunks = []
    for i in range(0, len(xv), batch):
        scores = model(_tensor(xv[i:i+batch]), _tensor(yv[i:i+batch]))
        chunks.append(torch.sigmoid(scores).cpu().numpy())
    return np.concatenate(chunks)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive, negative = scores[labels == 1], scores[labels == 0]
    return float(
        (positive[:, None] > negative[None, :]).mean()
        + .5*(positive[:, None] == negative[None, :]).mean()
    )


def evaluate(
    model: RelationModel | tuple[RelationModel, ...],
    law: Law,
    seed: int,
    *,
    n: int,
    radius: float,
) -> dict[str, float]:
    test = make_dataset(law, n, seed, radius=radius)
    rng = np.random.default_rng(seed + 2)
    negative_y = []
    for i in range(n):
        replacement = None
        for _ in range(80):
            k = int(rng.integers(0, n))
            candidate = test.y[k]
            # Only test negatives are verified by the hidden-law oracle.
            if not oracle_compatibility(
                law, test.x[i:i+1], np.array([candidate]), tolerance=.14
            )[0]:
                replacement = candidate
                break
        if replacement is None:
            replacement = test.y[i] + np.sign(test.y[i]+.01)*.65
        negative_y.append(replacement)
    xv = np.concatenate((test.x_view, test.x_view))
    yv = np.concatenate((test.y_view, _y_view(np.asarray(negative_y, dtype=np.float32))))
    labels = np.concatenate((np.ones(n), np.zeros(n)))
    models = (model,) if isinstance(model, RelationModel) else model
    p = np.mean([probability(m, xv, yv) for m in models], axis=0)
    return {
        "auc": _auc(p, labels),
        "brier": float(np.mean((p-labels)**2)),
        "balanced_accuracy": float(np.mean((p >= .5) == labels)),
    }


def _candidate_queries(
    initial: Dataset, *, seed: int, size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Proposals use ONLY already observed outcomes and perturbed X values."""
    rng = np.random.default_rng(seed)
    anchors = rng.integers(0, len(initial.x), size=size)
    x = initial.x[anchors].copy()
    x += rng.normal(0, .12, (size, 2)).astype(np.float32)
    x = np.clip(x, -1.15, 1.15)
    y = initial.y[anchors].copy()
    replacement = rng.random(size) < .5
    y[replacement] = initial.y[rng.integers(0, len(initial.y), size=int(replacement.sum()))]
    y += rng.normal(0, .09, size).astype(np.float32)
    return x, y, _x_view(x), _y_view(y)


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1-1e-7)
    return -p*np.log2(p) - (1-p)*np.log2(1-p)


def mutual_information(
    models: tuple[RelationModel, ...], xv: np.ndarray, yv: np.ndarray,
) -> np.ndarray:
    """Uniform-model binary EIG, vectorized counterpart to ScienceKernel.

    Batch proposals use an unusually strong membership-query oracle.
    Ensemble members are approximate hypotheses, not calibrated law posteriors.
    """
    probabilities = np.stack([probability(m, xv, yv) for m in models])
    return np.maximum(
        0., _binary_entropy(probabilities.mean(axis=0))
        - _binary_entropy(probabilities).mean(axis=0)
    )


@dataclass(frozen=True)
class RunResult:
    law: str
    seed: int
    model: str
    params: int
    auc: float
    brier: float
    balanced_accuracy: float
    encoder_x_std: float
    encoder_y_std: float
    training_seconds: float
    queries: int = 0
    query_positive_rate: float = 0.


def _result(
    law: Law, seed: int, label: str,
    model: RelationModel | tuple[RelationModel, ...],
    train: Dataset, score: dict[str, float], elapsed: float,
    *, queries: int = 0, positive_rate: float = 0.,
) -> RunResult:
    first = model[0] if isinstance(model, tuple) else model
    std_x, std_y = first.encoder_spread(_tensor(train.x_view), _tensor(train.y_view))
    return RunResult(
        law.value, seed, label, parameters(first),
        score["auc"], score["brier"], score["balanced_accuracy"],
        std_x, std_y, elapsed, queries, positive_rate,
    )


def experiment(
    law: Law, seed: int, config: ExperimentConfig, *, active: bool = True,
) -> tuple[RunResult, ...]:
    torch.set_num_threads(1)
    train = make_dataset(law, config.train_pairs, seed*100+3, radius=.9)
    rows: list[RunResult] = []
    for kind in (HeadKind.MLP, HeadKind.SPLINE_KAN):
        torch.manual_seed(seed*1000+7)
        model = RelationModel(kind)
        start = time.perf_counter()
        fit(model, train, seed=seed*1000+23, steps=config.steps, lr=config.lr)
        duration = time.perf_counter() - start
        score = evaluate(
            model, law, seed*100+500, n=config.test_pairs, radius=config.heldout_radius,
        )
        rows.append(_result(law, seed, kind.value, model, train, score, duration))

    hypotheses: list[RelationModel] = []
    variants = ((5, 12), (7, 9), (9, 7))
    start = time.perf_counter()
    for index in range(config.ensemble_members):
        torch.manual_seed(seed*1000+7+index)
        knots, hidden = variants[index % len(variants)]
        model = RelationModel(HeadKind.SPLINE_KAN, kan_knots=knots, kan_hidden=hidden)
        fit(
            model, train, seed=seed*1000+23+index,
            steps=config.steps, lr=config.lr,
        )
        hypotheses.append(model)
    ensemble = tuple(hypotheses)
    duration = time.perf_counter() - start
    scores = evaluate(
        ensemble, law, seed*100+500, n=config.test_pairs, radius=config.heldout_radius,
    )
    rows.append(_result(law, seed, "spline_kan_ensemble", ensemble, train, scores, duration))

    if active:
        qx, qy, qxv, qyv = _candidate_queries(train, seed=seed*1000+901)
        gains = mutual_information(ensemble, qxv, qyv)
        count = min(config.active_queries, len(gains))
        choices = np.argsort(-gains, kind="stable")[:count]
        random = np.random.default_rng(seed*1000+776).choice(len(gains), size=count, replace=False)
        for label, selected in (
            ("spline_kan_eig_batch", choices),
            ("spline_kan_random_batch", random),
        ):
            queried_labels = oracle_compatibility(law, qx[selected], qy[selected])
            queried = (qxv[selected], qyv[selected], queried_labels)
            trained = []
            start = time.perf_counter()
            for index in range(config.ensemble_members):
                torch.manual_seed(seed*1000+7+index)
                knots, hidden = variants[index % len(variants)]
                model = RelationModel(HeadKind.SPLINE_KAN, kan_knots=knots, kan_hidden=hidden)
                fit(
                    model, train, seed=seed*1000+23+index,
                    steps=config.steps+config.active_steps,
                    lr=config.lr, queried=queried,
                )
                trained.append(model)
            trained = tuple(trained)
            elapsed = time.perf_counter()-start
            score = evaluate(
                trained, law, seed*100+500, n=config.test_pairs,
                radius=config.heldout_radius,
            )
            rows.append(_result(
                law, seed, label, trained, train, score, elapsed,
                queries=count, positive_rate=float(queried_labels.mean()),
            ))
    return tuple(rows)


def run_benchmark(
    *, seeds: tuple[int, ...] = (0, 1),
    config: ExperimentConfig = ExperimentConfig(),
    active: bool = True,
) -> tuple[RunResult, ...]:
    return tuple(
        row for law in Law for seed in seeds
        for row in experiment(law, seed, config, active=active)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--train-pairs", type=int, default=96)
    parser.add_argument("--test-pairs", type=int, default=192)
    parser.add_argument("--active-queries", type=int, default=12)
    parser.add_argument("--active-steps", type=int, default=100)
    parser.add_argument("--no-active", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    config = ExperimentConfig(
        steps=args.steps, train_pairs=args.train_pairs,
        test_pairs=args.test_pairs, active_queries=args.active_queries,
        active_steps=args.active_steps,
    )
    results = run_benchmark(
        seeds=tuple(args.seeds), config=config, active=not args.no_active,
    )
    output = {
        "config": asdict(config),
        "seeds": args.seeds,
        "results": [asdict(row) for row in results],
        "warning": "Membership oracle and synthetic laws; not causal discovery.",
    }
    body = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        from pathlib import Path
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
