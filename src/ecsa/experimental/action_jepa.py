"""Small action-conditioned JEPA learner with EMA targets and anti-collapse loss.

Learner never imports simulator mechanism or privileged simulator state.
JEPA-inspired prototype, not LeWorldModel/SIGReg or V-JEPA 2.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .jepa_world import Action, TransitionDataset, _ACTIONS

@dataclass(frozen=True)
class JepaConfig:
    latent_dim: int = 8
    hidden_dim: int = 64
    batch_size: int = 128
    steps: int = 400
    lr: float = .002
    ema_decay: float = .985
    variance_weight: float = .2
    covariance_weight: float = .03
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("latent_dim", "hidden_dim", "batch_size", "steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.ema_decay < 1:
            raise ValueError("ema_decay must be between 0 and 1")
        if self.lr <= 0 or not isfinite(self.lr):
            raise ValueError("lr must be finite and positive")
        if not all(isfinite(value) and value >= 0 for value in (
            self.variance_weight, self.covariance_weight,
        )):
            raise ValueError("regularization weights must be nonnegative")


class ActionJEPA(nn.Module):
    """Small online encoder, EMA target encoder, action-conditioned predictor."""

    def __init__(self, observation_dim: int, config: JepaConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.target_encoder = deepcopy(self.encoder)
        self.target_encoder.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(config.latent_dim + len(_ACTIONS), config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

    def predict_tensor(self, encoded: Tensor, actions: Tensor) -> Tensor:
        if encoded.ndim != 2 or actions.shape != (len(encoded), len(_ACTIONS)):
            raise ValueError("expected [N, latent] and [N, actions]")
        return encoded + self.predictor(torch.cat((encoded, actions), dim=1))

    def encode(self, observations: np.ndarray) -> np.ndarray:
        data = np.asarray(observations, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("encode requires observation matrix [N,D]")
        device = next(self.encoder.parameters()).device
        with torch.inference_mode():
            output = self.encoder(torch.as_tensor(data, device=device))
        return output.float().cpu().numpy()

    @torch.no_grad()
    def update_target(self) -> None:
        decay = self.config.ema_decay
        for target, online in zip(self.target_encoder.parameters(),self.encoder.parameters()):
            target.lerp_(online, 1.0 - decay)


@dataclass(frozen=True)
class TrainingReport:
    start_loss: float
    end_loss: float
    train_steps: int
    shuffled_actions: bool


def _decorrelation_penalty(z: Tensor) -> Tensor:
    centered = z - z.mean(dim=0)
    cov = (centered.T @ centered) / max(len(z) - 1, 1)
    off_diagonal = cov - torch.diag(torch.diag(cov))
    return off_diagonal.square().sum() / z.shape[1]


def train_jepa(
    data: TransitionDataset, config: JepaConfig,
    *, device: str = "cpu", shuffle_actions: bool = False,
) -> tuple[ActionJEPA, TrainingReport]:
    """Online encoder and predictor are trained without simulator state labels.

    Shuffled-action control preserves observations/targets and reassigns only
    train-time public action tokens; heldout evaluation remains unchanged.
    """
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("requested training device is unavailable")
    if len(data.before) < 2:
        raise ValueError("need at least two transitions")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = ActionJEPA(data.before.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=config.lr, weight_decay=1e-4,
    )
    inputs = torch.as_tensor(data.before, device=device)
    targets = torch.as_tensor(data.after, device=device)
    actions = np.array(data.actions, copy=True)
    if shuffle_actions:
        actions = actions[rng.permutation(len(actions))]
    act = torch.as_tensor(actions, device=device)
    model.train()
    start_loss = float("nan")
    end_loss = float("nan")
    for step in range(config.steps):
        indices = torch.as_tensor(
            rng.integers(0, len(data.before), config.batch_size), device=device,
        )
        z = model.encoder(inputs[indices])
        predicted = model.predict_tensor(z, act[indices])
        with torch.no_grad():
            target = model.target_encoder(targets[indices])
        # Online variance/covariance controls; target has no gradient.
        pred_loss = F.mse_loss(predicted, target)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        variance = F.relu(.6 - std).mean()
        loss = (
            pred_loss + config.variance_weight * variance +
            config.covariance_weight * _decorrelation_penalty(z)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.update_target()
        if step == 0:
            start_loss = float(loss.detach())
        end_loss = float(loss.detach())
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    return model, TrainingReport(start_loss, end_loss, config.steps, shuffle_actions)


@dataclass(frozen=True)
class PredictionEvaluation:
    prediction_mse: float
    action_choice_accuracy: float
    mean_latent_std: float
    effective_rank: float
    heldout_count: int


def evaluate_prediction(model: ActionJEPA, heldout: TransitionDataset) -> PredictionEvaluation:
    """Heldout latent regression and correct-vs-alternative action ranking."""
    device = next(model.parameters()).device
    with torch.inference_mode():
        before = torch.as_tensor(heldout.before, device=device)
        after = torch.as_tensor(heldout.after, device=device)
        actions = torch.as_tensor(heldout.actions, device=device)
        z = model.encoder(before)
        target = model.encoder(after)
        candidate = model.predict_tensor(z, actions)
        mse = float(F.mse_loss(candidate, target))
        correct_d = (candidate - target).square().sum(dim=1)
        comparisons = []
        for shift in (1, 2):
            wrong_actions = actions.roll(shifts=shift, dims=1)
            wrong = model.predict_tensor(z, wrong_actions)
            wrong_d = (wrong - target).square().sum(dim=1)
            comparisons.append((correct_d < wrong_d).float() +
                               .5 * (correct_d == wrong_d).float())
        accuracy = float(torch.stack(comparisons).mean())
    embeddings = model.encode(heldout.before)
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular.astype(np.float64) ** 2
    rank = float(energy.sum() ** 2 / max(np.square(energy).sum(), 1e-12))
    return PredictionEvaluation(
        prediction_mse=mse,
        action_choice_accuracy=accuracy,
        mean_latent_std=float(embeddings.std(axis=0).mean()),
        effective_rank=rank,
        heldout_count=len(heldout.before),
    )



class RawObservationEncoder:
    def encode(self, observations: np.ndarray) -> np.ndarray:
        return np.asarray(observations, dtype=np.float32)


class RandomEncoder:
    """Frozen, untrained encoder with the same width as JEPA."""
    def __init__(self, observation_dim: int, config: JepaConfig) -> None:
        torch.manual_seed(config.seed)
        self._model = ActionJEPA(observation_dim, config).eval()

    def encode(self, observations: np.ndarray) -> np.ndarray:
        return self._model.encode(observations)


