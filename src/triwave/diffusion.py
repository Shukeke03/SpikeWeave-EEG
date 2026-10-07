"""Cosine DDPM v-prediction and a separately trained linear-flow comparison."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

Predictor = Callable[[Tensor, Tensor], Tensor]


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000):
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be at least two")
        self.timesteps = timesteps
        grid = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        curve = torch.cos(((grid / timesteps + 0.008) / 1.008) * math.pi / 2).square()
        curve = curve / curve[0]
        betas = (1 - curve[1:] / curve[:-1]).clamp(0, 0.999)
        cumulative = torch.cumprod(1 - betas, dim=0).float()
        self.register_buffer("alpha", cumulative.sqrt())
        self.register_buffer("sigma", (1 - cumulative).sqrt())

    def factors(self, indices: Tensor, like: Tensor) -> tuple[Tensor, Tensor]:
        if indices.dtype != torch.long or indices.shape != (like.shape[0],):
            raise ValueError("diffusion indices must be int64 [batch]")
        if (indices < 0).any() or (indices >= self.timesteps).any():
            raise ValueError("diffusion index outside schedule")
        shape = (like.shape[0],) + (1,) * (like.ndim - 1)
        return self.alpha[indices].reshape(shape), self.sigma[indices].reshape(shape)

    def noise(self, clean: Tensor, epsilon: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
        alpha, sigma = self.factors(indices, clean)
        return alpha * clean + sigma * epsilon, alpha * epsilon - sigma * clean

    def clean_from_v(self, noisy: Tensor, velocity: Tensor, indices: Tensor) -> Tensor:
        alpha, sigma = self.factors(indices, noisy)
        return alpha * noisy - sigma * velocity

    def normalized_time(self, indices: Tensor) -> Tensor:
        return indices.float() / (self.timesteps - 1)

    @torch.no_grad()
    def sample(self, predict: Predictor, initial_noise: Tensor, steps: int = 30) -> Tensor:
        """Deterministic DDIM, including the final noiseless state (eta=0)."""
        if not 1 <= steps <= self.timesteps:
            raise ValueError("sampling steps must lie in [1, timesteps]")
        indices = torch.linspace(self.timesteps - 1, 0, steps, device=initial_noise.device)
        indices = indices.round().long()
        state = initial_noise.clone()
        for position, index in enumerate(indices):
            batch_indices = index.expand(state.shape[0])
            velocity = predict(state, self.normalized_time(batch_indices))
            alpha, sigma = self.factors(batch_indices, state)
            clean = alpha * state - sigma * velocity
            epsilon = sigma * state + alpha * velocity
            if position + 1 == steps:
                state = clean
            else:
                next_indices = indices[position + 1].expand(state.shape[0])
                next_alpha, next_sigma = self.factors(next_indices, state)
                state = next_alpha * clean + next_sigma * epsilon
        return state


def flow_pair(clean: Tensor, epsilon: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
    """t=0 is data, t=1 is noise; target is dx/dt = noise - data."""
    if time.shape != (clean.shape[0],) or ((time < 0) | (time > 1)).any():
        raise ValueError("flow time must be [batch] in [0,1]")
    scale = time.reshape((-1,) + (1,) * (clean.ndim - 1))
    return (1 - scale) * clean + scale * epsilon, epsilon - clean


@torch.no_grad()
def sample_flow(predict: Predictor, initial_noise: Tensor, steps: int = 30) -> Tensor:
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    state = initial_noise.clone()
    for step in range(steps):
        time = torch.full((state.shape[0],), 1 - step / steps, device=state.device)
        state = state - predict(state, time) / steps
    return state
