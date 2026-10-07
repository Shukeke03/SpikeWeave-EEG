"""VP clean prediction and audited-NFE samplers; legacy diffusion comparisons."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

Predictor = Callable[[Tensor, Tensor], Tensor]


def vp_factors(time: Tensor, like: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Continuous VP schedule; diffusion time is distinct from EEG sample time."""
    if time.ndim != 1 or not time.is_floating_point() or not torch.isfinite(time).all():
        raise ValueError("VP time must be finite floating point [batch]")
    if ((time < 0) | (time > 1)).any():
        raise ValueError("VP time must lie in [0,1]")
    if like is not None and time.shape != (like.shape[0],):
        raise ValueError("VP time must match the state batch")
    work = time if time.dtype == torch.float64 else time.float()
    log_alpha = -0.05 * work - 4.975 * work.square()
    alpha = log_alpha.exp()
    # expm1 retains precision near s=0; sigma(0)=0 is defined exactly.
    sigma = (-torch.expm1(2 * log_alpha)).clamp_min(0).sqrt()
    if like is not None:
        shape = (like.shape[0],) + (1,) * (like.ndim - 1)
        alpha, sigma = alpha.reshape(shape), sigma.reshape(shape)
    return alpha, sigma


def vp_beta(time: Tensor) -> Tensor:
    vp_factors(time)  # Validate coordinates before computing a drift.
    return 0.1 + 19.9 * time


class VPCleanDiffusion(nn.Module):
    """VP x0 prediction in intrinsic montage coordinates, not DDPM-v or CFM.

    ``noise`` returns the noisy tensor only. The training target is clean data.
    The supplied predictor returns x0 in the same coordinates as its input;
    modality dropout and classifier-free guidance are the caller's responsibility.
    """

    def __init__(self, s_min: float = 0.001, s_max: float = 1.0, snr_cap: float = 5.0):
        super().__init__()
        if not 0 < s_min < s_max <= 1 or not math.isfinite(snr_cap) or snr_cap <= 0:
            raise ValueError("VP endpoints require 0<s_min<s_max<=1 and a positive SNR cap")
        self.s_min, self.s_max, self.snr_cap = s_min, s_max, snr_cap

    def factors(self, time: Tensor, like: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return vp_factors(time, like)

    def noise(self, clean: Tensor, epsilon: Tensor, time: Tensor) -> Tensor:
        if clean.shape != epsilon.shape or clean.ndim < 2:
            raise ValueError("clean and Gaussian noise must share [batch,...] shape")
        if not torch.isfinite(clean).all() or not torch.isfinite(epsilon).all():
            raise ValueError("training signal and Gaussian noise must be finite")
        alpha, sigma = self.factors(time, clean)
        return alpha * clean + sigma * epsilon

    def snr(self, time: Tensor) -> Tensor:
        alpha, sigma = self.factors(time)
        return alpha.square() / sigma.square().clamp_min(torch.finfo(alpha.dtype).tiny)

    def clean_weight(self, time: Tensor) -> Tensor:
        return self.snr(time).clamp_max(self.snr_cap)

    def auxiliary_weight(self, time: Tensor) -> Tensor:
        alpha, _ = self.factors(time)
        return alpha.square()

    def clean_loss(self, prediction: Tensor, clean: Tensor, time: Tensor,
                   valid: Tensor | None = None) -> Tensor:
        if prediction.shape != clean.shape or clean.ndim < 2:
            raise ValueError("clean prediction and target must have matching [batch,...] shape")
        mask = torch.ones_like(clean, dtype=torch.bool) if valid is None else valid
        if mask.dtype != torch.bool:
            raise ValueError("validity mask must be boolean")
        try:
            mask = torch.broadcast_to(mask, clean.shape)
        except RuntimeError as error:
            raise ValueError("validity must broadcast to clean prediction shape") from error
        observed_prediction = torch.where(mask, prediction, 0).float()
        observed_clean = torch.where(mask, clean, 0).float()
        if not torch.isfinite(observed_prediction).all() or not torch.isfinite(observed_clean).all():
            raise ValueError("valid clean prediction and target must be finite")
        count = mask.flatten(1).sum(-1)
        per_sample = (observed_prediction - observed_clean).square().flatten(1).sum(-1)
        per_sample = per_sample / count.clamp_min(1)
        observed = count > 0
        weights = self.clean_weight(time)
        if weights.shape != count.shape:
            raise ValueError("noise time must match prediction batch")
        return (per_sample * weights * observed).sum() / observed.sum().clamp_min(1)

    def probability_flow_drift(self, state: Tensor, clean_prediction: Tensor,
                               time: Tensor) -> Tensor:
        if state.shape != clean_prediction.shape:
            raise ValueError("the VP predictor must return clean state coordinates")
        alpha, sigma = self.factors(time, state)
        if (sigma == 0).any():
            raise ValueError("probability-flow integration requires a finite noise endpoint")
        beta = vp_beta(time).reshape((state.shape[0],) + (1,) * (state.ndim - 1))
        # f - (g^2/2) score, with score=(alpha*x0_hat-x_s)/sigma^2.
        return 0.5 * beta * alpha * (alpha * state - clean_prediction) / sigma.square()

    def _predict(self, predictor: Predictor, state: Tensor, time: Tensor) -> Tensor:
        prediction = predictor(state, time)
        if prediction.shape != state.shape or not torch.isfinite(prediction).all():
            raise ValueError("predictor must return finite clean coordinates matching its input")
        return prediction.to(state.dtype)

    def _initial(self, initial_noise: Tensor) -> Tensor:
        if (initial_noise.ndim < 2 or initial_noise.shape[0] < 1
                or not initial_noise.is_floating_point() or not torch.isfinite(initial_noise).all()):
            raise ValueError("initial Gaussian noise must be nonempty finite [batch,...]")
        # Keep integration arithmetic outside reduced-precision model autocast.
        return initial_noise.clone().to(torch.float64 if initial_noise.dtype == torch.float64
                                        else torch.float32)

    @torch.no_grad()
    def sample_heun(self, predict: Predictor, initial_noise: Tensor, intervals: int = 50,
                    *, return_stats: bool = False) -> Tensor | tuple[Tensor, dict]:
        if not isinstance(intervals, int) or intervals < 1:
            raise ValueError("Heun intervals must be a positive integer")
        state = self._initial(initial_noise)
        # Quadratic spacing resolves the low-noise endpoint without an infinite
        # log-SNR step. 50 intervals cost 101 predictor calls, including x0 return.
        grid = torch.linspace(math.sqrt(self.s_max), math.sqrt(self.s_min), intervals + 1,
                              dtype=state.dtype, device=state.device).square()
        grid[0], grid[-1] = self.s_max, self.s_min
        nfe = 0
        for start, end in zip(grid[:-1], grid[1:]):
            time = start.expand(state.shape[0])
            clean = self._predict(predict, state, time)
            nfe += 1
            drift = self.probability_flow_drift(state, clean, time)
            delta = end - start
            proposal = state + delta * drift
            end_time = end.expand(state.shape[0])
            end_clean = self._predict(predict, proposal, end_time)
            nfe += 1
            end_drift = self.probability_flow_drift(proposal, end_clean, end_time)
            state = state + 0.5 * delta * (drift + end_drift)
            if not torch.isfinite(state).all():
                raise FloatingPointError("Heun state diverged; finite endpoint integration failed")
        # The endpoint predictor in Heun saw the Euler proposal, not the final
        # corrected state. This extra call is mandatory and counted explicitly.
        result = self._predict(predict, state, grid[-1].expand(state.shape[0]))
        nfe += 1
        stats = {"sampler": "vp-probability-flow-heun", "intervals": intervals,
                 "predictor_calls": nfe, "nfe": nfe,
                 "s_start": self.s_max, "s_final": self.s_min,
                 "returned_state": "clean_prediction", "grid": "quadratic"}
        return (result, stats) if return_stats else result

    @torch.no_grad()
    def sample_ddim(self, predict: Predictor, initial_noise: Tensor, steps: int = 101,
                    *, return_stats: bool = False) -> Tensor | tuple[Tensor, dict]:
        """Deterministic VP DDIM comparison with exactly ``steps`` predictor calls."""
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("DDIM predictor steps must be positive")
        state = self._initial(initial_noise)
        grid = torch.linspace(math.sqrt(self.s_max), math.sqrt(self.s_min), steps,
                              dtype=state.dtype, device=state.device).square()
        if steps > 1:
            grid[0], grid[-1] = self.s_max, self.s_min
        for position, value in enumerate(grid):
            time = value.expand(state.shape[0])
            clean = self._predict(predict, state, time)
            if position + 1 == steps:
                result = clean
                break
            alpha, sigma = self.factors(time, state)
            epsilon = (state - alpha * clean) / sigma
            next_time = grid[position + 1].expand(state.shape[0])
            next_alpha, next_sigma = self.factors(next_time, state)
            state = next_alpha * clean + next_sigma * epsilon
        stats = {"sampler": "vp-ddim", "predictor_calls": steps, "nfe": steps,
                 "s_start": self.s_max, "s_final": float(grid[-1].item()),
                 "returned_state": "clean_prediction", "grid": "quadratic"}
        return (result, stats) if return_stats else result

    def sample(self, predict: Predictor, initial_noise: Tensor, steps: int = 50,
               *, return_stats: bool = False) -> Tensor | tuple[Tensor, dict]:
        """Compatibility call: ``steps`` denotes Heun intervals, not NFE."""
        return self.sample_heun(predict, initial_noise, steps, return_stats=return_stats)


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
