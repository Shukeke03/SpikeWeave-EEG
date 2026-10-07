"""Small, original attention primitives with explicit True-is-valid masks."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def sinusoidal_embedding(position: Tensor, dim: int) -> Tensor:
    """Embed an arbitrary physical coordinate or diffusion coordinate separately."""
    half = dim // 2
    if half == 0:
        return position.unsqueeze(-1)
    frequency = torch.exp(-math.log(10000.0) * torch.arange(
        half, device=position.device, dtype=torch.float32) / max(half - 1, 1))
    phase = position.float().unsqueeze(-1) * frequency
    output = torch.cat((phase.sin(), phase.cos()), dim=-1)
    if dim % 2:
        output = F.pad(output, (0, 1))
    return output.to(position.dtype)


def masked_mean(values: Tensor, weight: Tensor, dim: int | tuple[int, ...]) -> Tensor:
    """Average only observed entries; fractional weights represent valid coverage."""
    weight = weight.to(values.dtype)
    clean = torch.where(weight.unsqueeze(-1) > 0, values, 0)
    numerator = (clean * weight.unsqueeze(-1)).sum(dim=dim)
    denominator = weight.sum(dim=dim).unsqueeze(-1).clamp_min(1e-8)
    return numerator / denominator


def visible_null(tokens: Tensor, valid: Tensor, null: Tensor) -> tuple[Tensor, Tensor]:
    """Use one visible null token for empty samples without marking missing data real."""
    tokens = torch.where(valid.unsqueeze(-1), tokens, 0)
    empty = ~valid.any(dim=1)
    if tokens.shape[1] == 0:
        raise ValueError("token sequences must have at least one slot")
    tokens = tokens.clone()
    valid = valid.clone()
    tokens[:, 0] = torch.where(empty[:, None], null.to(tokens.dtype), tokens[:, 0])
    valid[:, 0] |= empty
    return tokens, valid


class MaskedAttention(nn.Module):
    """SDPA attention; handles empty windows and never consumes masked NaNs."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0,
                 context_dim: int | None = None):
        super().__init__()
        if dim % heads:
            raise ValueError("attention dimension must divide evenly into heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        context_dim = dim if context_dim is None else context_dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(context_dim, dim)
        self.value = nn.Linear(context_dim, dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x: Tensor, valid: Tensor, context: Tensor | None = None,
                context_valid: Tensor | None = None, allowed: Tensor | None = None,
                bias: Tensor | None = None) -> Tensor:
        context = x if context is None else context
        context_valid = valid if context_valid is None else context_valid
        valid, context_valid = valid.bool(), context_valid.bool()
        if valid.shape != x.shape[:2] or context_valid.shape != context.shape[:2]:
            raise ValueError("attention masks must match token sequence shapes")
        x = torch.where(valid[..., None], x, 0)
        context = torch.where(context_valid[..., None], context, 0)
        batch, nq, dim = x.shape
        nk = context.shape[1]
        if nk == 0:
            raise ValueError("attention context must include a token or a null token")
        q = self.query(x).view(batch, nq, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(context).view(batch, nk, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(context).view(batch, nk, self.heads, self.head_dim).transpose(1, 2)
        permit = context_valid[:, None, None, :].expand(batch, 1, nq, nk)
        if allowed is not None:
            if allowed.ndim == 2:
                allowed = allowed[None, None]
            elif allowed.ndim == 3:
                allowed = allowed[:, None]
            permit = permit & allowed.bool()
        # Padded, completely empty windows still need a defined attention row.
        # Their query mask zeros the output; this fallback is never pooled as data.
        empty_row = ~permit.any(dim=-1, keepdim=True)
        first_key = torch.zeros((1, 1, 1, nk), dtype=torch.bool, device=x.device)
        first_key[..., 0] = True
        permit = permit | (empty_row & first_key)
        if bias is None:
            attention_mask = permit
        else:
            if bias.ndim == 2:
                bias = bias[None, None]
            elif bias.ndim == 3:
                bias = bias[None]
            attention_mask = bias.to(q.dtype).expand(batch, self.heads, nq, nk)
            attention_mask = attention_mask.masked_fill(~permit, float("-inf"))
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                                dropout_p=self.dropout if self.training else 0.0)
        output = output.transpose(1, 2).reshape(batch, nq, dim)
        return torch.where(valid[..., None], self.output(output), 0)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = MaskedAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, x: Tensor, valid: Tensor, allowed: Tensor | None = None,
                bias: Tensor | None = None) -> Tensor:
        x = torch.where(valid[..., None], x, 0)
        x = x + self.attention(self.norm1(x), valid, allowed=allowed, bias=bias)
        x = x + self.mlp(self.norm2(x))
        return torch.where(valid[..., None], x, 0)


class AxialLeadBlock(nn.Module):
    """Separate physical time and same-time lead attention, not image adjacency."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.temporal = TransformerBlock(dim, heads, dropout)
        self.lead = TransformerBlock(dim, heads, dropout)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        batch, channels, length, dim = x.shape
        x = self.temporal(x.reshape(batch * channels, length, dim),
                          valid.reshape(batch * channels, length))
        x = x.reshape(batch, channels, length, dim).transpose(1, 2)
        x = self.lead(x.reshape(batch * length, channels, dim),
                      valid.transpose(1, 2).reshape(batch * length, channels))
        return x.reshape(batch, length, channels, dim).transpose(1, 2)


def window_partition(x: Tensor, window: int) -> Tensor:
    batch, height, width, dim = x.shape
    return x.view(batch, height // window, window, width // window, window, dim).permute(
        0, 1, 3, 2, 4, 5).reshape(-1, window * window, dim)


def window_reverse(x: Tensor, window: int, batch: int, height: int, width: int) -> Tensor:
    return x.view(batch, height // window, width // window, window, window, -1).permute(
        0, 1, 3, 2, 4, 5).reshape(batch, height, width, -1)


class ShiftedWindowBlock(nn.Module):
    """Numerical frequency/time Swin block with true shifted-window masking."""

    def __init__(self, dim: int, heads: int, window: int, shift: int,
                 dropout: float = 0.0):
        super().__init__()
        if window <= 0 or not 0 <= shift < window:
            raise ValueError("invalid Swin window/shift")
        self.window, self.shift = window, shift
        self.block = TransformerBlock(dim, heads, dropout)
        self.relative_bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        coordinates = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window),
                                                 indexing="ij")).flatten(1)
        relative = coordinates[:, :, None] - coordinates[:, None, :]
        relative = relative.permute(1, 2, 0) + window - 1
        index = relative[..., 0] * (2 * window - 1) + relative[..., 1]
        self.register_buffer("relative_index", index, persistent=False)
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def _shift_mask(self, height: int, width: int, device: torch.device) -> Tensor | None:
        if not self.shift:
            return None
        region = torch.zeros((1, height, width, 1), device=device)
        slices = (slice(0, -self.window), slice(-self.window, -self.shift),
                  slice(-self.shift, None))
        counter = 0
        for row in slices:
            for col in slices:
                region[:, row, col] = counter
                counter += 1
        ids = window_partition(region, self.window).squeeze(-1)
        return ids[:, :, None] == ids[:, None, :]

    def forward(self, x: Tensor, weight: Tensor) -> Tensor:
        batch, height, width, dim = x.shape
        pad_h, pad_w = (-height) % self.window, (-width) % self.window
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        valid = F.pad(weight > 0, (0, pad_w, 0, pad_h))
        hp, wp = height + pad_h, width + pad_w
        if self.shift:
            x = torch.roll(x, (-self.shift, -self.shift), (1, 2))
            valid = torch.roll(valid, (-self.shift, -self.shift), (1, 2))
        windows = window_partition(x, self.window)
        window_valid = window_partition(valid[..., None], self.window).squeeze(-1)
        allowed = self._shift_mask(hp, wp, x.device)
        if allowed is not None:
            allowed = allowed.repeat(batch, 1, 1)
        bias = self.relative_bias[self.relative_index].permute(2, 0, 1)
        windows = self.block(windows, window_valid, allowed=allowed, bias=bias)
        x = window_reverse(windows, self.window, batch, hp, wp)
        if self.shift:
            x = torch.roll(x, (self.shift, self.shift), (1, 2))
        x = x[:, :height, :width]
        return torch.where((weight > 0)[..., None], x, 0)


class MaskedPatchMerging(nn.Module):
    """Swin 2x2 concatenating merge, excluding absent children from normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
        height, width = x.shape[1:3]
        x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        weight = F.pad(weight, (0, width % 2, 0, height % 2))
        children = torch.stack((x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                                x[:, 0::2, 1::2], x[:, 1::2, 1::2]), dim=-2)
        weights = torch.stack((weight[:, 0::2, 0::2], weight[:, 1::2, 0::2],
                               weight[:, 0::2, 1::2], weight[:, 1::2, 1::2]), dim=-1)
        observed = weights[..., None]
        children = torch.where(observed > 0, children, 0)
        denominator = (weights.sum(-1, keepdim=True) * x.shape[-1]).clamp_min(1e-8)
        mean = (children * observed).sum((-2, -1), keepdim=False)[..., None] / denominator
        variance = ((children - mean[..., None]) ** 2 * observed).sum(
            (-2, -1), keepdim=False)[..., None] / denominator
        children = torch.where(observed > 0,
                               (children - mean[..., None]) * torch.rsqrt(variance[..., None] + 1e-5), 0)
        merged_weight = weights.mean(-1)
        merged = self.reduction(children.flatten(-2))
        return torch.where((merged_weight > 0)[..., None], merged, 0), merged_weight
