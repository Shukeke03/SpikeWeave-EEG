"""Differentiable signal transforms with explicit finite-window conventions.

The dense DWT matrices are a reference implementation, not a computational
compression. Morlet CWT uses our documented discrete kernels rather than
claiming numerical equivalence to PyWavelets' CWT.
"""

from __future__ import annotations

import math

import numpy as np
import pywt
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _signal_input(x: Tensor, samples: int | None = None) -> None:
    if x.ndim != 3 or x.shape[-1] == 0:
        raise ValueError("signal must have shape [batch, channels, samples] with samples > 0")
    if not torch.is_floating_point(x):
        raise TypeError("signal must be a real floating-point tensor")
    if samples is not None and x.shape[-1] != samples:
        raise ValueError(f"expected {samples} samples, got {x.shape[-1]}")


class WaveletTransform(nn.Module):
    """Complete signed, packed orthogonal DWT in ``A_J,D_J,...,D_1`` order.

    PyWavelets builds the fixed analysis matrix only during initialization.
    Both directions thereafter use torch matrix multiplication, including on
    CUDA, and preserve autograd. At most 4096 samples are accepted: storing a
    dense float64 analysis matrix alone takes 128 MiB at that limit.

    With periodization and length divisible by 2**levels the transform is a
    square orthogonal operator. Its transpose is therefore the exact synthesis
    operator, up to numerical precision. No coefficient magnitudes are taken.
    Half/bfloat16 inputs are promoted to float32; autocast is disabled for the
    transform. CPU and CUDA are the tested backends. For MPS, cast this module
    to float32 before moving it, because MPS cannot store float64 buffers.
    """

    def __init__(
        self,
        samples: int,
        levels: int = 3,
        wavelet: str = "db4",
        mode: str = "periodization",
    ) -> None:
        super().__init__()
        if not isinstance(samples, int) or samples <= 0 or samples > 4096:
            raise ValueError("samples must be an integer in [1, 4096] for the dense DWT")
        if not isinstance(levels, int) or levels < 1:
            raise ValueError("levels must be a positive integer")
        if samples % (2**levels):
            raise ValueError("samples must be divisible by 2**levels")
        if mode != "periodization":
            raise ValueError("this packed square DWT requires mode='periodization'")
        basis_wavelet = pywt.Wavelet(wavelet)
        if not basis_wavelet.orthogonal:
            raise ValueError("this transpose-synthesis implementation requires an orthogonal wavelet")
        self.samples = samples
        self.levels = levels
        self.wavelet = wavelet
        self.mode = mode
        self.band_names = (f"A{levels}", *(f"D{j}" for j in range(levels, 0, -1)))
        self.band_lengths = (
            samples // 2**levels,
            *(samples // 2**j for j in range(levels, 0, -1)),
        )
        # Each row of the identity is one input basis vector. Concatenated
        # outputs form A.T, where y = A x in the column-vector convention.
        basis = np.eye(samples, dtype=np.float64)
        coefficients = pywt.wavedec(
            basis, basis_wavelet, mode=mode, level=levels, axis=-1
        )
        analysis = np.concatenate(coefficients, axis=-1).T.copy()
        self.register_buffer("analysis_matrix", torch.from_numpy(analysis))

    def _matrix(self, x: Tensor) -> tuple[Tensor, Tensor]:
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        return x.to(dtype=dtype), self.analysis_matrix.to(device=x.device, dtype=dtype)

    def encode(self, x: Tensor) -> Tensor:
        _signal_input(x, self.samples)
        with torch.autocast(device_type=x.device.type, enabled=False):
            value, matrix = self._matrix(x)
            return value @ matrix.T

    def decode(self, coefficients: Tensor) -> Tensor:
        _signal_input(coefficients, self.samples)
        with torch.autocast(device_type=coefficients.device.type, enabled=False):
            value, matrix = self._matrix(coefficients)
            return value @ matrix

    def forward(self, x: Tensor) -> Tensor:
        return self.encode(x)

    def coefficient_valid(self, time_valid: Tensor) -> Tensor:
        """Mark a coefficient valid only when all of its contributing samples are valid.

        This is optional support propagation for temporal gaps. Missing entire
        channels can instead be excluded directly. Thresholding at 1e-12 removes
        roundoff zeros, not meaningful db4 taps.
        """
        if time_valid.ndim != 3 or time_valid.shape[-1] != self.samples:
            raise ValueError("time_valid must have shape [batch, channels, samples]")
        if time_valid.dtype != torch.bool:
            raise TypeError("time_valid must be boolean")
        support = self.analysis_matrix.to(time_valid.device).abs() > 1e-12
        touched = (~time_valid).float() @ support.T.float()
        return touched == 0


class MorletCWT(nn.Module):
    """Finite-window complex Morlet power, pooled *before* the natural logarithm.

    The center parameter is omega0=6. Each sampled Gaussian/carrier kernel has
    scale omega0*sample_rate/(2*pi*frequency), is truncated at +/-4 scales,
    corrected to zero discrete DC, and normalized to unit discrete L2 norm.
    Coefficients therefore retain input amplitude units, with a documented
    frequency-dependent wavelet gain; this is not a calibrated spectral density.

    Linear FFT convolution uses zero padding, never circular wrapping. The
    Morlet cone mask excludes ceil(sqrt(2)*scale) samples at each window edge.
    Interior missing samples invalidate the entire truncated kernel support.
    Masked adaptive mean pooling acts on linear |W|**2, then log(power+1e-12).
    A pooled bin is valid if at least one valid coefficient contributes; all
    invalid bins are zero with an accompanying False mask.

    ``time_valid`` accepts [B,C,L], [B,1,L], [B,L], or [B,C]. CPU and CUDA FFT
    are supported. This implementation explicitly rejects MPS FFT; use CPU or
    CUDA for the transform rather than silently detaching for a fallback.
    """

    def __init__(
        self,
        sample_rate: float,
        frequencies: int = 64,
        min_frequency: float = 2.0,
        max_frequency: float = 80.0,
        time_bins: int = 128,
    ) -> None:
        super().__init__()
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("sample_rate must be finite and positive")
        if not isinstance(frequencies, int) or frequencies < 1:
            raise ValueError("frequencies must be a positive integer")
        if not isinstance(time_bins, int) or time_bins < 1:
            raise ValueError("time_bins must be a positive integer")
        if not 0 < min_frequency < max_frequency < sample_rate / 2:
            raise ValueError("frequencies must lie strictly between zero and Nyquist")
        self.sample_rate = float(sample_rate)
        self.time_bins = time_bins
        self.omega0 = 6.0
        self.epsilon = 1e-12
        hz = torch.logspace(
            math.log10(min_frequency), math.log10(max_frequency), frequencies,
            dtype=torch.float64,
        )
        scales = self.omega0 * self.sample_rate / (2 * math.pi * hz)
        radii = torch.ceil(4 * scales).long()
        self.kernel_radius = int(radii.max())
        offsets = torch.arange(-self.kernel_radius, self.kernel_radius + 1,
                               dtype=torch.float64)
        normalized_time = offsets[None] / scales[:, None]
        envelope = torch.exp(-0.5 * normalized_time.square())
        envelope = envelope * (offsets.abs()[None] <= radii[:, None])
        carrier = torch.polar(torch.ones_like(normalized_time), self.omega0 * normalized_time)
        dc = (envelope * carrier).sum(-1, keepdim=True) / envelope.sum(-1, keepdim=True)
        kernel = envelope * (carrier - dc)
        kernel = kernel / kernel.abs().square().sum(-1, keepdim=True).sqrt()
        self.register_buffer("frequencies", hz)
        self.register_buffer("kernel_real", kernel.real.contiguous())
        self.register_buffer("kernel_imag", kernel.imag.contiguous())
        self.register_buffer("support_radii", radii)
        self.register_buffer("cone_radii", torch.ceil(math.sqrt(2) * scales).long())

    @staticmethod
    def _valid_input(x: Tensor, time_valid: Tensor | None) -> Tensor:
        batch, channels, samples = x.shape
        if time_valid is None:
            return torch.ones_like(x, dtype=torch.bool)
        if time_valid.dtype != torch.bool:
            raise TypeError("time_valid must be boolean")
        mask = time_valid.to(device=x.device)
        if mask.ndim == 2 and mask.shape == (batch, samples):
            mask = mask[:, None, :]
        elif mask.ndim == 2 and mask.shape == (batch, channels):
            mask = mask[:, :, None]
        try:
            return torch.broadcast_to(mask, x.shape)
        except RuntimeError as error:
            raise ValueError("time_valid must broadcast to [batch, channels, samples]") from error

    def forward(self, x: Tensor, time_valid: Tensor | None = None) -> tuple[Tensor, Tensor]:
        _signal_input(x)
        if x.device.type == "mps":
            raise NotImplementedError("MorletCWT FFT uses CPU or CUDA; MPS is not supported")
        mask = self._valid_input(x, time_valid)
        if not bool(torch.isfinite(x).logical_or(~mask).all()):
            raise ValueError("observed signal samples must be finite")
        batch, channels, length = x.shape
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            values = torch.where(mask, x, 0).to(dtype=dtype).reshape(batch * channels, length)
            kernel = torch.complex(
                self.kernel_real.to(device=x.device, dtype=dtype),
                self.kernel_imag.to(device=x.device, dtype=dtype),
            )
            convolution_length = length + kernel.shape[-1] - 1
            fft_length = 1 << (convolution_length - 1).bit_length()
            signal_fft = torch.fft.fft(values, n=fft_length)
            kernel_fft = torch.fft.fft(kernel, n=fft_length)
            coefficients = torch.fft.ifft(signal_fft[:, None] * kernel_fft[None], dim=-1)
            coefficients = coefficients[..., self.kernel_radius:self.kernel_radius + length]
            power = coefficients.abs().square()

            positions = torch.arange(length, device=x.device)
            radii = self.support_radii.to(x.device)[:, None]
            left = (positions[None] - radii).clamp(min=0)
            right = (positions[None] + radii + 1).clamp(max=length)
            missing = (~mask).reshape(batch * channels, length).to(dtype=dtype)
            prefix = F.pad(missing.cumsum(-1), (1, 0))
            no_missing = (prefix[:, right] - prefix[:, left]) == 0
            cone = self.cone_radii.to(x.device)[:, None]
            boundary_valid = (positions[None] >= cone) & (length - 1 - positions[None] >= cone)
            valid = no_missing & boundary_valid[None]
            weights = valid.to(dtype=dtype)
            numerator = F.adaptive_avg_pool1d(power * weights, self.time_bins)
            denominator = F.adaptive_avg_pool1d(weights, self.time_bins)
            pooled_valid = denominator > 0
            pooled_power = numerator / denominator.clamp_min(self.epsilon)
            log_power = torch.log(pooled_power + self.epsilon)
            log_power = torch.where(pooled_valid, log_power, 0)
            shape = (batch, channels, self.frequencies.numel(), self.time_bins)
            return log_power.reshape(shape), pooled_valid.reshape(shape)


def lead_cycle_residual(x: Tensor, lead_matrix: Tensor) -> Tensor:
    """Return X-AA^+X, not its norm, for an explicitly supplied measurement matrix.

    Rows of lead_matrix must correspond to x's lead order. This enforces only
    the declared linear measurement hypothesis, not physiological connectivity.
    """
    _signal_input(x)
    if lead_matrix.ndim != 2 or lead_matrix.shape[0] != x.shape[1]:
        raise ValueError("lead_matrix must have shape [channels, electrode endpoints]")
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        matrix = lead_matrix.to(device=x.device, dtype=dtype)
        projection = matrix @ torch.linalg.pinv(matrix)
        values = x.to(dtype=dtype)
        return values - torch.einsum("cd,bdl->bcl", projection, values)
