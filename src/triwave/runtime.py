"""Explicit device selection and reproducible execution settings."""

from __future__ import annotations

import random

import numpy as np
import torch


def select_device(request: str = "auto") -> torch.device:
    # MorletCWT uses complex FFT: CPU is the supported macOS verification path.
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if request not in {"cpu", "cuda"}:
        raise ValueError("supported devices are cpu, cuda, auto")
    return torch.device(request)


def seed_everything(seed: int = 0, threads: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
