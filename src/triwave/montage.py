"""Linear measurement subspaces of explicitly identified bipolar EEG leads.

The basis coordinates are measurement modes, never electrode identities. No
geometry matching, inferred reference, graph smoothing or source inversion is
performed here. Different missing-lead patterns require different subspaces.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class MontageSubspace(nn.Module):
    def __init__(self, lead_matrix: Tensor, lead_valid: Tensor | None = None,
                 *, max_relative_residual: float = 1e-3):
        super().__init__()
        matrix = torch.as_tensor(lead_matrix)
        if matrix.ndim != 2 or min(matrix.shape) < 1:
            raise ValueError("lead matrix must be nonempty [leads,electrodes]")
        if not torch.isfinite(matrix).all():
            raise ValueError("lead matrix must be finite and explicitly identified")
        if not math.isfinite(max_relative_residual) or max_relative_residual < 0:
            raise ValueError("residual tolerance must be finite and nonnegative")
        valid = (torch.ones(matrix.shape[0], dtype=torch.bool, device=matrix.device)
                 if lead_valid is None else torch.as_tensor(lead_valid, device=matrix.device))
        if valid.dtype != torch.bool or valid.shape != (matrix.shape[0],) or not valid.any():
            raise ValueError("lead_valid must identify at least one measured lead")
        # An incidence row must identify a voltage difference. A general matrix
        # with arbitrary entries could silently assert an unknown reference.
        work = matrix.to(torch.float64)
        active = work[valid]
        if not ((active == 1).sum(-1).eq(1) & (active == -1).sum(-1).eq(1)
                & ((active == 0) | (active == 1) | (active == -1)).all(-1)).all():
            raise ValueError("each measured row must have one +1 and one -1 endpoint")
        effective = torch.where(valid[:, None], work, 0)
        u, singular, _ = torch.linalg.svd(effective, full_matrices=False)
        tolerance = max(effective.shape) * torch.finfo(torch.float64).eps * singular[0]
        rank = int((singular > tolerance).sum().item())
        if not rank:
            raise ValueError("montage must have a nonzero measurement rank")
        q = u[:, :rank]
        # SVD roundoff on unobserved rows must not turn missing leads into data.
        q = torch.where(valid[:, None], q, 0)
        self.rank = rank
        self.max_relative_residual = max_relative_residual
        self.register_buffer("lead_matrix", work.float())
        self.register_buffer("lead_valid", valid)
        self.register_buffer("Q", q.float())

    @classmethod
    def from_endpoints(cls, endpoints: Tensor, num_electrodes: int | None = None,
                       lead_valid: Tensor | None = None, **kwargs) -> MontageSubspace:
        endpoints = torch.as_tensor(endpoints)
        if endpoints.ndim != 2 or endpoints.shape[1] != 2 or endpoints.shape[0] < 1:
            raise ValueError("endpoints must be [leads,2] positive/negative indices")
        if endpoints.dtype not in (torch.int32, torch.int64):
            raise ValueError("endpoint identities must be explicit integer indices")
        if (endpoints < 0).any() or (endpoints[:, 0] == endpoints[:, 1]).any():
            raise ValueError("endpoints must be distinct nonnegative electrode indices")
        inferred = int(endpoints.max().item()) + 1
        electrodes = inferred if num_electrodes is None else num_electrodes
        if not isinstance(electrodes, int) or electrodes < inferred:
            raise ValueError("num_electrodes must include every identified endpoint")
        matrix = torch.zeros((len(endpoints), electrodes), device=endpoints.device)
        rows = torch.arange(len(endpoints), device=endpoints.device)
        matrix[rows, endpoints[:, 0]] = 1
        matrix[rows, endpoints[:, 1]] = -1
        return cls(matrix, lead_valid, **kwargs)

    def _observed(self, values: Tensor, valid: Tensor | None = None) -> Tensor:
        if values.ndim != 3 or values.shape[1] != len(self.lead_valid):
            raise ValueError("measurements must be [batch,leads,time or coefficients]")
        if not values.is_floating_point() or values.shape[0] < 1 or values.shape[2] < 1:
            raise ValueError("measurements must be nonempty floating-point tensors")
        expected = self.lead_valid.to(values.device)[None].expand(values.shape[:2])
        if valid is not None:
            if valid.dtype != torch.bool or valid.shape != expected.shape:
                raise ValueError("measurement validity must be bool [batch,leads]")
            if not torch.equal(valid.to(expected.device), expected):
                raise ValueError("missing-lead patterns change the subspace; construct a new montage")
        observed = torch.where(expected[..., None], values, 0)
        if not torch.isfinite(observed).all():
            raise ValueError("observed measurements must be finite")
        return observed

    def audit(self, values: Tensor, valid: Tensor | None = None,
              *, max_relative_residual: float | None = None) -> dict:
        threshold = (self.max_relative_residual if max_relative_residual is None
                     else max_relative_residual)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("residual tolerance must be finite and nonnegative")
        observed = self._observed(values, valid).double()
        q = self.Q.to(device=values.device, dtype=torch.float64)
        residual = observed - q @ (q.T @ observed)
        denominator = int(self.lead_valid.sum().item()) * observed.shape[-1]
        rms = (residual.square().sum((1, 2)) / denominator).sqrt()
        source_rms = (observed.square().sum((1, 2)) / denominator).sqrt()
        relative = rms / source_rms.clamp_min(torch.finfo(torch.float64).eps)
        accepted = relative <= threshold
        return {"rank": self.rank, "lead_count": len(self.lead_valid),
                "observed_leads": int(self.lead_valid.sum().item()),
                "max_relative_residual": threshold,
                "residual_rms": rms.detach().cpu().tolist(),
                "relative_residual": relative.detach().cpu().tolist(),
                "passed_per_sample": accepted.detach().cpu().tolist(),
                "passed": bool(accepted.all().item())}

    def reduce(self, values: Tensor, valid: Tensor | None = None,
               *, require_audit: bool = True,
               max_relative_residual: float | None = None) -> Tensor:
        observed = self._observed(values, valid)
        if require_audit:
            report = self.audit(observed, valid, max_relative_residual=max_relative_residual)
            if not report["passed"]:
                raise ValueError("montage residual audit failed; hard projection would discard "
                                 "unexplained measurement residual")
        return self.Q.to(device=values.device, dtype=values.dtype).T @ observed

    def lift(self, coordinates: Tensor) -> Tensor:
        if coordinates.ndim != 3 or coordinates.shape[1] != self.rank:
            raise ValueError("measurement-mode coordinates must be [batch,rank,length]")
        if not coordinates.is_floating_point() or not torch.isfinite(coordinates).all():
            raise ValueError("measurement-mode coordinates must be finite floating point")
        return self.Q.to(device=coordinates.device, dtype=coordinates.dtype) @ coordinates

    def project(self, values: Tensor, valid: Tensor | None = None,
                *, require_audit: bool = True,
                max_relative_residual: float | None = None) -> Tensor:
        return self.lift(self.reduce(values, valid, require_audit=require_audit,
                                     max_relative_residual=max_relative_residual))

    def residual(self, values: Tensor, valid: Tensor | None = None) -> Tensor:
        """Retain the unexplained component for audit; never silently delete it."""
        return self._observed(values, valid) - self.project(values, valid, require_audit=False)
