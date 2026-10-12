"""Fixed-Q sparse local readout; no Memory, labels, or repartitioning.

Only the readout distribution changes: ``R_ik ∝ Q_ik exp(score_ik)``.
The supplied Q and region IDs remain the route support and pooling partition.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _normalize_routes(
    q: torch.Tensor,
    valid_candidates: torch.Tensor,
    scores: torch.Tensor | None,
) -> torch.Tensor:
    """Internal normalization after public shape/value checks."""
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    weights = q.to(dtype)
    active = valid_candidates & (weights > 0)
    # Substitute one only *inside log* at inactive entries, then close those
    # routes with -inf. This is not an epsilon prior and never opens a route.
    logits = torch.where(active, weights, 1).log()
    if scores is not None:
        logits = logits + scores.to(dtype).masked_fill(~active, 0)
    logits = logits.masked_fill(~active, -torch.inf)
    maximum = logits.amax(dim=1, keepdim=True)
    maximum = torch.where(active.any(dim=1, keepdim=True), maximum, 0)
    # Including log(Q) in the shift also protects score gradients when only
    # extremely small Q mass survives invalid-region masking.
    weights = (logits - maximum).exp()
    denominator = weights.sum(dim=1, keepdim=True)
    safe_denominator = torch.where(denominator > 0, denominator, 1)
    return weights / safe_denominator


def normalize_routes(
    q: torch.Tensor,
    valid_candidates: torch.Tensor,
    scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize candidates along dimension 1 of ``[B,S,N]`` tensors.

    Zero Q and invalid candidates remain exactly zero. With ``scores=None``
    this returns valid, renormalized Q; otherwise it returns the stable
    multiplicative content reweighting. Empty rows return all zeros. No epsilon
    is added to Q. Half/bfloat16 inputs accumulate and return float32 routes.
    """
    if q.ndim != 3 or min(q.shape) < 1 or not q.is_floating_point():
        raise ValueError("Q must be a nonempty floating [B,S,N] tensor")
    if valid_candidates.shape != q.shape or valid_candidates.dtype != torch.bool:
        raise ValueError("valid_candidates must be bool with Q's [B,S,N] shape")
    if valid_candidates.device != q.device:
        raise ValueError("Q and candidate validity must share a device")
    if not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
        raise ValueError("Q must be finite and nonnegative")
    if scores is not None:
        if scores.shape != q.shape or scores.device != q.device or not scores.is_floating_point():
            raise ValueError("Scores must be floating and match Q's shape/device")
        if not bool(torch.isfinite(scores[valid_candidates & (q > 0)]).all()):
            raise ValueError("Active route scores must be finite")
    return _normalize_routes(q, valid_candidates, scores)


class LocalSelectiveReadout(nn.Module):
    """Content-selective readout restricted to each pixel's existing routes.

    Pixels are processed in chunks, with at most ``B*S*chunk_size`` scores and
    ``B*chunk_size*D`` gathered values at a time. No ``N*K`` matrix is created.
    Exactly one scorer projection is zero at initialization, so R initially
    equals normalized valid Q while the zero projection receives gradients.
    """

    def __init__(
        self,
        pixel_dim: int = 256,
        token_dim: int = 128,
        score_dim: int = 32,
        chunk_size: int = 4096,
    ):
        super().__init__()
        if min(pixel_dim, token_dim, score_dim, chunk_size) < 1:
            raise ValueError("Readout dimensions and chunk_size must be positive")
        self.pixel_dim = int(pixel_dim)
        self.token_dim = int(token_dim)
        self.score_dim = int(score_dim)
        self.chunk_size = int(chunk_size)
        self.phi = nn.Linear(self.pixel_dim, self.score_dim, bias=False)
        self.psi = nn.Linear(self.token_dim, self.score_dim, bias=False)
        nn.init.zeros_(self.phi.weight)
        self.scale = self.score_dim ** -0.5

    def _validate(self, pixel, tokens, q, ids, valid, mode):
        if mode not in ("q", "content"):
            raise ValueError(f"Unknown readout mode: {mode}")
        if pixel.ndim != 4 or tokens.ndim != 3 or q.ndim != 3 or ids.ndim != 2:
            raise ValueError("Expected pixel[B,C,H,W], tokens[B,K,D], Q[B,S,N], ids[S,N]")
        batch, channels, height, width = pixel.shape
        supports, pixels = ids.shape
        if min(batch, height, width, supports, tokens.shape[1]) < 1:
            raise ValueError("Readout dimensions must be nonempty")
        if channels != self.pixel_dim or tokens.shape[0] != batch or tokens.shape[2] != self.token_dim:
            raise ValueError("Pixel/token dimensions do not match this readout")
        if pixels != height * width or q.shape != (batch, supports, pixels):
            raise ValueError("Sparse association shape mismatch")
        if valid.shape != tokens.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Region validity must be bool [B,K]")
        if ids.dtype != torch.long:
            raise ValueError("Candidate IDs must use torch.long")
        if any(tensor.device != pixel.device for tensor in (tokens, q, ids, valid)):
            raise ValueError("All readout inputs must share a device")
        if not all(tensor.is_floating_point() for tensor in (pixel, tokens, q)):
            raise ValueError("Pixel features, tokens and Q must be floating tensors")
        if bool((ids < 0).any()) or bool((ids >= tokens.shape[1]).any()):
            raise ValueError("Candidate ID outside the region token table")
        if not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
            raise ValueError("Q must be finite and nonnegative")

    def forward(
        self,
        pixel: torch.Tensor,
        tokens: torch.Tensor,
        q: torch.Tensor,
        ids: torch.Tensor,
        valid: torch.Tensor,
        mode: str = "q",
        *,
        return_routes: bool = False,
    ):
        """Return ``(context[B,D,H,W], diagnostics)``.

        Diagnostics are detached Python floats. For audits/tests only,
        ``return_routes=True`` adds R[B,S,N] as a third return item. Forward
        never modifies Q. Both modes use the same invalid-region masking.
        """
        self._validate(pixel, tokens, q, ids, valid, mode)
        batch, _, height, width = pixel.shape
        supports, pixels = ids.shape
        pixel_rows = pixel.flatten(2).transpose(1, 2)
        accumulator_dtype = torch.float64 if tokens.dtype == torch.float64 or q.dtype == torch.float64 else torch.float32
        totals = torch.zeros(8, device=pixel.device, dtype=accumulator_dtype)
        maxima = torch.zeros(2, device=pixel.device, dtype=accumulator_dtype)
        outputs, route_outputs = [], []
        with torch.autocast(device_type=pixel.device.type, enabled=False):
            token_keys = self.psi(tokens.to(self.psi.weight.dtype)) if mode == "content" else None
            for start in range(0, pixels, self.chunk_size):
                stop = min(start + self.chunk_size, pixels)
                chunk_ids = ids[:, start:stop]
                q_chunk = q[:, :, start:stop]
                validity = valid[:, chunk_ids]
                baseline = _normalize_routes(q_chunk, validity, None)
                scores = None
                if mode == "content":
                    queries = self.phi(pixel_rows[:, start:stop].to(self.phi.weight.dtype))
                    score_rows = []
                    for support in range(supports):
                        keys = token_keys[:, chunk_ids[support]]
                        score_rows.append((queries.to(accumulator_dtype) * keys.to(accumulator_dtype)).sum(-1) * self.scale)
                    scores = torch.stack(score_rows, dim=1)
                routes = baseline if scores is None else _normalize_routes(q_chunk, validity, scores)
                context = torch.zeros(batch, stop - start, self.token_dim, device=tokens.device, dtype=accumulator_dtype)
                for support in range(supports):
                    values = tokens[:, chunk_ids[support]].to(accumulator_dtype)
                    active = validity[:, support] & (q_chunk[:, support] > 0)
                    values = values.masked_fill(~active[..., None], 0)
                    context = context + values * routes[:, support, :, None]
                outputs.append(context)
                if return_routes:
                    route_outputs.append(routes)
                with torch.no_grad():
                    r, base = routes.detach(), baseline.detach()
                    active = validity & (q_chunk > 0)
                    # clamp only log arguments for diagnostics, never routes.
                    tiny = torch.finfo(r.dtype).tiny
                    log_r, log_base = r.clamp_min(tiny).log(), base.clamp_min(tiny).log()
                    difference = (r - base).abs()
                    active_scores = torch.zeros_like(r) if scores is None else scores.detach().masked_fill(~active, 0)
                    totals += torch.stack((
                        -(r * log_r).sum(),
                        -(base * log_base).sum(),
                        difference.sum(),
                        (r * (log_r - log_base)).sum(),
                        (~active.any(dim=1)).sum().to(r.dtype),
                        active.sum().to(r.dtype),
                        active_scores.abs().sum(),
                        r.masked_fill(active, 0).sum(),
                    )).to(accumulator_dtype)
                    maxima = torch.maximum(maxima, torch.stack((difference.amax(), active_scores.abs().amax())).to(accumulator_dtype))

        normalizer = float(batch * pixels)
        values = (totals / normalizer).cpu().tolist()
        max_delta, score_max = maxima.cpu().tolist()
        stats = dict(zip(("entropy_mean", "q_entropy_mean", "route_l1_mean", "route_kl_mean", "empty_fraction", "active_candidates_mean", "score_abs_mean", "invalid_route_mass"), values))
        # Score mean is over active candidates, while route metrics are per pixel.
        stats["score_abs_mean"] /= stats["active_candidates_mean"] if stats["active_candidates_mean"] > 0 else 1.0
        stats.update(max_route_delta=max_delta, score_abs_max=score_max)
        if not all(math.isfinite(value) for value in stats.values()):
            raise FloatingPointError("Non-finite local readout diagnostics")
        context_map = torch.cat(outputs, dim=1).transpose(1, 2).reshape(batch, self.token_dim, height, width).to(tokens.dtype)
        if return_routes:
            return context_map, stats, torch.cat(route_outputs, dim=2)
        return context_map, stats
