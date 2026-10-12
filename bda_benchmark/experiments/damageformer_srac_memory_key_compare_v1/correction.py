"""Matched-capacity pixel corrections of a frozen native grid_probq baseline.

All three arms consume 256 pixel features, four baseline probabilities and
ten evidence channels. Query-only zeros those ten channels, retaining the
F256 retrieval coverage mask. No label/GT input is accepted by this module.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


TRAIN_ARMS = ("query_only", "t128_memory", "f256_memory")
FORMAL_STEPS, SMOKE_STEPS = 5000, 20
FIT_FORMAT = "SRAC_MEMORY_KEY_CORRECTION_FIT_V1"
EVAL_FORMAT = "SRAC_MEMORY_KEY_CORRECTION_EVAL_V1"


class PixelMemoryCorrection(nn.Module):
    """Small gated, bounded correction at the frozen feature-grid resolution.

    A zero-initialized delta is exactly the baseline in the forward pass, with
    the ordinary normalized-probability derivative retained for learning.
    An empty support or force_off returns the original baseline tensor itself.
    """

    def __init__(self, hidden=64, bound=2.0):
        super().__init__()
        if type(hidden) is not int or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("bound must be finite and positive")
        self.hidden, self.bound = hidden, float(bound)
        self.trunk = nn.Sequential(nn.Conv2d(270, hidden, 1), nn.GELU())
        self.delta = nn.Conv2d(hidden, 4, 1)
        self.gate = nn.Conv2d(hidden, 1, 1)
        for layer in (self.delta, self.gate):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, feature, base_probability, evidence_grid, coverage, *, force_off=False):
        if (feature.ndim != 4 or feature.shape[1] != 256
                or base_probability.ndim != 4 or base_probability.shape[1] != 4
                or evidence_grid.shape != (feature.shape[0], 10, *feature.shape[-2:])
                or coverage.shape != (feature.shape[0], 1, *feature.shape[-2:])
                or base_probability.shape[0] != feature.shape[0]):
            raise ValueError("Expected feature[B,256,h,w], base[B,4,H,W], evidence[B,10,h,w], coverage[B,1,h,w]")
        tensors = (feature, base_probability, evidence_grid, coverage)
        if any(x.device != feature.device or x.dtype != torch.float32 for x in tensors):
            raise ValueError("All correction inputs must share device and FP32 dtype")
        if any(not bool(torch.isfinite(x).all()) for x in tensors):
            raise ValueError("Nonfinite correction input")
        if bool((base_probability < 0).any()) or bool(((coverage < 0) | (coverage > 1 + 2e-5)).any()):
            raise ValueError("Invalid probability or coverage")
        if not torch.allclose(base_probability.sum(1), torch.ones_like(base_probability[:, 0]), atol=2e-5, rtol=2e-5):
            raise ValueError("Baseline class probabilities must sum to one")
        if force_off or not bool((coverage > 0).any()):
            return dict(probability=base_probability, gate=coverage * 0,
                        delta=feature.new_zeros(feature.shape[0], 4, *feature.shape[-2:]))
        grid_base = F.interpolate(base_probability, feature.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.trunk(torch.cat((feature, grid_base, evidence_grid), 1))
        gate = self.gate(hidden).sigmoid()
        delta = self.bound * self.delta(hidden).tanh() * gate * coverage.clamp(0, 1)
        full_delta = F.interpolate(delta, base_probability.shape[-2:], mode="bilinear", align_corners=False)
        full_coverage = F.interpolate(coverage, base_probability.shape[-2:], mode="bilinear", align_corners=False)
        weighted = base_probability * full_delta.exp()
        candidate = weighted / weighted.sum(1, keepdim=True).clamp_min(torch.finfo(weighted.dtype).tiny)
        # Avoid baseline normalization rounding at initialization while keeping
        # useful gradients through the initially zero delta. Evaluation returns
        # the original tensor, so initialized predictions are exact identities.
        if not bool((delta.detach() != 0).any()):
            probability = (base_probability + (candidate - candidate.detach())
                           if torch.is_grad_enabled() and candidate.requires_grad else base_probability)
        else:
            probability = torch.where(full_coverage > 0, candidate, base_probability)
        return dict(probability=probability, gate=gate, delta=delta)


def make_heads(hidden=64, bound=2.0):
    template = PixelMemoryCorrection(hidden, bound)
    return nn.ModuleDict({arm: copy.deepcopy(template) for arm in TRAIN_ARMS})


@torch.no_grad()
def correction_inputs(feature, result, base_probability, evidence, *, query_only=False):
    """Unpool the ten region evidence channels using the unchanged Q routes."""
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import unpool
    if (feature.shape[0] != 1 or tuple(feature.shape[-2:]) != tuple(result["shape"])
            or evidence["posterior"].shape != (result["count"], 4)
            or evidence["quality"].shape != (result["count"], 6)
            or evidence["supported"].shape != (result["count"],)):
        raise ValueError("Evidence, original Q and feature shapes disagree")
    channels = torch.cat((evidence["posterior"], evidence["quality"]), -1).float()
    if query_only:
        channels = torch.zeros_like(channels)
    grid = unpool(channels[None], result["q"], result["ids"], result["shape"])
    coverage = unpool(evidence["supported"][None, :, None].float(),
                      result["q"], result["ids"], result["shape"]).clamp(0, 1)
    frozen_base = base_probability.detach().float() if base_probability.requires_grad else base_probability.float()
    return dict(feature=feature.detach().float(), base_probability=frozen_base,
                evidence_grid=grid.detach(), coverage=coverage.detach())


def apply_heads(heads, inputs, *, force_off=False):
    if set(heads) != set(TRAIN_ARMS) or set(inputs) != set(TRAIN_ARMS):
        raise ValueError("All three matched arms are required")
    return {arm: heads[arm](**inputs[arm], force_off=force_off) for arm in TRAIN_ARMS}


def assert_evaluation(report, *, smoke=False, initial=False):
    immutable = report.get("immutable")
    if (report.get("complete") is not True or not immutable
            or not all(value is True for value in immutable.values())
            or report.get("native_a1_replay_ok") is not True
            or report.get("pool_ok") is not True
            or report.get("test_accessed") is not False
            or report.get("smoke") is not smoke):
        raise ValueError("Incomplete correction evaluation or failed source/native/pool checks")
    if initial and report.get("initial_identity_passed") is not True:
        raise ValueError("Correction initialization did not reproduce grid_probq exactly")
    if smoke:
        if report.get("image_count") != 2 or report.get("full_validation_evidence") is not False:
            raise ValueError("Smoke correction evaluation must contain exactly two nonformal Val images")
    elif (report.get("image_count") != 450
          or report.get("full_validation_evidence") is not True
          or report.get("reproduction", {}).get("passed") is not True
          or report.get("e0_metric_comparison", {}).get("passed") is not True):
        raise ValueError("Formal correction evaluation requires passed unchanged Val450 reproduction")
