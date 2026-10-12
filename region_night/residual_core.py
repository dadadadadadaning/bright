"""Frozen-Q region assistance with an equally sized local-only control.

Both arms retain grid_probq as the probability base.  The trainable residual
sees F256 and either a second copy of local F256 or Q-mean/Q-unpool F256.
This is a new readout candidate, not an adaptation of the original T128 head.
No target labels enter preparation or prediction.
"""
from __future__ import annotations

import copy

import torch
from torch import nn

from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import (
    mean_pool, resize_probability, unpool,
)

ARMS = ("local_delta", "region_delta")
FORMAT = "BRIGHT_REGION_RESIDUAL_V1"
DELTA_BOUND = 2.0


def _parent(front_head):
    if (not isinstance(front_head, nn.Conv2d) or front_head.in_channels != 256
            or front_head.out_channels != 4 or front_head.kernel_size != (1, 1)
            or front_head.stride != (1, 1) or front_head.padding != (0, 0)
            or front_head.groups != 1 or front_head.dilation != (1, 1)):
        raise ValueError("Requires the original FP32 affine 256-to-4 A1 head")
    if any(p.dtype != torch.float32 or not bool(torch.isfinite(p).all())
           for p in front_head.parameters()):
        raise ValueError("A1 classifier parameters must be finite FP32")
    return front_head


class BoundedResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Conv2d(512, 64, 1)
        self.activation = nn.GELU()
        self.output = nn.Conv2d(64, 4, 1)
        # Keep a live derivative: zero initialization is not a prediction branch.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, local, context):
        if local.shape != context.shape or local.ndim != 4 or local.shape[1] != 256:
            raise ValueError("Residual requires matching local/context F256 grids")
        raw = self.output(self.activation(self.hidden(torch.cat((local, context), 1))))
        return DELTA_BOUND * torch.tanh(raw / DELTA_BOUND)


def make_heads(front):
    parent = _parent(front.base.clf_2)
    prototype = BoundedResidual().to(device=parent.weight.device, dtype=torch.float32)
    heads = nn.ModuleDict({arm: copy.deepcopy(prototype) for arm in ARMS})
    heads.requires_grad_(True)
    return heads


def parameter_checks(front, heads):
    if set(heads) != set(ARMS):
        raise ValueError("Both residual arms are required")
    a, b = (heads[arm] for arm in ARMS)
    a_state, b_state = a.state_dict(), b.state_dict()
    a_ids, b_ids = ({id(p) for p in x.parameters()} for x in (a, b))
    source_ids = {id(p) for p in front.base.clf_2.parameters()}
    counts = {arm: sum(p.numel() for p in heads[arm].parameters()) for arm in ARMS}
    flags = dict(
        same_initialization=set(a_state) == set(b_state)
        and all(torch.equal(a_state[k], b_state[k]) for k in a_state),
        equal_parameter_count=len(set(counts.values())) == 1,
        independent_parameters=not bool(a_ids & b_ids),
        excludes_source_parameters=not bool((a_ids | b_ids) & source_ids),
        private_trainable_fp32=all(p.requires_grad and p.dtype == torch.float32
                                   and bool(torch.isfinite(p).all())
                                   for p in heads.parameters()),
        zero_initial_residual=all(bool((h.output.weight == 0).all())
                                  and bool((h.output.bias == 0).all()) for h in (a, b)),
    )
    return dict(flags, passed=all(flags.values()), parameters_per_arm=counts,
                local_context="F256 repeated; no Q-derived feature input",
                region_context="exact Q mean of F256, then same-Q unpool",
                capacity_limit="equal architecture/parameter count; duplicated local input is redundant",
                delta_bound=DELTA_BOUND, original_T128_used=False)


@torch.no_grad()
def prepare(feature, result, front_head):
    parent = _parent(front_head)
    if any(k not in result for k in ("q", "ids", "count", "shape")):
        raise ValueError("Incomplete frozen Q output")
    feature = feature.detach().float()
    q, ids = result["q"].detach().float(), result["ids"].detach()
    shape = tuple(result["shape"])
    if (feature.ndim != 4 or feature.shape[1] != 256
            or tuple(feature.shape[-2:]) != shape or feature.device != parent.weight.device):
        raise ValueError("Expected finite F256 on the original Q grid and A1 device")
    mean, mass = mean_pool(feature, q, ids, result["count"])
    rebuilt = unpool(mean, q, ids, shape)
    local_probabilities = parent(feature).float().softmax(1)
    probability_tokens, _ = mean_pool(local_probabilities, q, ids, result["count"])
    base = unpool(probability_tokens, q, ids, shape)
    if not bool(torch.isfinite(base).all()) or bool((base < 0).any()):
        raise ValueError("Invalid frozen grid_probq probability")
    return dict(feature=feature, mean_tokens=mean.detach(), mean_rebuilt=rebuilt.detach(), base_grid=base.detach(),
                q=q, ids=ids, count=result["count"], shape=shape, mass=mass.detach())


def probability(arm, head, inputs, output_size):
    """Differentiable in the private residual only; correction precedes resizing.

Zero initialization matches original grid_probq within FP32 normalization
roundoff.  A zero base class remains zero; no epsilon changes its evidence.
"""
    if arm not in ARMS:
        raise ValueError("Unknown residual arm: " + str(arm))
    local = inputs["feature"].detach()
    context = local if arm == "local_delta" else inputs["mean_rebuilt"].detach()
    delta = head(local, context)
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("Nonfinite private residual")
    base = inputs["base_grid"].detach()
    weighted = base * delta.exp()
    denominator = weighted.sum(1, keepdim=True)
    if (not bool(torch.isfinite(denominator).all()) or bool((denominator <= 0).any())):
        raise ValueError("Nonfinite or zero corrected probability mass")
    corrected = weighted / denominator
    return resize_probability(corrected, output_size)


# Compatibility names for experiment drivers.
prepare_inputs = prepare
head_probability = probability
initial_parameter_checks = parameter_checks
