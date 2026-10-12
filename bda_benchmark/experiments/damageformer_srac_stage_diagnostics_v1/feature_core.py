"""Fixed-F/Q affine probes. No GT enters any prediction function."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .shared import close_report
from .grouped_pool import GroupedRouting, POOL_IMPLEMENTATION


def validate_routes(q, ids, count, shape):
    if q.ndim != 3 or ids.ndim != 2 or ids.dtype != torch.long:
        raise ValueError("Expected Q[B,S,N], ids[S,N] with long IDs")
    if type(count) is not int or count < 1 or len(shape) != 2 or min(shape) < 1:
        raise ValueError("Invalid region count or grid shape")
    if tuple(ids.shape) != tuple(q.shape[1:]) or q.shape[-1] != shape[0]*shape[1]:
        raise ValueError("Q, IDs and grid shape disagree")
    if not q.is_floating_point() or min(q.shape) < 1 or q.device != ids.device:
        raise ValueError("Invalid Q/device")
    if not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
        raise ValueError("Q must be finite and nonnegative")
    if bool((ids < 0).any()) or bool((ids >= count).any()):
        raise ValueError("Out-of-range region ID")
    check = close_report(q.sum(1), torch.ones_like(q[:, 0]), atol=2e-5, rtol=2e-5)
    if not check['passed']:
        raise ValueError("Q is not normalized: " + str(check))
    return check


def mean_pool(feature, q, ids, count, *, routing=None):
    """Exact normalized mean for positive mass, zero token for exactly empty mass.

    This is the diagnostic mathematical mean, not the parent's attention/epsilon
    pooling. Tiny positive masses are NOT clamped to 1e-6; they are reported.
    Duplicate region IDs are accumulated, not treated as different regions.
    """
    if feature.ndim != 4 or feature.shape[0] != q.shape[0]:
        raise ValueError("Expected feature[B,C,H,W]")
    validate_routes(q, ids, count, feature.shape[-2:])
    if feature.device != q.device or not bool(torch.isfinite(feature).all()):
        raise ValueError("Nonfinite feature or device mismatch")
    plan = GroupedRouting(ids, count) if routing is None else routing
    return plan.pool(feature, q, ids, count)


def unpool(tokens, q, ids, shape):
    validate_routes(q, ids, tokens.shape[1], shape)
    if tokens.ndim != 3 or tokens.shape[0] != q.shape[0] or tokens.device != q.device:
        raise ValueError("Expected matching tokens[B,K,C]")
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("Nonfinite tokens")
    result = tokens.new_zeros(tokens.shape[0], q.shape[-1], tokens.shape[-1])
    for s in range(q.shape[1]):
        result.add_(tokens[:, ids[s]] * q[:, s, :, None].to(tokens.dtype))
    return result.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], *shape)


def resize_probability(probability, output_size):
    p = F.interpolate(probability, output_size, mode="bilinear", align_corners=False)
    # Same output normalization convention as the parent probability projector.
    return p / p.sum(1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def feature_branches(feature, q, ids, count, head, output_size, *, progress=None):
    def mark(name):
        if progress is not None:
            progress(name)
    if not isinstance(head, nn.Conv2d) or head.kernel_size != (1, 1) or head.groups != 1:
        raise ValueError("This audit requires the original affine 1x1 classifier")
    if head.stride != (1, 1) or head.padding != (0, 0) or head.out_channels != 4:
        raise ValueError("Classifier must preserve the grid and output four classes")
    if feature.shape[1] != head.in_channels:
        raise ValueError("Do not feed the 128-d adapter into the 256-d A1 head")
    shape = feature.shape[-2:]
    validate_routes(q, ids, count, shape)
    mark('build_grouped_route_plan')
    routing = GroupedRouting(ids, count)
    mark('mean_pool_F')
    token, mass = mean_pool(feature, q, ids, count, routing=routing)
    mark('unpool_F_and_classify')
    reconstructed = unpool(token, q, ids, shape)
    direct_logits = head(feature)
    after_logits = head(reconstructed)
    token_logits = F.linear(token, head.weight[:, :, 0, 0], head.bias)
    mark('mean_pool_logits')
    logit_token, _ = mean_pool(direct_logits, q, ids, count, routing=routing)
    logit_rebuilt = unpool(logit_token, q, ids, shape)
    mark('mean_pool_constants')
    one_token, _ = mean_pool(torch.ones_like(feature[:, :1]), q, ids, count, routing=routing)
    constants = unpool(one_token, q, ids, shape)
    checks = dict(
        affine_commutation=close_report(after_logits, logit_rebuilt),
        token_affine_commutation=close_report(token_logits[mass > 0], logit_token[mass > 0]),
        constant_reconstruction=close_report(constants, torch.ones_like(constants)),
        q_normalization=close_report(q.sum(1), torch.ones_like(q[:, 0])),
    )
    # Empty tokens need not satisfy affine bias equality because no pixels use them.
    mark('mean_pool_probabilities')
    prob_token, _ = mean_pool(direct_logits.softmax(1), q, ids, count, routing=routing)
    mark('project_feature_branches')
    grids = dict(
        pixel_grid=direct_logits.softmax(1),
        feature_mean=after_logits.softmax(1),
        region_mean=unpool(token_logits.softmax(-1), q, ids, shape),
        grid_probq=unpool(prob_token, q, ids, shape),
    )
    probabilities = {name: resize_probability(p, output_size) for name, p in grids.items()}
    checks.update(empty_regions=int((mass == 0).sum()),
                  tiny_positive_regions=int(((mass > 0) & (mass < 1e-6)).sum()),
                  pool_implementation=POOL_IMPLEMENTATION,
                  mean_pool_denominator="exact positive mass; one only for zero mass")
    mark('feature_branches_complete')
    return probabilities, checks, (direct_logits, after_logits)


def damaged_margin_sums(before, after, target):
    """Offline GT-area-weighted feature-grid D-minus-I margin, not a new predictor."""
    if before.shape != after.shape or before.shape[1] != 4:
        raise ValueError("Margin logits disagree")
    weight = F.interpolate((target == 2).float()[:, None], before.shape[-2:], mode="area")[:, 0]
    a, b = before[:, 2]-before[:, 1], after[:, 2]-after[:, 1]
    return dict(damaged_area_feature_cells=float(weight.sum()),
                before_margin_sum=float((a*weight).sum()), after_margin_sum=float((b*weight).sum()),
                positive_to_nonpositive_mass=float((weight*((a > 0) & (b <= 0))).sum()),
                nonpositive_to_positive_mass=float((weight*((a <= 0) & (b > 0))).sum()))
