"""Isolated partition-learning primitives; no GT enters assignment generation.

The structural loss is evaluated on UNSPLIT Q, before predicted ROI routing.
Dual assignment compares a position with a center *within each modality*;
it never interprets optical minus SAR descriptors as physical change.
"""
import math

import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_iterative_partition_v1.model import grouped_centers


def _local_indices(ids, batch, locations, count, device):
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('Region count must be a positive integer')
    if ids.dtype != torch.long or ids.device != device:
        raise ValueError('Candidate IDs must be int64 on the descriptor/Q device')
    if ids.shape == (9, locations):
        ids = ids[None].expand(batch, -1, -1)
    if ids.shape != (batch, 9, locations):
        raise ValueError('Candidate IDs must have shape [9,N] or [B,9,N]')
    if bool(((ids < 0) | (ids >= count)).any()):
        raise ValueError('Candidate IDs must be in [0, region count)')
    return ids


def structure_reconstruction_loss(q, ids, k, shape, gt, ignore_index=255):
    """Binary structure reconstruction with valid-area weighting, FP32.

    q: [B,9,h*w], row-normalized UNSPLIT associations.
    ids: [9,h*w] or [B,9,h*w], k: allocated seed count.
    shape: native (h,w) feature grid; gt: integer [B,H,W] in 0..3/ignore.

    Foreground and valid masks are area-downsampled independently. Ignored
    area contributes neither to the regional target nor the loss denominator.
    Empty regions are excluded, but positive tiny masses are NOT discarded or
    clamped to an arbitrary occupancy threshold. GT is supervision only.
    """
    if q.ndim != 3 or q.shape[1] != 9 or not q.is_floating_point():
        raise ValueError('Unsplit Q must be floating [B,9,N]')
    b, _, n = q.shape
    if len(shape) != 2 or any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in shape):
        raise ValueError('shape must be a positive integer (h,w)')
    h, w = shape
    if h*w != n or gt.ndim != 3 or gt.shape[0] != b:
        raise ValueError('Q, native shape and GT batch must agree')
    if gt.device != q.device or gt.is_floating_point() or gt.is_complex():
        raise ValueError('GT must be an integer label tensor on the Q device')
    if gt.shape[-2] < h or gt.shape[-1] < w:
        raise ValueError('GT resolution must not be smaller than the native feature grid')
    index = _local_indices(ids, b, n, k, q.device)
    with torch.autocast(device_type=q.device.type, enabled=False):
        weights = q.float()
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError('Q must contain finite, nonnegative weights')
        if not torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]), atol=1e-5, rtol=1e-5):
            raise ValueError('Q must sum to one at every native feature location')
        valid_gt = gt != ignore_index
        if bool((valid_gt & ((gt < 0) | (gt > 3))).any()):
            raise ValueError('Valid GT must use BRIGHT classes 0..3')
        valid_area = F.interpolate(valid_gt[:, None].float(), size=(h, w), mode='area').flatten(1)
        fg_area = F.interpolate((valid_gt & (gt > 0))[:, None].float(), size=(h, w), mode='area').flatten(1)
        total_valid = valid_area.sum()
        if not bool(total_valid > 0):
            return weights.sum()*0.

        mass = weights.new_zeros(b, k)
        foreground_sum = weights.new_zeros(b, k)
        for j in range(9):
            mass = mass.scatter_add(1, index[:, j], weights[:, j]*valid_area)
            foreground_sum = foreground_sum.scatter_add(1, index[:, j], weights[:, j]*fg_area)
        occupied = mass > 0
        safe_mass = torch.where(occupied, mass, torch.ones_like(mass))
        center_foreground = foreground_sum/safe_mass

        reconstructed = weights.new_zeros(b, n)
        read_mass = weights.new_zeros(b, n)
        for j in range(9):
            readable = occupied.gather(1, index[:, j]).to(weights.dtype)
            read_weight = weights[:, j]*readable
            reconstructed = reconstructed + read_weight*center_foreground.gather(1, index[:, j])
            read_mass = read_mass + read_weight
        active = valid_area > 0
        if bool((active & (read_mass <= 0)).any()):
            raise RuntimeError('Valid GT area has no occupied region to reconstruct from')
        reconstructed = reconstructed/torch.where(read_mass > 0, read_mass, torch.ones_like(read_mass))
        target = fg_area/torch.where(active, valid_area, torch.ones_like(valid_area))
        # Clamp only floating-point roundoff at probability bounds, never mass.
        # Selecting valid positions prevents ignored targets affecting the loss.
        per_location = F.binary_cross_entropy(reconstructed[active].clamp(0., 1.), target[active], reduction='none')
        return (per_location*valid_area[active]).sum()/total_valid


def dual_assignment(rgb_embedding, sar_or_rgb_embedding, ids, valid, k, stride,
                    iterations=3, beta=.2, temperature=.2, spatial_weight=.1):
    """Nine-candidate differentiable local clustering in two descriptor spaces.

    Embeddings are [B,C_r,h,w] / [B,C_s,h,w]; channels may differ. The native
    feature grid is shared, not an optical--SAR alignment/search window.
    beta is a fixed mixture weight, NOT a calibrated reliability estimate.

    Center normalization / 1e-6 occupancy match the existing generator so
    beta=0 reproduces iterative_assignment(..., building=zeros). No gradient
    is detached through descriptors, assignments or center updates.
    """
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError('iterations must be a positive integer')
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError('stride must be a positive integer')
    if (not math.isfinite(beta) or not 0 <= beta <= 1 or
            not math.isfinite(temperature) or temperature <= 0 or
            not math.isfinite(spatial_weight) or spatial_weight < 0):
        raise ValueError('Invalid beta, temperature or spatial weight')
    if rgb_embedding.ndim != 4 or sar_or_rgb_embedding.ndim != 4:
        raise ValueError('Both embeddings must be [B,C,h,w]')
    b, cr, h, w = rgb_embedding.shape
    if min(b, cr, h, w, sar_or_rgb_embedding.shape[1]) < 1:
        raise ValueError('Embedding dimensions must be nonempty')
    if (sar_or_rgb_embedding.shape[0] != b or sar_or_rgb_embedding.shape[-2:] != (h, w) or
            sar_or_rgb_embedding.device != rgb_embedding.device):
        raise ValueError('Both descriptor branches must share batch, native grid and device')
    if not rgb_embedding.is_floating_point() or not sar_or_rgb_embedding.is_floating_point():
        raise ValueError('Embeddings must be floating point')
    index = _local_indices(ids, b, h*w, k, rgb_embedding.device)
    if valid.dtype != torch.bool or valid.device != rgb_embedding.device:
        raise ValueError('Candidate mask must be bool on the descriptor device')
    if valid.shape == (9, h*w):
        valid = valid[None].expand(b, -1, -1)
    if valid.shape != (b, 9, h*w) or not bool(valid[:, 4].all()):
        raise ValueError('Mask must match candidates and every home candidate must be valid')

    with torch.autocast(device_type=rgb_embedding.device.type, enabled=False):
        if not bool(torch.isfinite(rgb_embedding).all()) or not bool(torch.isfinite(sar_or_rgb_embedding).all()):
            raise ValueError('Embeddings must be finite')
        rgb = F.normalize(rgb_embedding.float(), dim=1).flatten(2).transpose(1, 2)
        sar = F.normalize(sar_or_rgb_embedding.float(), dim=1).flatten(2).transpose(1, 2)
        cs = sar.shape[-1]
        yy, xx = torch.meshgrid(torch.arange(h, device=rgb.device), torch.arange(w, device=rgb.device), indexing='ij')
        xy = torch.stack((yy, xx), -1).reshape(1, h*w, 2).float()/stride
        xy = xy.expand(b, -1, -1)
        values = torch.cat((rgb, sar, xy), -1)
        q = rgb.new_zeros(b, 9, h*w)
        q[:, 4] = 1.
        for _ in range(iterations):
            centers, mass = grouped_centers(values, q, index, k)
            logits = []
            for j in range(9):
                center = centers.gather(1, index[:, j, :, None].expand(-1, -1, cr+cs+2))
                rgb_similarity = (rgb*F.normalize(center[..., :cr], dim=-1)).sum(-1)
                sar_similarity = (sar*F.normalize(center[..., cr:cr+cs], dim=-1)).sum(-1)
                similarity = (1.-beta)*rgb_similarity + beta*sar_similarity
                distance = (xy-center[..., -2:]).square().sum(-1)
                score = (similarity-spatial_weight*distance)/temperature
                occupied = mass.gather(1, index[:, j]) > 1e-6
                logits.append(score.masked_fill(~(valid[:, j] & occupied), -torch.inf))
            scores = torch.stack(logits, 1)
            if bool((~torch.isfinite(scores).any(1)).any()):
                raise RuntimeError('A position lost all occupied candidate centers')
            q = scores.softmax(1)
        return q
