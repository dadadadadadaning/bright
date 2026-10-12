"""Keep Q and the binary B decision fixed; change only cross-group weight.

Both hard and exchange use 18 local supports and 2*K slots, with exactly the
same readout parameters as the previous S2. GT never enters this module.
This is NOT a trainable partition generator, instance mask, or Memory model.
"""
import copy
import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_roi_partition_control_v1 import core as old
from bda_benchmark.experiments.damageformer_roi_partition_factorial_v1.core import (
    partition as checked_partition, implementation_hash as parent_hash,
)

capture = old.capture
DEFAULT_VARIANTS = ('hard', 'exchange')
VARIANTS = DEFAULT_VARIANTS


@torch.no_grad()
def partition(reference, mode, epsilon=.05, threshold=.5, stride=8):
    if mode not in VARIANTS:
        raise ValueError('Unknown exchange variant: '+str(mode))
    if not math.isfinite(epsilon) or not 0 <= epsilon <= .5:
        raise ValueError('epsilon must be finite and in [0, 0.5]')
    base = checked_partition(reference, 's0', threshold=threshold, stride=stride)
    q, ids, k = base['q'], base['ids'], base['k']
    effective = 0. if mode == 'hard' else float(epsilon)
    group = (reference['building'].detach().float().flatten(2) >= threshold).float()
    membership = effective + (1. - 2.*effective)*group
    weights = torch.cat([q*(1.-membership), q*membership], dim=1)
    support_ids = torch.cat([2*ids, 2*ids+1], dim=1)
    return dict(q=weights, ids=support_ids, k=2*k, shape=base['shape'],
                epsilon=effective, threshold=float(threshold))


def pool(values, q, ids, k):
    """FP32 weighted means. Supports are dynamic, NOT hard-coded to nine.

    Empty slots have zero means and zero contribution on unpool. The 1e-6
    denominator guard matches old S2; no thresholding/pruning is introduced.
    """
    if values.ndim != 3 or q.ndim != 3 or ids.shape != q.shape:
        raise ValueError('Expected values BNC and association/IDs BSN')
    b, n, c = values.shape
    if q.shape[0] != b or q.shape[2] != n or ids.dtype != torch.long or k <= 0:
        raise ValueError('Pool shape, index type, or slot count mismatch')
    values, q = values.float(), q.float()
    mass = values.new_zeros(b, k)
    sums = values.new_zeros(b, k, c)
    for j in range(q.shape[1]):
        mass = mass.scatter_add(1, ids[:, j], q[:, j])
        sums = sums.scatter_add(1, ids[:, j, :, None].expand(-1, -1, c),
                                values*q[:, j, :, None])
    return sums/mass.clamp_min(1e-6)[:, :, None], mass


def unpool(values, q, ids, shape, output_size):
    """Use the SAME routing as pooling; preserve the existing output resize."""
    b, _, c = values.shape
    n = q.shape[2]
    result = values.new_zeros(b, n, c)
    for j in range(q.shape[1]):
        result = result + values.gather(1, ids[:, j, :, None].expand(-1, -1, c))*q[:, j, :, None]
    result = result.transpose(1, 2).reshape(b, c, *shape)
    result = F.interpolate(result, output_size, mode='bilinear', align_corners=False)
    return result/result.sum(1, keepdim=True).clamp_min(1e-8)


class Readout(old.Readout):
    """State-dict compatible full readout; only the support axis is generalized."""
    def forward(self, feature, routing, output_size):
        f = feature.detach().float()
        q, ids, k = routing['q'].detach().float(), routing['ids'], routing['k']
        e = self.embedding
        weights = e.score(f).float().sigmoid().flatten(2)
        projected = e.project(f).float().flatten(2).transpose(1, 2)
        main, _ = pool(projected, q*weights, ids, k)
        stats, _ = pool(torch.cat([f, f.square()], 1).flatten(2).transpose(1, 2), q, ids, k)
        mean, second = stats.chunk(2, -1)
        std = (second-mean.square()).clamp_min(0).add(1e-6).sqrt()
        z = e.norm(e.main(main) + e.alpha.tanh()*e.stats(torch.cat([mean, std], -1)))
        prob = self.classifier(z).float().softmax(-1)
        return unpool(prob, q, ids, routing['shape'], output_size)


def make_heads(modes, dim, device):
    if not modes or len(set(modes)) != len(modes) or any(m not in VARIANTS for m in modes):
        raise ValueError('Expected unique hard/exchange modes')
    proto = Readout(dim).to(device)
    return {mode: copy.deepcopy(proto) for mode in modes}


@torch.no_grad()
def routing_stats(routing, building):
    q, ids, k = routing['q'], routing['ids'], routing['k']
    b, _, n = q.shape
    g = (building.detach().float().flatten(2).transpose(1, 2) >= routing.get('threshold', .5)).float()
    ratio, mass = pool(g, q, ids, k)
    ratio = ratio[..., 0].clamp(0, 1)  # floating roundoff only, not model routing
    hard_ids = ids.gather(1, q.argmax(1, keepdim=True)).squeeze(1)
    hard_mass = mass.new_zeros(b, k).scatter_add(1, hard_ids, torch.ones_like(hard_ids, dtype=mass.dtype))
    active = mass > 1e-6
    odd = torch.arange(k, device=q.device)[None] % 2 == 1
    foreign = torch.where(odd, 1-ratio, ratio)
    groups_fg, groups_bg = active & odd, active & ~odd
    def average(mask):
        # Mean over occupied slots, not pixels; tiny groups are reported separately.
        return float((foreign*mask).sum()/mask.sum().clamp_min(1))
    mismatched = ((ids % 2).float() != g[..., 0][:, None]).float()
    return dict(allocated=float(k), candidates=float(q.shape[1]),
                soft_occupied=float(active.sum(1).float().mean()),
                hard_occupied=float((hard_mass>0).sum(1).float().mean()),
                empty_groups=float((mass==0).sum(1).float().mean()),
                tiny_mass_groups=float(((mass>1e-6)&(mass<1)).sum(1).float().mean()),
                assignment_entropy=float(-(q*q.clamp_min(1e-8).log()).sum(1).mean()),
                predicted_binary_mixing=float((2*torch.minimum(ratio, 1-ratio)*mass).sum()/mass.sum().clamp_min(1)),
                cross_group_weight_fraction=float((q*mismatched).sum()/q.sum().clamp_min(1)),
                building_slots_pred_background_fraction_mean=average(groups_fg),
                background_slots_pred_building_fraction_mean=average(groups_bg),
                max_occupied_foreign_fraction=float(foreign.masked_fill(~active, 0).max()))


def verify_hard_parity(feature, reference, head, output_size, stride=8, threshold=.5):
    """Read-only real-input regression against old nine-support S2.

    Copies have identical state and no optimizer. Neither trained head gradients
    nor RNG nor frontend state are changed. Numerically close, not bitwise CUDA
    scatter determinism. Run once before any training update.
    """
    old_route = old.partition(reference, 's2', threshold=threshold, stride=stride)
    new_route = partition(reference, 'hard', threshold=threshold, stride=stride)
    # Bypass initialization: copying modules does not advance RNG.
    a, b = copy.deepcopy(head).eval(), copy.deepcopy(head).eval()
    a.requires_grad_(True); b.requires_grad_(True)
    with torch.enable_grad():
        before = old.Readout.forward(a, feature, old_route, output_size)
        after = b(feature, new_route, output_size)
        if not torch.isfinite(before).all() or not torch.isfinite(after).all():
            raise RuntimeError('Nonfinite hard parity output')
        torch.testing.assert_close(before, after, atol=2e-5, rtol=2e-4)
        # Fixed four-class NLL probe includes every output channel.
        h, w = output_size
        target = (torch.arange(h*w, device=before.device).reshape(1, h, w)%4).expand(len(before), -1, -1)
        def loss(prob):
            return -prob.gather(1, target[:, None]).clamp_min(1e-8).log().mean()
        ga = torch.autograd.grad(loss(before), tuple(a.parameters()), allow_unused=False)
        gb = torch.autograd.grad(loss(after), tuple(b.parameters()), allow_unused=False)
        max_grad = 0.
        for (name, _), x, y in zip(a.named_parameters(), ga, gb):
            torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-4, msg=name)
            max_grad = max(max_grad, float((x-y).abs().max()))
        return dict(passed=True, prob_max_abs=float((before-after).abs().max().detach()),
                    gradient_max_abs=max_grad, atol=2e-5, rtol=2e-4,
                    scope='same real batch; old S2 nine supports versus new hard eighteen supports')


def implementation_hash():
    digest = hashlib.sha256(parent_hash().encode())
    for path in sorted(Path(__file__).parent.glob('*')):
        if path.is_file() and path.suffix in ('.py', '.sh'):
            digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()
