"""S0 original Q; S1 soft compatibility; S2 hard ROI; C2 spatial split.

All variants consume the SAME original Soft-ROI content. No GT enters routing.
These are fixed association interventions with fresh trained full readouts,
not jointly learned partition generators or building-instance segmentation.
"""
import copy
import hashlib
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bda_benchmark.experiments.damageformer_modulation_controls_v1.core import capture as capture_content
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.model import RegionEmbedding

VARIANTS = ('s0', 's1', 's2', 'c2')


@torch.no_grad()
def capture(model, pre, post):
    # Capture native-resolution logits BEFORE the old forward upsamples them.
    loc = []
    handle = model.base.clf_1.register_forward_hook(lambda m, i, o: loc.append(o))
    try:
        r = capture_content(model, pre, post)
    finally:
        handle.remove()
    if len(loc) != 1:
        raise RuntimeError('Expected exactly one building-head call')
    building = loc[0].float().softmax(1)[:, 1:2].detach()
    if building.shape[-2:] != r['shape']:
        raise RuntimeError('Building/Q resolution mismatch; no implicit resampling allowed')
    r['building'] = building
    return r


def pool(values, q, ids, k):
    """values B,N,C; q/ids B,9,N. FP32 scatter, independent per image."""
    b, n, c = values.shape
    mass = values.new_zeros(b, k)
    sums = values.new_zeros(b, k, c)
    for j in range(9):
        mass = mass.scatter_add(1, ids[:, j], q[:, j])
        sums = sums.scatter_add(1, ids[:, j, :, None].expand(-1, -1, c),
                                values * q[:, j, :, None])
    return sums / mass.clamp_min(1e-6)[:, :, None], mass


@torch.no_grad()
def partition(reference, mode, strength=1., threshold=.5, stride=8):
    if mode not in VARIANTS or not math.isfinite(strength) or strength < 0:
        raise ValueError('Invalid mode or soft strength')
    if not 0 < threshold < 1 or stride < 1:
        raise ValueError('Invalid threshold/stride')
    q = reference['q'].detach().float()
    b, nine, n = q.shape
    if nine != 9 or not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError('Invalid Q')
    if not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-5):
        raise ValueError('Q must be normalized per location')
    ids = reference['ids'][None].expand(b, -1, -1)
    k = reference['tokens'].shape[1]
    building = reference['building'].detach().float().flatten(2).transpose(1, 2)
    if building.shape != (b, n, 1) or not torch.isfinite(building).all():
        raise ValueError('Invalid building probabilities')
    if (building < 0).any() or (building > 1).any():
        raise ValueError('Building probabilities outside [0,1]')
    if mode == 's1' and strength != 0:
        centers, _ = pool(building, q, ids, k)
        center_b = torch.stack([centers[..., 0].gather(1, ids[:, j]) for j in range(9)], 1)
        penalty = (building[..., 0][:, None] - center_b).abs()
        # Preserve exact zeros: invalid/empty candidates must never be revived.
        logq = q.clamp_min(torch.finfo(q.dtype).tiny).log().masked_fill(q == 0, -torch.inf)
        q = (logq - strength * penalty).softmax(1)
    elif mode == 's2':
        group = (building[..., 0] >= threshold).long()[:, None]
        ids = ids * 2 + group
        k *= 2
    elif mode == 'c2':
        # Per-candidate split at that seed cell's x midpoint. No B is used.
        h, w = reference['shape']
        gw = (w + stride - 1) // stride
        xx = torch.arange(w, device=q.device).expand(h, w).reshape(1, 1, n)
        seed_x = (ids % gw) * stride
        right = (seed_x + stride).clamp_max(w)
        group = (xx.float() + .5 >= (seed_x + right).float() / 2).long()
        ids = ids * 2 + group
        k *= 2
    return dict(q=q, ids=ids, k=k, shape=reference['shape'])


class Readout(nn.Module):
    """Same parameters/operations as V2 RegionEmbedding; supports batched IDs."""
    def __init__(self, dim=128):
        super().__init__()
        self.embedding = RegionEmbedding(channels=256, dim=dim)
        self.classifier = nn.Linear(dim, 4)

    def forward(self, feature, routing, output_size):
        f = feature.detach().float()
        q, ids, k = routing['q'].detach(), routing['ids'], routing['k']
        e = self.embedding
        weights = e.score(f).float().sigmoid().flatten(2)
        projected = e.project(f).float().flatten(2).transpose(1, 2)
        main, _ = pool(projected, q * weights, ids, k)
        stats, _ = pool(torch.cat([f, f.square()], 1).flatten(2).transpose(1, 2), q, ids, k)
        mean, second = stats.chunk(2, -1)
        std = (second-mean.square()).clamp_min(0).add(1e-6).sqrt()
        z = e.norm(e.main(main) + e.alpha.tanh()*e.stats(torch.cat([mean, std], -1)))
        prob = self.classifier(z).float().softmax(-1)
        b, _, n = q.shape
        result = prob.new_zeros(b, n, 4)
        for j in range(9):
            result = result + prob.gather(1, ids[:, j, :, None].expand(-1, -1, 4))*q[:, j, :, None]
        result = result.transpose(1, 2).reshape(b, 4, *routing['shape'])
        result = F.interpolate(result, output_size, mode='bilinear', align_corners=False)
        return result / result.sum(1, keepdim=True).clamp_min(1e-8)


def make_heads(modes, dim, device):
    proto = Readout(dim).to(device)
    return {mode: copy.deepcopy(proto) for mode in modes}


@torch.no_grad()
def routing_stats(routing, building):
    q, ids, k = routing['q'], routing['ids'], routing['k']
    b, _, n = q.shape
    values = building.float().flatten(2).transpose(1, 2)
    mean, mass = pool(values, q, ids, k)
    hard = ids.gather(1, q.argmax(1, keepdim=True)).squeeze(1)
    hard_mass = mass.new_zeros(b, k).scatter_add(1, hard, torch.ones_like(hard, dtype=mass.dtype))
    g = (values >= .5).float()
    ratio, _ = pool(g, q, ids, k)
    # Predicted-group mixing only: not GT purity and not a task-quality score.
    mixing = (2*torch.minimum(ratio[..., 0], 1-ratio[..., 0])*mass).sum()/mass.sum().clamp_min(1)
    return dict(allocated=float(k), soft_occupied=float((mass > 1e-6).sum(1).float().mean()),
                hard_occupied=float((hard_mass > 0).sum(1).float().mean()),
                tiny_mass_groups=float(((mass > 1e-6) & (mass < 1)).sum(1).float().mean()),
                assignment_entropy=float(-(q*q.clamp_min(1e-8).log()).sum(1).mean()),
                predicted_binary_mixing_at_05=float(mixing))


def implementation_hash():
    repo = Path(__file__).resolve().parents[3]
    paths = list(Path(__file__).parent.glob('*.py'))
    for folder in ('damageformer_region_content_v1', 'damageformer_modulation_controls_v1',
                   'damageformer_region_evidence_v1'):
        paths += list((repo/'bda_benchmark/experiments'/folder).glob('*.py'))
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.relative_to(repo).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()
