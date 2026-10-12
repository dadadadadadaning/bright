"""Separate gathering from reading without altering old experiment modules.

All routes share 2K global slot IDs and 18 local support entries. No GT enters
these functions. Feature inputs are frozen, but Q gradients are deliberately
preserved for the separate Q-learning experiment.
"""
import math
import torch
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import pool, unpool


def route_from_q(q, ids, k, shape, building, epsilon=0., threshold=.5):
    if not math.isfinite(epsilon) or not 0 <= epsilon <= .5 or not 0 < threshold < 1:
        raise ValueError('Invalid epsilon/threshold')
    if q.ndim != 3 or q.shape[1] != 9 or k <= 0:
        raise ValueError('Expected B,9,N base Q and positive K')
    b, _, n = q.shape
    if tuple(building.shape) != (b, 1, *shape) or n != shape[0]*shape[1]:
        raise ValueError('Building and Q must share native feature coordinates')
    if ids.ndim == 2:
        ids = ids[None].expand(b, -1, -1)
    if ids.shape != q.shape or ids.dtype != torch.long or (ids < 0).any() or (ids >= k).any():
        raise ValueError('Invalid candidate IDs')
    if not torch.isfinite(q).all() or (q < 0).any() or not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-5):
        raise ValueError('Q must be finite, nonnegative and normalized')
    building = building.detach().float()
    if not torch.isfinite(building).all() or (building < 0).any() or (building > 1).any():
        raise ValueError('Invalid predicted building probabilities')
    group = (building.flatten(2) >= threshold).float()
    u = epsilon + (1 - 2*epsilon)*group
    return dict(q=torch.cat([q.float()*(1-u), q.float()*u], 1),
                ids=torch.cat([2*ids, 2*ids+1], 1), k=2*k,
                shape=tuple(shape), epsilon=float(epsilon), threshold=float(threshold))


def encode_regions(head, feature, route):
    f = feature.detach().float()
    q, ids, k = route['q'].float(), route['ids'], route['k']
    e = head.embedding
    attention = e.score(f).float().sigmoid().flatten(2)
    projected = e.project(f).float().flatten(2).transpose(1, 2)
    main, attention_mass = pool(projected, q*attention, ids, k)
    stats, mass = pool(torch.cat([f, f.square()], 1).flatten(2).transpose(1, 2), q, ids, k)
    mean, second = stats.chunk(2, -1)
    std = (second-mean.square()).clamp_min(0).add(1e-6).sqrt()
    tokens = e.norm(e.main(main) + e.alpha.tanh()*e.stats(torch.cat([mean, std], -1)))
    return dict(tokens=tokens, mass=mass, attention_mass=attention_mass)


def valid_read_route(read_route, mass):
    """Mask strictly empty gather slots, then renormalize BEFORE interpolation.

    Tiny positive slots are retained: an epsilon guard is not a purity filter.
    No fallback silently assigns a pixel to an unsupported slot.
    """
    q, ids = read_route['q'].float(), read_route['ids']
    if mass.shape != (q.shape[0], read_route['k']):
        raise ValueError('Gather mass/read slot mismatch')
    if not torch.isfinite(mass).all() or (mass < 0).any():
        raise ValueError('Invalid gather mass')
    supported = torch.stack([mass.gather(1, ids[:, j]) > 0 for j in range(ids.shape[1])], 1)
    weights = q*supported
    denominator = weights.sum(1, keepdim=True)
    if not torch.isfinite(q).all() or (q < 0).any() or (denominator <= 0).any():
        raise ValueError('Invalid read weight or no reachable nonempty slot')
    return dict(read_route, q=weights/denominator)


def decode_regions(logits, read_route, mass, output_size):
    if logits.ndim != 3 or logits.shape[:2] != mass.shape or logits.shape[-1] != 4:
        raise ValueError('Expected B,K,4 logits and B,K mass')
    route = valid_read_route(read_route, mass)
    return unpool(logits.float().softmax(-1), route['q'], route['ids'], route['shape'], output_size)


def forward_separate(head, feature, pool_route, read_route, output_size):
    if pool_route['k'] != read_route['k'] or pool_route['shape'] != read_route['shape'] or not torch.equal(pool_route['ids'], read_route['ids']):
        raise ValueError('Pool/read routes must refer to identical ordered slot IDs')
    encoded = encode_regions(head, feature, pool_route)
    return dict(encoded, prob=decode_regions(head.classifier(encoded['tokens']), read_route, encoded['mass'], output_size))


@torch.no_grad()
def route_diagnostics(pool_route, read_route, mass, attention_mass, building):
    adjusted = valid_read_route(read_route, mass)
    before, after = read_route['q'].float(), adjusted['q']
    group = (building.detach().flatten(2) >= read_route['threshold']).long()
    opposite = (read_route['ids'] % 2 != group)
    supported = torch.stack([mass.gather(1, read_route['ids'][:, j]) > 0 for j in range(before.shape[1])], 1)
    def avg_count(mask):
        return float(mask.sum(1).float().mean())
    return dict(allocated=float(pool_route['k']),
                occupied=avg_count(mass > 0), occupied_gt_1e6=avg_count(mass > 1e-6),
                strictly_empty=avg_count(mass == 0), tiny_positive_le_1e6=avg_count((mass > 0)&(mass <= 1e-6)),
                attention_guard_slots=avg_count(attention_mass < 1e-6),
                stats_guard_slots=avg_count(mass < 1e-6),
                blocked_read_mass_fraction=float((before*~supported).sum()/before.sum()),
                requested_cross_read_fraction=float((before*opposite).sum()/before.sum()),
                actual_cross_read_fraction=float((after*opposite).sum()/after.sum()),
                pixels_without_opposite_support=float((~(supported&opposite).any(1)).float().mean()))
