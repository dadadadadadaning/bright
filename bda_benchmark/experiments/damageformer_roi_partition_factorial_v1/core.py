"""Change routing only. Original Soft-ROI content and FULL readout stay fixed.

G0/G1: hard home-cell grid, without/with predicted building split.
S0/S2: existing soft Q, without/with predicted building split.
H0/H1: optional argmax of that same Q, without/with building split.
No GT inputs, Memory, new loss, learned Q, pixel residual, or image cropping.
"""
import hashlib
import math
from pathlib import Path

import torch
from bda_benchmark.experiments.damageformer_roi_partition_control_v1.core import (
    capture, Readout, make_heads, pool, routing_stats,
    partition as original_partition,
    implementation_hash as parent_implementation_hash,
)
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import candidate_map

DEFAULT_VARIANTS = ('g0', 'g1', 's0', 's2')
VARIANTS = DEFAULT_VARIANTS + ('h0', 'h1')
DESCRIPTIONS = {
    'g0': 'home-cell hard grid, no building split',
    'g1': 'home-cell hard grid + predicted building/background split',
    's0': 'frozen learned SOFT Q, no building split (old S0)',
    's2': 'frozen learned SOFT Q + predicted building/background split (old S2)',
    'h0': 'argmax of frozen Q, no connectivity repair, no building split',
    'h1': 'argmax of frozen Q + predicted building/background split',
}


@torch.no_grad()
def partition(reference, mode, threshold=.5, stride=8):
    if mode not in VARIANTS:
        raise ValueError('Unknown factorial variant: '+str(mode))
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Building threshold must be finite and inside (0,1)')
    if not isinstance(stride, int) or stride < 1:
        raise ValueError('Require positive integer feature-grid stride')
    # Reuse precisely the old validated routing; never change the reference.
    base = original_partition(reference, 's0', threshold=threshold, stride=stride)
    q, ids, k = base['q'], base['ids'], base['k']
    b, _, n = q.shape
    h, w = base['shape']
    if h*w != n or reference['building'].shape != (b, 1, h, w):
        raise ValueError('Feature/Q/building shape mismatch; no implicit resize')
    expected, valid, gh, gw = candidate_map(h, w, stride, q.device)
    if ids.dtype != torch.long or ids.shape != (b, 9, n):
        raise ValueError('Require int64 native 9-candidate IDs')
    if k != gh*gw or not torch.equal(ids, expected[None].expand(b, -1, -1)):
        raise ValueError('Snapshot candidate map and stride disagree')
    if (q.masked_select(~valid[None].expand_as(q)) != 0).any():
        raise ValueError('Invalid edge candidates have nonzero association')
    if mode.startswith('g'):
        q = torch.zeros_like(q)
        q[:, 4, :] = 1.  # candidate_map center offset (dy=0, dx=0).
    elif mode.startswith('h'):
        # Argmax in original candidate order; IDs, not candidate index, identify regions.
        q = torch.zeros_like(q).scatter_(1, q.argmax(1, keepdim=True), 1.)
    if mode in ('g1', 's2', 'h1'):
        group = (reference['building'].detach().flatten(2) >= threshold).long()
        ids = ids * 2 + group
        k *= 2
    return dict(q=q, ids=ids, k=k, shape=(h, w))


def implementation_hash():
    """Archive protocol, diagnostics and imported readout/capture dependencies."""
    h = hashlib.sha256()
    h.update(parent_implementation_hash().encode())
    for p in sorted(Path(__file__).parent.glob('*')):
        if p.is_file() and p.suffix in ('.py', '.sh'):
            h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()
