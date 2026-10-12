"""Observation-only Q means without colliding CUDA scatter/indexed reductions.

Only integer route planning is on CPU. Features, Q weights and reductions stay
on the caller's device. No exception retry, automatic CPU fallback or dense NxK.
"""
from __future__ import annotations

import torch


POOL_IMPLEMENTATION = 'grouped_gather_sum_v1'


class GroupedRouting:
    """Stable per-region list of (candidate, pixel) occurrences, including duplicates.

    Work tensors are bounded by both region blocks and route chunks. A highly
    concentrated partition therefore does not allocate K*max_region_size*C.
    This is an inference audit operator, not a replacement training layer.
    """

    def __init__(self, ids, count, *, max_work_elements=2**22, regions_per_block=32):
        if ids.ndim != 2 or ids.dtype != torch.long or min(ids.shape) < 1:
            raise ValueError('Routing IDs must be nonempty long [S,N]')
        if type(count) is not int or count < 1:
            raise ValueError('Region count must be a positive integer')
        if type(max_work_elements) is not int or max_work_elements < 1:
            raise ValueError('max_work_elements must be a positive integer')
        if type(regions_per_block) is not int or regions_per_block < 1:
            raise ValueError('regions_per_block must be a positive integer')
        self.ids = ids.detach().cpu().contiguous().clone()
        if bool(((self.ids < 0) | (self.ids >= count)).any()):
            raise ValueError('Out-of-range routing ID')
        self.count = count
        self.max_work_elements = max_work_elements
        self.regions_per_block = regions_per_block
        flat = self.ids.reshape(-1)
        # Sorting metadata never changes Q or removes duplicate slot contributions.
        self.order = torch.argsort(flat, stable=True)
        lengths = torch.bincount(flat, minlength=count)
        self.lengths = lengths.tolist()
        self.starts = (lengths.cumsum(0)-lengths).tolist()
        self.nonempty = [k for k, length in enumerate(self.lengths) if length]

    def pool(self, feature, q, ids, count):
        if count != self.count or not torch.equal(ids.detach().cpu(), self.ids):
            raise ValueError('Routing plan belongs to different IDs/count')
        if feature.ndim != 4 or q.ndim != 3 or q.shape[0] != feature.shape[0]:
            raise ValueError('Expected F[B,C,h,w], Q[B,S,h*w]')
        batch, channels, height, width = feature.shape
        if tuple(q.shape[1:]) != tuple(self.ids.shape) or height*width != self.ids.shape[1]:
            raise ValueError('Feature, Q and routing shapes disagree')
        if feature.device != q.device or ids.device != q.device:
            raise ValueError('Feature, Q and IDs must share the caller device')
        if batch*channels > self.max_work_elements:
            raise ValueError('Work budget is smaller than one batch-by-channel contribution')
        dtype = torch.float64 if feature.dtype == torch.float64 else torch.float32
        rows = feature.to(dtype).flatten(2).transpose(1, 2).contiguous()
        weights = q.to(dtype).reshape(batch, -1).contiguous()
        numerator = rows.new_zeros(batch, count, channels)
        mass = rows.new_zeros(batch, count)
        regions_per_block = min(self.regions_per_block, self.max_work_elements // (batch*channels))
        for block_start in range(0, len(self.nonempty), regions_per_block):
            group = self.nonempty[block_start:block_start+regions_per_block]
            nregion = len(group)
            longest = max(self.lengths[k] for k in group)
            route_width = min(longest, self.max_work_elements // (batch*nregion*channels))
            block_numerator = rows.new_zeros(batch, nregion, channels)
            block_mass = rows.new_zeros(batch, nregion)
            for offset in range(0, longest, route_width):
                size = min(route_width, longest-offset)
                # Padding refers to a valid source index with explicitly zero weight.
                route = torch.zeros(nregion, size, dtype=torch.long, device='cpu')
                present = torch.zeros(nregion, size, dtype=torch.bool, device='cpu')
                for j, k in enumerate(group):
                    take = min(size, max(0, self.lengths[k]-offset))
                    if take:
                        start = self.starts[k]+offset
                        route[j, :take] = self.order[start:start+take]
                        present[j, :take] = True
                route_device = route.reshape(-1).to(rows.device)
                keep = present.to(rows.device)[None]
                pixel = route_device.remainder(height*width)
                w = weights.index_select(1, route_device).reshape(batch, nregion, size)
                w = torch.where(keep, w, torch.zeros_like(w))
                values = rows.index_select(1, pixel).reshape(batch, nregion, size, channels)
                block_numerator = block_numerator + (values*w[..., None]).sum(2)
                block_mass = block_mass + w.sum(2)
            # Disjoint plain slices: no advanced indexed accumulation kernel.
            for j, k in enumerate(group):
                numerator[:, k:k+1, :] = block_numerator[:, j:j+1, :]
                mass[:, k:k+1] = block_mass[:, j:j+1]
        denominator = torch.where(mass > 0, mass, torch.ones_like(mass))
        tokens = numerator / denominator[..., None]
        if not bool(torch.isfinite(tokens).all()) or not bool(torch.isfinite(mass).all()):
            raise FloatingPointError('Nonfinite grouped mean/mass')
        return tokens, mass
