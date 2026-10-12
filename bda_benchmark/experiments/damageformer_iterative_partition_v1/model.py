"""Local differentiable center refinement, inspired by SSN, not a reproduction.

Same RGB+XY association parameter count and pixel residual readout as old T3.
Only predicted building groups enter the forward pass; never GT masks.
"""
import torch
from torch.nn import functional as F
from bda_benchmark.experiments.damageformer_joint_region_v1.model import (
    JointHead, JointModel, load_encoder_pretrain, pool_context,
    candidate_map, normalized_coordinate_grid, masked_local_softmax,
)

VARIANTS = ('pixel', 'single', 'iter1', 'iter3', 'grid')


def grouped_centers(values, q, index, count):
    """values [B,N,D], q/index [B,9,N]. Differentiable FP32 reductions."""
    b, n, d = values.shape
    mass = values.new_zeros(b, count)
    sums = values.new_zeros(b, count, d)
    for j in range(9):
        mass = mass.scatter_add(1, index[:, j], q[:, j])
        sums = sums.scatter_add(1, index[:, j, :, None].expand(-1, -1, d),
                                values * q[:, j, :, None])
    return sums / mass.clamp_min(1e-6)[:, :, None], mass


def iterative_assignment(embedding, ids, valid, building, count, stride,
                         iterations=3, temperature=0.2, spatial_weight=0.1):
    """Local soft clustering; no detached centers or no_grad assignment loop.

    Empty centers cannot attract pixels. Each pixel's home group starts occupied,
    so at least one valid candidate exists even for 1x1/partial edge cells.
    Nine candidates refer to seed cells, NOT cross-modal alignment offsets.
    """
    if iterations < 1 or temperature <= 0 or spatial_weight < 0:
        raise ValueError('Invalid clustering configuration')
    b, _, h, w = embedding.shape
    with torch.autocast(device_type=embedding.device.type, enabled=False):
        e = F.normalize(embedding.float(), dim=1).flatten(2).transpose(1, 2)
        yy, xx = torch.meshgrid(torch.arange(h, device=e.device),
                                torch.arange(w, device=e.device), indexing='ij')
        xy = torch.stack((yy, xx), -1).reshape(1, h*w, 2).float() / stride
        xy = xy.expand(b, -1, -1)
        index = ids[None].expand(b, -1, -1)*2 + (building.detach().flatten(2) >= .5).long()
        q = e.new_zeros(b, 9, h*w)
        q[:, 4] = 1.
        values = torch.cat((e, xy), -1)
        for _ in range(iterations):
            centers, mass = grouped_centers(values, q, index, 2*count)
            logits = []
            for j in range(9):
                center = centers.gather(1, index[:, j, :, None].expand(-1, -1, values.shape[-1]))
                similarity = (e * F.normalize(center[..., :-2], dim=-1)).sum(-1)
                distance = (xy-center[..., -2:]).square().sum(-1)
                score = (similarity-spatial_weight*distance)/temperature
                occupied = mass.gather(1, index[:, j]) > 1e-6
                score = score.masked_fill(~(valid[j][None] & occupied), -torch.inf)
                logits.append(score)
            q = torch.stack(logits, 1).softmax(1)
        return q


class PartitionHead(JointHead):
    def __init__(self, variant, channels=256, dim=64, stride=8,
                 temperature=.2, spatial_weight=.1):
        if variant not in VARIANTS or variant == 'pixel':
            raise ValueError(variant)
        super().__init__('t2' if variant == 'grid' else 't3', channels, dim, stride)
        self.partition_variant = variant
        self.temperature, self.spatial_weight = temperature, spatial_weight
        self.diagnostics = {}

    def forward(self, rgb, feature, building):
        z = self.project(feature)
        b, _, h, w = z.shape
        ids, valid, gh, gw = candidate_map(h, w, self.stride, z.device)
        if self.partition_variant == 'grid':
            q = z.new_zeros(b, 9, h*w, dtype=torch.float32)
            q[:, 4] = 1.
        else:
            xy = normalized_coordinate_grid(b, h, w, rgb)
            embedding = self.association(torch.cat((rgb, xy), 1))
            if self.partition_variant == 'single':
                q = masked_local_softmax(embedding.float(), valid)
            else:
                q = iterative_assignment(embedding, ids, valid, building, gh*gw,
                    self.stride, 1 if self.partition_variant == 'iter1' else 3,
                    self.temperature, self.spatial_weight)
        context = pool_context(z, q, ids, building, gh*gw)
        # Diagnostics only, never used to mask the supervised pixels.
        with torch.no_grad():
            index = ids[None].expand(b, -1, -1)*2 + (building.detach().flatten(2) >= .5).long()
            mass = q.new_zeros(b, 2*gh*gw)
            for j in range(9):
                mass.scatter_add_(1, index[:, j], q[:, j])
            hard_ids = index.gather(1, q.argmax(1, keepdim=True)).squeeze(1)
            hard_mass = torch.zeros_like(mass).scatter_add_(1, hard_ids, torch.ones_like(hard_ids, dtype=q.dtype))
            self.diagnostics = dict(
                assignment_entropy=(-(q*q.clamp_min(1e-8).log()).sum(1)).mean(),
                soft_occupied_groups=(mass > 1e-6).float().sum(1).mean(),
                hard_occupied_groups=(hard_mass > 0).float().sum(1).mean(),
                allocated_groups=q.new_tensor(2*gh*gw),
                building_fraction=(building.detach() >= .5).float().mean())
        return self.readout(torch.cat((z, context.to(z.dtype), z), 1))


class PartitionModel(JointModel):
    def __init__(self, variant, dim=64, stride=8, temperature=.2, spatial_weight=.1):
        if variant not in VARIANTS:
            raise ValueError(variant)
        # Identical base initialization order; no discarded randomly initialized heads.
        super().__init__('t0', dim, stride)
        if variant != 'pixel':
            self.head = PartitionHead(variant, dim=dim, stride=stride,
                                      temperature=temperature, spatial_weight=spatial_weight)
