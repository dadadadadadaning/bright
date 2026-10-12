import torch
from bda_benchmark.experiments.damageformer_srac_approx_v1.modules import (
    RegionModule, candidate_map, masked_local_softmax,
    normalized_coordinate_grid, soft_pool_unpool,
)


def child_ids(ids, building):
    """Per-image candidate IDs; prediction only, never ground truth."""
    group = (building.detach().flatten() >= 0.5).long()
    return 2 * ids + group[None]


class SplitRegionModule(RegionModule):
    def __init__(self, *args, split=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split

    def forward(self, rgb, content, building):
        if not self.split:
            return super().forward(rgb, content, building)
        b, c, h, w = content.shape
        if building.shape != (b, 1, h, w):
            raise ValueError('Building probability must match feature resolution')
        ids, valid, gh, gw = candidate_map(h, w, self.stride, content.device)
        xy = normalized_coordinate_grid(b, h, w, content)
        logits = self.association(torch.cat([rgb, xy], 1))
        if not self.learned:
            logits = torch.full_like(logits, -1e4)
            logits[:, 4] = 0
        q = masked_local_softmax(logits.float(), valid)
        hard = ids[None].expand(b, -1, -1).gather(
            1, q.argmax(1, keepdim=True)).reshape(b, h, w)
        # Preserve original S3 modulation: do not add split boundaries here.
        edge = content.new_zeros(b, 1, h, w)
        edge[:, :, 1:] = (hard[:, 1:] != hard[:, :-1]).unsqueeze(1)
        edge[:, :, :, 1:] = torch.maximum(edge[:, :, :, 1:],
            (hard[:, :, 1:] != hard[:, :, :-1]).unsqueeze(1).float())
        projected = self.project(content)
        if self.modulation:
            projected = projected + self.modulator(torch.cat([
                content * (0.1 + 0.9 * building.detach()), building.detach(), edge], 1))
        weights = self.score(projected).sigmoid().flatten(2)
        if not self.weighted:
            weights = projected.new_ones(b, 1, h*w)
        pixels, tokens, masses = [], [], []
        for n in range(b):
            idx = child_ids(ids, building[n])
            prob = q[n:n+1]
            pooled = soft_pool_unpool(projected[n:n+1].float(),
                prob * weights[n:n+1], idx, 2*gh*gw)['tokens']
            x = content[n:n+1]
            stats = soft_pool_unpool(torch.cat([x, x.square()], 1).float(),
                prob, idx, 2*gh*gw)
            mean, second = stats['tokens'].chunk(2, -1)
            z = self.main(pooled)
            if self.weighted:
                std = (second - mean.square()).clamp_min(0).add(1e-6).sqrt()
                z = z + self.alpha.tanh() * self.skip(torch.cat([mean, std], -1))
            regional = self.classifier(z)
            pixel = content.new_zeros(1, h*w, 4)
            for j in range(9):
                pixel = pixel + regional.gather(1,
                    idx[j][None, :, None].expand(1, -1, 4)) * prob[:, j, :, None]
            pixels.append(pixel)
            tokens.append(z)
            masses.append(stats['mass'])
        # Keep UNSPLIT assignment for the unchanged original partition loss.
        # Child masses/counts are named separately to avoid inconsistent aux.
        return dict(logits=torch.cat(pixels).transpose(1, 2).reshape(b, 4, h, w),
            tokens=torch.cat(tokens), q=q, ids=ids, count=gh*gw, xy=xy,
            hard_ids=hard, child_mass=torch.cat(masses), child_count=2*gh*gw)
