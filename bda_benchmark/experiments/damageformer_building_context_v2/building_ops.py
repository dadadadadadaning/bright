"""Building and region grouping primitives.

Main-method region formation is NOT based on RGB raw-color clustering.
The learned region assignments are produced by the existing C3-C7
SCN-style 9-neighbour association module.

This file only groups those learned regions into predicted building instances.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def predicted_building_instances(
    p_building: torch.Tensor,
    size: Tuple[int, int],
    threshold: float = 0.50,
    min_area: int = 8,
) -> torch.Tensor:
    """Create 4-connected predicted building instances at region-feature resolution."""
    from scipy import ndimage

    prob = F.interpolate(
        p_building.detach(), size=size, mode="bilinear", align_corners=False
    )[:, 0]

    structure = np.asarray(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )
    outputs = []
    for sample in prob.float().cpu().numpy():
        labels, n = ndimage.label(sample >= threshold, structure=structure)
        compact = np.zeros_like(labels, dtype=np.int64)
        new_id = 1
        for old_id in range(1, n + 1):
            mask = labels == old_id
            if int(mask.sum()) < min_area:
                continue
            compact[mask] = new_id
            new_id += 1
        outputs.append(torch.from_numpy(compact))
    return torch.stack(outputs).to(p_building.device)


@torch.no_grad()
def hard_region_ids_from_soft_assignment(
    probabilities: torch.Tensor,
    candidate_ids: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Hard IDs only for building grouping. Pool/unpool remains soft/differentiable."""
    b, nine, n = probabilities.shape
    if nine != 9 or n != height * width:
        raise ValueError(
            f"association mismatch: got {tuple(probabilities.shape)}, expected Bx9x{height*width}"
        )
    choice = probabilities.argmax(dim=1)  # B,N
    candidates = candidate_ids.unsqueeze(0).expand(b, -1, -1)
    selected = candidates.gather(1, choice.unsqueeze(1)).squeeze(1)
    return selected.reshape(b, height, width)


@torch.no_grad()
def region_building_membership(
    region_ids: torch.Tensor,
    building_ids: torch.Tensor,
    token_count: int,
    purity_threshold: float = 0.60,
) -> Dict[str, torch.Tensor]:
    """Assign each learned region to its dominant predicted building."""
    if region_ids.shape != building_ids.shape:
        raise ValueError(
            f"region/building shape mismatch: {region_ids.shape} vs {building_ids.shape}"
        )

    b = region_ids.shape[0]
    building_of_region = torch.zeros(
        b, token_count, dtype=torch.long, device=region_ids.device
    )
    purity = torch.zeros(
        b, token_count, dtype=torch.float32, device=region_ids.device
    )
    region_count_out = torch.zeros(
        b, token_count, dtype=torch.float32, device=region_ids.device
    )

    for bi in range(b):
        r = region_ids[bi].reshape(-1).long()
        g = building_ids[bi].reshape(-1).long()

        region_count = torch.bincount(r, minlength=token_count).float()
        region_count_out[bi] = region_count

        max_building = int(g.max().item())
        if max_building <= 0:
            continue

        valid = g > 0
        pair = g[valid] * token_count + r[valid]
        counts = torch.bincount(
            pair, minlength=(max_building + 1) * token_count
        ).reshape(max_building + 1, token_count).float()
        counts[0].zero_()

        dominant_count, dominant_building = counts.max(dim=0)
        p = dominant_count / region_count.clamp_min(1.0)
        keep = p >= float(purity_threshold)

        building_of_region[bi, keep] = dominant_building[keep]
        purity[bi] = p

    return {
        "building_of_region": building_of_region,
        "region_building_purity": purity,
        "region_pixel_count": region_count_out,
    }


def soft_unpool_tokens(
    tokens: torch.Tensor,
    probabilities: torch.Tensor,
    candidate_ids: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Reproject refined BxKxC tokens using the original soft 9-way association."""
    b, _, c = tokens.shape
    _, nine, n = probabilities.shape
    if nine != 9 or n != height * width:
        raise ValueError("soft_unpool_tokens: invalid association shape")

    reconstruction = tokens.new_zeros((b, n, c))
    probs = probabilities.to(dtype=tokens.dtype)

    for j in range(9):
        idx = candidate_ids[j].view(1, n, 1).expand(b, n, c)
        reconstruction.add_(tokens.gather(1, idx) * probs[:, j].unsqueeze(-1))

    return reconstruction.transpose(1, 2).reshape(b, c, height, width)
