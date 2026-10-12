"""Offline GT-only partition diagnostics; never called by model.forward."""
import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def partition_quality(route, gt):
    """Additive totals, not class/size-conditioned full-image mIoU.

    Fractional-area labels at native Q resolution, no interpolation of class IDs.
    class_c_self_sum / class_c_mass is the expected same-class probability in a
    region reached by a GT class-c contribution. Background reported separately.
    Fragmentation is measured on argmax routing, 8-connected, native grid.
    """
    from scipy import ndimage
    q, ids, k = route['q'].detach().float(), route['ids'], route['k']
    b, _, n = q.shape
    h, w = route['shape']
    if gt.ndim != 3 or len(gt) != b or h*w != n:
        raise ValueError('Diagnostic label shape mismatch')
    fraction = F.interpolate(torch.stack([(gt == c).float() for c in range(4)], 1),
                             size=(h, w), mode='area').flatten(2).transpose(1, 2)
    counts = q.new_zeros(b, k, 4)
    for j in range(q.shape[1]):
        counts.scatter_add_(1, ids[:, j, :, None].expand(-1, -1, 4),
                            fraction*q[:, j, :, None])
    mass = counts.sum(-1)
    probability = counts/torch.where(mass > 0, mass, torch.ones_like(mass))[..., None]
    result = dict(native_valid_area=float(mass.sum()),
                  native_majority_mass=float(counts.max(-1).values.sum()))
    for c in range(4):
        result[f'class_{c}_mass'] = float(counts[..., c].sum())
        result[f'class_{c}_self_sum'] = float((counts[..., c]*probability[..., c]).sum())
    fg = counts[..., 1:].sum(-1)
    result['foreground_mass'] = float(fg.sum())
    result['foreground_receives_background_sum'] = float((fg*probability[..., 0]).sum())
    result['background_receives_foreground_sum'] = float((counts[..., 0]*(1-probability[..., 0])).sum())
    hard = ids.gather(1, q.argmax(1, keepdim=True))[:, 0].reshape(b, h, w).cpu().numpy()
    used = fragmented = components = 0
    for labels in hard:
        # Bounding-box work is usually local; no N*K full image allocations.
        for index, box in enumerate(ndimage.find_objects(labels+1)):
            if box is None:
                continue
            _, number = ndimage.label(labels[box] == index, structure=np.ones((3, 3), dtype=np.uint8))
            used += 1; components += number; fragmented += int(number > 1)
    result.update(hard_used_regions=used, hard_fragmented_regions=fragmented,
                  hard_connected_components=components, image_count=b)
    return result
