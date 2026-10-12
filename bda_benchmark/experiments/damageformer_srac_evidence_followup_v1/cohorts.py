"""Post-prediction retrieval diagnostics on the SAME J1 pixel cohorts.

GT only selects reporting positions. It never chooses or reranks a neighbor.
Q-projected fractions are NOT Reader attention or pixel retrieval accuracy.
"""
import torch
from torch.nn import functional as F
from bda_benchmark.experiments.damageformer_srac_joint_context_v1.model import unpool_region_tokens


@torch.no_grad()
def difficult_retrieval(target, reference, selected, bank, q, ids, shape):
    if target.shape != reference.shape or target.ndim != 3 or target.shape[0] != 1:
        raise ValueError('Expected one image and matching GT/J1 maps')
    valid = selected >= 0
    if selected.ndim != 2 or len(bank['distribution']) == 0:
        raise ValueError('Expected nonempty bank and region-by-neighbor IDs')
    if bool((selected >= len(bank['distribution'])).any()) or bool((selected < -1).any()):
        raise ValueError('Invalid selected neighbor IDs')
    classes = bank['distribution'][selected.clamp_min(0)].argmax(-1)
    slots = valid.sum(-1)
    d = valid & (classes == 2)
    intact = valid & (classes == 1)
    values = torch.stack((valid.any(-1).float(), d.any(-1).float(),
                          d.sum(-1).float()/slots.clamp_min(1),
                          intact.sum(-1).float()/slots.clamp_min(1)), dim=-1)
    field = unpool_region_tokens(values[None], q.float(), ids, shape)
    field = F.interpolate(field, target.shape[-2:], mode='bilinear', align_corners=False)
    if not torch.isfinite(field).all() or bool((field < -2e-5).any()) or bool((field > 1+2e-5).any()):
        raise ValueError('Invalid projected retrieval field')
    y, a, f = target.detach().cpu(), reference.detach().cpu(), field.detach().double().cpu()
    cohorts = dict(damaged_j1_wrong=(y == 2) & (a != 2),
                   damaged_j1_correct=(y == 2) & (a == 2),
                   intact_j1_correct=(y == 1) & (a == 1),
                   destroyed_j1_correct=(y == 3) & (a == 3))
    names = ('support_mass_sum', 'damaged_hit_mass_sum',
             'damaged_neighbor_fraction_mass_sum', 'intact_neighbor_fraction_mass_sum')
    return {name: dict(count=int(mask.sum()), **{
        key: float(f[:, index][mask].sum()) for index, key in enumerate(names)})
        for name, mask in cohorts.items()}
