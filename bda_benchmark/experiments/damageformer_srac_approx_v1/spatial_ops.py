"""Prediction-only interventions; no GT arguments or classifier modifications."""
import torch
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import candidate_map, soft_pool_unpool


def grid_project(prob, stride):
    ids, _, gh, gw = candidate_map(*prob.shape[-2:], stride, prob.device)
    q = prob.new_zeros(prob.shape[0], 9, prob.shape[-2]*prob.shape[-1])
    q[:, 4] = 1
    return soft_pool_unpool(prob, q, ids, gh*gw)['reconstruction']


def split_project(prob, q, ids, count, predicted_building):
    """Duplicate each token into BG/FG children. Same q for pool AND unpool.

    No class probability clamping or hard final foreground/background masking.
    Empty children receive no query mass. Same group may contain disconnected sites.
    """
    if predicted_building.shape != (prob.shape[0], *prob.shape[-2:]):
        raise ValueError('Expected B,H,W predicted building mask')
    results = []
    for i in range(len(prob)):
        child_ids = 2*ids + predicted_building[i].bool().flatten().long()[None]
        results.append(soft_pool_unpool(prob[i:i+1], q[i:i+1], child_ids, 2*count)['reconstruction'])
    return torch.cat(results)
