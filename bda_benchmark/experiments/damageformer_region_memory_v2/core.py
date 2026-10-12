"""Matched residual readers on a frozen hard-split region representation.

This is a controlled retrieval adaptation, not an official SRAC reproduction.
The source classifier is retained exactly at initialization. Zero initialization
does NOT guarantee that later optimization preserves its correct predictions.
Frozen front/token/base-logit boundaries belong to the caller (use no_grad).
"""
import copy

import torch
from torch import nn

from bda_benchmark.experiments.damageformer_roi_followup_v1.memory import (
    INTERVENTIONS, device_bank, retrieve,
)


HEAD_FORMAT = 'REGION_RESIDUAL_MEMORY_HEAD_V2'
MODES = ('nomemory', 'memory')


class ResidualSupportHead(nn.Module):
    """Same trainable tensors in retrieval and learned-slot controls.

    Memory keys are retrieved feature vectors; values additionally contain
    *support* soft-label embeddings. No-memory uses four learned label slots
    conditioned on the query. Both always include the query fallback, even when
    every retrieved support is excluded. Query GT is never an input.
    """
    def __init__(self, dim=128):
        super().__init__()
        if not isinstance(dim, int) or dim < 4 or dim % 4:
            raise ValueError('Attention dimension must be a positive multiple of four')
        self.dim = dim
        self.labels = nn.Embedding(4, dim)
        self.attention = nn.MultiheadAttention(dim, 4, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 4))
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def correction(self, query, support=None, distribution=None, valid=None, mode='memory'):
        if mode not in MODES or query.ndim != 2 or query.shape[-1] != self.dim:
            raise ValueError('Expected N-by-D queries and memory/nomemory mode')
        if not query.is_floating_point():
            raise ValueError('Query features must be floating point')
        if mode == 'nomemory':
            # Deliberately independent of any bank/support arguments.
            keys = query[:, None] + self.labels.weight[None]
            values = keys
            valid = torch.ones(keys.shape[:2], dtype=torch.bool, device=query.device)
        else:
            if support is None or distribution is None or valid is None:
                raise ValueError('Memory mode requires support features, fractions, and validity')
            if (support.ndim != 3 or support.shape[0] != len(query)
                    or support.shape[-1] != self.dim or valid.shape != support.shape[:2]
                    or distribution.shape != (*support.shape[:2], 4)
                    or valid.dtype != torch.bool):
                raise ValueError('Invalid support/distribution/mask dimensions or dtype')
            if any(value.device != query.device for value in (support, distribution, valid)):
                raise ValueError('Queries and support tensors must be on the same device')
            # Masked entries must not contaminate attention even if a caller
            # supplies undefined padding. Valid bank rows are checked at load.
            keys = torch.where(valid[..., None], support, torch.zeros_like(support))
            fractions = torch.where(valid[..., None], distribution, torch.zeros_like(distribution))
            values = keys + fractions @ self.labels.weight
        keys = torch.cat((keys, query[:, None]), dim=1)
        values = torch.cat((values, query[:, None]), dim=1)
        valid = torch.cat((valid, torch.ones((len(query), 1), dtype=torch.bool,
                                            device=query.device)), dim=1)
        context, _ = self.attention(query[:, None], keys, values,
                                    key_padding_mask=~valid, need_weights=False)
        return self.ffn(self.norm(query + context[:, 0]))

    def forward(self, query, base_logits, support=None, distribution=None, valid=None, mode='memory'):
        if base_logits.shape != (len(query), 4) or base_logits.device != query.device:
            raise ValueError('Expected source logits N-by-4 on query device')
        return base_logits + self.correction(query, support, distribution, valid, mode)


def make_heads(dim, device):
    prototype = ResidualSupportHead(dim).to(device)
    return {name: copy.deepcopy(prototype) for name in MODES}


def _shuffled_allowed_labels(bank, event_id, image_id, exclude_same_event, generator):
    """One fixed support-label permutation per query image, not per chunk."""
    allowed = bank['images'] != int(image_id)
    if exclude_same_event:
        allowed = allowed & (bank['events'] != int(event_id))
    eligible = allowed.nonzero().flatten()
    permutation = torch.arange(len(bank['tokens']), device=bank['tokens'].device)
    permutation[eligible] = eligible[torch.randperm(len(eligible),
        device=permutation.device, generator=generator)]
    return dict(bank, distribution=bank['distribution'][permutation])


def classify_regions(head, tokens, mass, base_logits, bank, names, event_names, mode,
                     topk=8, query_chunk=256, exclude_same_event=True,
                     intervention='normal', generator=None):
    """Correct occupied slots without changing IDs, routing, or empty slots.

    Normal/shuffled inference is invariant to the query chunk size (up to
    floating-point attention roundoff). ``shuffle_labels`` fixes one permutation
    among allowed bank rows before any query chunk; excluded rows stay excluded.
    Readout diagnostics count regions, NOT full-resolution pixel changes.
    """
    if mode not in MODES or intervention not in INTERVENTIONS:
        raise ValueError('Unknown classification mode or intervention')
    if (tokens.ndim != 3 or mass.shape != tokens.shape[:2]
            or base_logits.shape != (*tokens.shape[:2], 4)
            or tokens.shape[-1] != head.dim or len(names) != len(tokens)
            or len(event_names) != len(tokens) or query_chunk < 1 or topk < 1):
        raise ValueError('Invalid region dimensions, names, chunk size, or topk')
    if mass.device != tokens.device or base_logits.device != tokens.device:
        raise ValueError('Region tensors must share a device')
    if mode == 'memory' and bank is None:
        raise ValueError('Memory classification needs a validated Train bank')
    images, events = {}, {}
    if mode == 'memory':
        images = bank.get('image_lookup') or {name: idx for idx, name in enumerate(bank['image_names'])}
        events = bank.get('event_lookup') or {name: idx for idx, name in enumerate(bank['event_names'])}
    statistics = dict(queries=0, supported=0, neighbor_slots=0, valid_neighbors=0,
                      changed_regions=0, scalar_values=0, delta_abs_sum=0., base_abs_sum=0.)
    # One host synchronization at the end, not per chunk/diagnostic scalar.
    sums = torch.zeros(5, dtype=torch.float64, device=tokens.device)
    result = []
    for index, (name, event_name) in enumerate(zip(names, event_names)):
        active = (mass[index] > 0).nonzero().flatten()
        event_id, image_id = events.get(event_name, -1), images.get(name, -1)
        query_bank, retrieval_intervention = bank, intervention
        if mode == 'memory' and intervention == 'shuffle_labels':
            query_bank = _shuffled_allowed_labels(bank, event_id, image_id,
                                                  exclude_same_event, generator)
            retrieval_intervention = 'normal'
        outputs = []
        for begin in range(0, len(active), query_chunk):
            selected = active[begin:begin + query_chunk]
            query, base = tokens[index, selected], base_logits[index, selected]
            if mode == 'nomemory':
                logits = head(query, base, mode='nomemory')
            else:
                support, distribution, valid = retrieve(query, query_bank, event_id, image_id,
                    topk=topk, exclude_same_event=exclude_same_event,
                    intervention=retrieval_intervention, generator=generator)
                logits = head(query, base, support, distribution, valid, mode='memory')
                sums[0] += valid.any(-1).sum()
                sums[1] += valid.sum()
                statistics['neighbor_slots'] += valid.numel()
            with torch.no_grad():
                sums[2] += (logits.argmax(-1) != base.argmax(-1)).sum()
                sums[3] += (logits.detach().float() - base.detach().float()).abs().double().sum()
                sums[4] += base.detach().abs().double().sum()
            statistics['queries'] += len(query)
            statistics['scalar_values'] += base.numel()
            outputs.append(logits)
        result.append(base_logits[index].index_copy(0, active, torch.cat(outputs))
                      if outputs else base_logits[index])
    if not result:
        raise ValueError('An empty image batch cannot be classified')
    supported, valid_neighbors, changed, delta_sum, base_sum = sums.cpu().tolist()
    statistics.update(supported=int(supported), valid_neighbors=int(valid_neighbors),
                      changed_regions=int(changed), delta_abs_sum=delta_sum, base_abs_sum=base_sum)
    return torch.stack(result), statistics


def add_statistics(total, row):
    """Aggregate raw counts/sums without truncating floating-point diagnostics."""
    for key, value in row.items():
        total[key] = total.get(key, 0) + value


def finish_statistics(total):
    queries, scalars = max(1, total.get('queries', 0)), max(1, total.get('scalar_values', 0))
    return dict(total,
                support_coverage=total.get('supported', 0) / queries,
                mean_valid_neighbors=total.get('valid_neighbors', 0) / queries,
                correction_abs_mean=total.get('delta_abs_sum', 0.) / scalars,
                base_abs_mean=total.get('base_abs_sum', 0.) / scalars,
                correction_to_base_abs_ratio=total.get('delta_abs_sum', 0.) / max(1e-12, total.get('base_abs_sum', 0.)),
                region_prediction_change_rate=total.get('changed_regions', 0) / queries)
