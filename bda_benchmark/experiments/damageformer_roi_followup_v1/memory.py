"""Compact, fixed-space Train memory; query labels never enter retrieval."""
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import pool


BANK_FORMAT = 'ROI_FOLLOWUP_MEMORY_BANK_V1'
HEAD_FORMAT = 'ROI_FOLLOWUP_MEMORY_HEAD_V1'
INTERVENTIONS = ('normal', 'random', 'shuffle_labels', 'off')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


@torch.no_grad()
def target_distribution(gt, routing):
    """Support labels use the exact dynamic routing, including all 18 entries.

    Ignored pixels contribute zero mass. This function is for TRAIN bank
    construction only; neither retrieval nor the classification head accepts GT.
    """
    valid = (gt >= 0) & (gt < 4)
    onehot = F.one_hot(gt.clamp(0, 3).long(), 4).permute(0, 3, 1, 2).float()
    fractions = F.interpolate(onehot * valid[:, None], routing['shape'], mode='area')
    means, mass = pool(fractions.flatten(2).transpose(1, 2), routing['q'],
                       routing['ids'], routing['k'])
    # pool() guards its mean denominator at 1e-6. Undo THAT denominator so
    # counts recover the true weighted label sums even for tiny positive slots.
    counts = means * mass.clamp_min(1e-6)[:, :, None]
    labelled = counts.sum(-1)
    # Positive tiny exchange slots still need a normalized distribution. Their
    # small size is reflected in reservoir candidacy/counts, not invalid labels.
    return counts / labelled.clamp_min(torch.finfo(counts.dtype).tiny)[:, :, None], labelled


class BalancedReservoir:
    """A bounded random-priority reservoir per (event, dominant class)."""
    def __init__(self, per_bucket=64, seed=3407):
        if per_bucket < 1:
            raise ValueError('Positive reservoir capacity required')
        self.per_bucket = int(per_bucket)
        self.generator = torch.Generator().manual_seed(seed)
        self.buckets = {}
        self.seen = {}
        self.image_events = {}

    @torch.no_grad()
    def add(self, tokens, distribution, labelled_mass, event_id, image_id):
        if int(image_id) in self.image_events and self.image_events[int(image_id)] != int(event_id):
            raise ValueError('A source image cannot belong to multiple events')
        self.image_events[int(image_id)] = int(event_id)
        tokens, distribution, labelled_mass = (x.detach().float().cpu()
                                               for x in (tokens, distribution, labelled_mass))
        if tokens.ndim != 2 or distribution.shape != (len(tokens), 4) or labelled_mass.shape != (len(tokens),):
            raise ValueError('Invalid region data for reservoir')
        if not all(torch.isfinite(x).all() for x in (tokens, distribution, labelled_mass)):
            raise ValueError('Nonfinite bank candidate')
        if (distribution < 0).any() or (labelled_mass < 0).any():
            raise ValueError('Negative label distribution or mass')
        eligible = labelled_mass > 0
        if not torch.allclose(distribution[eligible].sum(-1), torch.ones(int(eligible.sum())), atol=1e-5):
            raise ValueError('Labelled support distributions must sum to one')
        for cls in range(4):
            selected = (eligible & (distribution.argmax(-1) == cls)).nonzero().flatten()
            if not len(selected):
                continue
            key = (int(event_id), cls)
            self.seen[key] = self.seen.get(key, 0) + len(selected)
            row = dict(tokens=tokens[selected].clone(), distribution=distribution[selected].clone(),
                       events=torch.full((len(selected),), int(event_id), dtype=torch.long),
                       images=torch.full((len(selected),), int(image_id), dtype=torch.long),
                       regions=selected.clone(),
                       priority=torch.rand(len(selected), generator=self.generator))
            if key in self.buckets:
                row = {name: torch.cat([self.buckets[key][name], value]) for name, value in row.items()}
            keep = row['priority'].topk(min(self.per_bucket, len(row['tokens']))).indices
            self.buckets[key] = {name: value[keep].clone() for name, value in row.items()}

    def finish(self, image_names, event_names, identity, smoke):
        if not self.buckets:
            raise ValueError('No labelled Train regions; bank is incomplete')
        ordered = [value for _, value in sorted(self.buckets.items())]
        result = {name: torch.cat([row[name] for row in ordered])
                  for name in ordered[0] if name != 'priority'}
        result.update(format=BANK_FORMAT, split='train', test_accessed=False, smoke=bool(smoke),
                      identity=identity, image_names=list(image_names), event_names=list(event_names),
                      image_event_ids=[self.image_events[index] for index in range(len(image_names))],
                      per_bucket=self.per_bucket)
        return result

    def counts(self):
        return [{'event_id': event_id, 'class': cls, 'candidates': self.seen[(event_id, cls)],
                 'retained': len(value['tokens'])}
                for (event_id, cls), value in sorted(self.buckets.items())]


def validate_bank(bank, identity, smoke, train_names=None):
    """Reject legacy banks and a bank from any other embedding/routing/split."""
    if bank.get('format') != BANK_FORMAT or bank.get('split') != 'train' or bank.get('test_accessed') is not False:
        raise ValueError('Only a new complete Train-only ROI followup bank is accepted; rebuild legacy banks')
    if bool(bank.get('smoke')) != bool(smoke) or canonical(bank.get('identity')) != canonical(identity):
        raise ValueError('Bank token contract or smoke/formal mismatch')
    names, events = bank.get('image_names', []), bank.get('event_names', [])
    if not names or len(names) != len(set(names)) or not events or len(events) != len(set(events)):
        raise ValueError('Invalid bank source names')
    if train_names is not None and not set(names).issubset(set(train_names)):
        raise ValueError('Bank contains images outside the specified Train split')
    if smoke and len(names) != 2:
        raise ValueError('Smoke bank must use exactly two Train images')
    tokens, distribution = bank['tokens'], bank['distribution']
    if tokens.ndim != 2 or not len(tokens) or distribution.shape != (len(tokens), 4):
        raise ValueError('Invalid compact bank dimensions')
    if 'dim' in identity and tokens.shape[1] != identity['dim']:
        raise ValueError('Bank embedding dimension differs from the frozen token contract')
    if not torch.isfinite(tokens).all() or not torch.isfinite(distribution).all() or (distribution < 0).any():
        raise ValueError('Invalid compact bank values')
    if not torch.allclose(distribution.sum(-1), torch.ones(len(tokens)), atol=1e-5):
        raise ValueError('Invalid bank class fractions')
    for key, ceiling in (('images', len(names)), ('events', len(events))):
        value = bank[key]
        if value.shape != (len(tokens),) or value.dtype != torch.long or (value < 0).any() or (value >= ceiling).any():
            raise ValueError('Invalid bank source IDs: ' + key)
    if bank['regions'].shape != (len(tokens),) or bank['regions'].dtype != torch.long or (bank['regions'] < 0).any():
        raise ValueError('Invalid bank region IDs')
    # One source image cannot belong to multiple events. The builder also saves
    # the event mapping explicitly so corrupt IDs cannot bypass event exclusion.
    image_event_ids = bank.get('image_event_ids')
    if image_event_ids is None or len(image_event_ids) != len(names) or any(not isinstance(x, int) or not 0 <= x < len(events) for x in image_event_ids):
        raise ValueError('Invalid per-image event metadata')
    expected = torch.tensor(image_event_ids, dtype=torch.long)[bank['images']]
    if not torch.equal(expected, bank['events']):
        raise ValueError('Bank image and event source IDs disagree')
    for image_id in bank['images'].unique():
        if bank['events'][bank['images'] == image_id].unique().numel() != 1:
            raise ValueError('A Train image maps to multiple bank events')
    per_bucket = bank.get('per_bucket', 0)
    if not isinstance(per_bucket, int) or per_bucket < 1:
        raise ValueError('Invalid bank capacity')
    classes = distribution.argmax(-1)
    sizes = torch.bincount(bank['events'] * 4 + classes, minlength=len(events) * 4)
    if int(sizes.max()) > per_bucket:
        raise ValueError('Bank exceeds per-event/class capacity')
    return bank


def read_bank(path, identity, smoke, train_names=None):
    # Kept local so synthetic CPU tests need no data-loader/backbone imports.
    from .common import digest, event
    root = Path(path)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != BANK_FORMAT or manifest.get('complete') is not True:
        raise ValueError('Expected complete ROI followup bank; old caches are incompatible')
    if manifest.get('bank_sha256') != digest(root / 'bank.pt'):
        raise ValueError('Bank file hash mismatch')
    bank = torch.load(root / 'bank.pt', map_location='cpu', weights_only=False)
    validate_bank(bank, identity, smoke, train_names)
    if any(bank['event_names'][event_id] != event(name)
           for name, event_id in zip(bank['image_names'], bank['image_event_ids'])):
        raise ValueError('Bank event IDs do not match the Train image names')
    if canonical(manifest.get('identity')) != canonical(bank['identity']):
        raise ValueError('Bank manifest/payload token contract mismatch')
    if bool(manifest.get('smoke')) != bool(smoke) or manifest.get('split') != 'train' or manifest.get('test_accessed') is not False:
        raise ValueError('Invalid bank manifest split/smoke contract')
    if (manifest.get('bank_regions') != len(bank['tokens']) or manifest.get('dim') != bank['tokens'].shape[1]
            or manifest.get('image_count') != len(bank['image_names'])
            or manifest.get('image_names') != bank['image_names']
            or manifest.get('event_names') != bank['event_names']
            or manifest.get('per_bucket') != bank['per_bucket']):
        raise ValueError('Bank manifest and compact payload metadata disagree')
    return bank, manifest


def device_bank(bank, device):
    result = {key: value.to(device) if torch.is_tensor(value) else value for key, value in bank.items()}
    result['normalized_tokens'] = F.normalize(result['tokens'].float(), dim=-1)
    result['image_lookup'] = {name: idx for idx, name in enumerate(result['image_names'])}
    result['event_lookup'] = {name: idx for idx, name in enumerate(result['event_names'])}
    return result


@torch.no_grad()
def retrieve(query, bank, event_id, image_id, topk=8, exclude_same_event=True,
             intervention='normal', generator=None):
    """Feature-only nearest supports. Call on bounded query chunks."""
    if intervention not in INTERVENTIONS or topk < 1 or query.ndim != 2:
        raise ValueError('Invalid retrieval request')
    tokens = bank['tokens']
    if not len(tokens) or query.shape[-1] != tokens.shape[-1]:
        raise ValueError('Empty or incompatible bank')
    allowed = bank['images'] != int(image_id)
    if exclude_same_event:
        allowed = allowed & (bank['events'] != int(event_id))
    if intervention == 'random':
        scores = torch.rand((len(query), len(tokens)), device=query.device, generator=generator)
    else:
        normalized = bank.get('normalized_tokens')
        if normalized is None:
            normalized = F.normalize(tokens.float(), dim=-1)
        scores = F.normalize(query.detach().float(), dim=-1) @ normalized.T
    scores = scores.masked_fill(~allowed[None], -torch.inf)
    values, indices = scores.topk(min(topk, len(tokens)), dim=-1)
    valid = torch.isfinite(values)
    distributions = bank['distribution']
    if intervention == 'shuffle_labels':
        eligible = allowed.nonzero().flatten()
        permutation = torch.arange(len(tokens), device=query.device)
        permutation[eligible] = eligible[torch.randperm(len(eligible), device=query.device, generator=generator)]
        distributions = distributions[permutation]
    if intervention == 'off':
        valid = torch.zeros_like(valid)
    return tokens[indices], distributions[indices], valid


class SupportHead(nn.Module):
    """Same trainable tensors in Memory and learned-slot no-memory controls."""
    def __init__(self, dim=128):
        super().__init__()
        if dim < 4 or dim % 4:
            raise ValueError('Attention dimension must be a positive multiple of four')
        self.labels = nn.Embedding(4, dim)
        self.attention = nn.MultiheadAttention(dim, 4, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 4)

    def forward(self, query, support=None, distribution=None, valid=None, mode='memory'):
        if query.ndim != 2 or mode not in ('memory', 'nomemory'):
            raise ValueError('Expected N-by-D query and memory/nomemory mode')
        if mode == 'nomemory':
            keys = query[:, None] + self.labels.weight[None]
            values = keys
            valid = torch.ones(keys.shape[:2], dtype=torch.bool, device=query.device)
        else:
            if support is None or distribution is None or valid is None:
                raise ValueError('Memory head requires supports, label fractions and mask')
            if support.shape[:2] != valid.shape or distribution.shape != (*valid.shape, 4):
                raise ValueError('Support dimensions do not agree')
            keys = support
            values = support + distribution @ self.labels.weight
        keys = torch.cat([keys, query[:, None]], dim=1)
        values = torch.cat([values, query[:, None]], dim=1)
        valid = torch.cat([valid, torch.ones((len(query), 1), dtype=torch.bool, device=query.device)], dim=1)
        context, _ = self.attention(query[:, None], keys, values, key_padding_mask=~valid, need_weights=False)
        return self.classifier(self.norm(query + context[:, 0]))


def make_heads(dim, device):
    prototype = SupportHead(dim).to(device)
    return {name: copy.deepcopy(prototype) for name in ('nomemory', 'memory')}


def classify_regions(head, tokens, mass, bank, names, event_names, mode, topk=8,
                     query_chunk=256, exclude_same_event=True, intervention='normal', generator=None):
    """Classify occupied slots only; keep the original slot IDs for unpooling."""
    if mode not in ('memory', 'nomemory'):
        raise ValueError('Expected memory or nomemory classification mode')
    if query_chunk < 1 or tokens.ndim != 3 or mass.shape != tokens.shape[:2] or len(names) != len(tokens):
        raise ValueError('Invalid region classification shapes/chunk')
    if len(event_names) != len(names):
        raise ValueError('One event name required per image')
    images = bank.get('image_lookup') or {name: idx for idx, name in enumerate(bank['image_names'])}
    events = bank.get('event_lookup') or {name: idx for idx, name in enumerate(bank['event_names'])}
    result = []
    statistics = dict(queries=0, supported=0, neighbor_slots=0, valid_neighbors=0)
    for index, (name, event_name) in enumerate(zip(names, event_names)):
        active = (mass[index] > 0).nonzero().flatten()
        outputs = []
        for begin in range(0, len(active), query_chunk):
            selected = active[begin:begin + query_chunk]
            query = tokens[index, selected]
            if mode == 'nomemory':
                logits = head(query, mode='nomemory')
            else:
                support, distribution, valid = retrieve(query, bank, events.get(event_name, -1),
                    images.get(name, -1), topk, exclude_same_event, intervention, generator)
                logits = head(query, support, distribution, valid, mode='memory')
                statistics['supported'] += int(valid.any(-1).sum())
                statistics['neighbor_slots'] += valid.numel()
                statistics['valid_neighbors'] += int(valid.sum())
            statistics['queries'] += len(query)
            outputs.append(logits)
        if not outputs:
            raise ValueError('Image has no occupied region slots')
        result.append(tokens.new_zeros((tokens.shape[1], 4)).index_copy(0, active, torch.cat(outputs)))
    return torch.stack(result), statistics


def add_statistics(total, row):
    for key, value in row.items():
        total[key] = total.get(key, 0) + int(value)


def finish_statistics(total):
    return dict(**total,
                support_coverage=total.get('supported', 0) / max(1, total.get('queries', 0)),
                mean_valid_neighbors=total.get('valid_neighbors', 0) / max(1, total.get('queries', 0)))
