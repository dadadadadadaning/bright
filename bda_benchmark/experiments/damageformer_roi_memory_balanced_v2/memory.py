"""Quality-aware dual Train banks and matched retrieval policies."""
import copy
import inspect
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_roi_followup_v1.memory import (
    canonical, retrieve as v1_global_retrieve)
from bda_benchmark.experiments.damageformer_region_memory_v2.core import ResidualSupportHead


BANK_FORMAT = 'ROI_MEMORY_QUALITY_BANK_V2'
HEAD_FORMAT = 'ROI_MEMORY_BALANCED_HEAD_V2'
CLASS_NAMES = ('Background', 'Intact', 'Damaged', 'Destroyed')
MODES = ('nomemory', 'global_all', 'global_hard', 'balanced_hard')
INTERVENTIONS = ('normal', 'random', 'shuffle_labels', 'off')


@torch.no_grad()
def target_distribution_and_quality(gt, routing):
    """Return Train labels and threshold-free hard occupancy for each slot."""
    valid = (gt >= 0) & (gt < 4)
    onehot = F.one_hot(gt.clamp(0, 3).long(), 4).permute(0, 3, 1, 2).float()
    fractions = F.interpolate(onehot * valid[:, None], routing['shape'], mode='area')
    from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import pool
    means, mass = pool(fractions.flatten(2).transpose(1, 2), routing['q'],
                       routing['ids'], routing['k'])
    counts = means * mass.clamp_min(1e-6)[:, :, None]
    labelled_mass = counts.sum(-1)
    distribution = counts / labelled_mass.clamp_min(
        torch.finfo(counts.dtype).tiny)[:, :, None]
    ids = routing['ids']
    if ids.ndim == 2:
        ids = ids[None].expand(len(gt), -1, -1)
    if ids.shape != routing['q'].shape:
        raise ValueError('Routing IDs/Q disagree for hard occupancy')
    winner = routing['q'].argmax(1)
    slot = ids.gather(1, winner[:, None]).squeeze(1)
    hard_count = torch.zeros((len(gt), routing['k']), dtype=torch.long,
                             device=gt.device)
    hard_count.scatter_add_(1, slot, torch.ones_like(slot, dtype=torch.long))
    purity = distribution.max(-1).values
    entropy = -(distribution.clamp_min(torch.finfo(distribution.dtype).tiny).log()
                * distribution).sum(-1)
    return distribution, labelled_mass, hard_count, purity, entropy


class DualQualityReservoir:
    """Separate all/hard reservoirs using the same candidate priorities.

    Each event x dominant-class bucket has its own capacity in each policy.
    The hard bank is selected directly from all hard candidates, never by
    filtering an already capped all-bank bucket.
    """
    POLICIES = ('all', 'hard')

    def __init__(self, per_bucket=64, seed=3407):
        if per_bucket < 1:
            raise ValueError('Positive reservoir capacity required')
        self.per_bucket = int(per_bucket)
        self.generator = torch.Generator().manual_seed(seed)
        self.buckets = {name: {} for name in self.POLICIES}
        self.seen = {name: {} for name in self.POLICIES}
        self.image_events = {}

    def _insert(self, policy, key, row, eligible):
        selected = eligible.nonzero().flatten()
        if not len(selected):
            return
        self.seen[policy][key] = self.seen[policy].get(key, 0) + len(selected)
        candidate = {name: value[selected].clone() for name, value in row.items()}
        if key in self.buckets[policy]:
            candidate = {name: torch.cat([self.buckets[policy][key][name], value])
                         for name, value in candidate.items()}
        keep = candidate['priority'].topk(
            min(self.per_bucket, len(candidate['tokens']))).indices
        self.buckets[policy][key] = {
            name: value[keep].clone() for name, value in candidate.items()}

    @torch.no_grad()
    def add(self, tokens, distribution, labelled_mass, hard_count, purity,
            entropy, event_id, image_id):
        image_id, event_id = int(image_id), int(event_id)
        if image_id in self.image_events and self.image_events[image_id] != event_id:
            raise ValueError('A source image cannot belong to multiple events')
        self.image_events[image_id] = event_id
        tokens = tokens.detach().float().cpu()
        distribution = distribution.detach().float().cpu()
        labelled_mass = labelled_mass.detach().float().cpu()
        hard_count = hard_count.detach().long().cpu()
        purity = purity.detach().float().cpu()
        entropy = entropy.detach().float().cpu()
        length = len(tokens)
        if (tokens.ndim != 2 or distribution.shape != (length, 4)
                or any(x.shape != (length,) for x in
                       (labelled_mass, hard_count, purity, entropy))):
            raise ValueError('Invalid region candidates')
        if not all(torch.isfinite(x).all() for x in
                   (tokens, distribution, labelled_mass, purity, entropy)):
            raise ValueError('Nonfinite bank candidate')
        if ((distribution < 0).any() or (labelled_mass < 0).any()
                or (hard_count < 0).any() or (purity < 0).any()
                or (purity > 1).any() or (entropy < 0).any()):
            raise ValueError('Invalid support quality')
        labelled = labelled_mass > 0
        if not torch.allclose(distribution[labelled].sum(-1),
                              torch.ones(int(labelled.sum())), atol=1e-5):
            raise ValueError('Labelled support distributions must sum to one')
        # Priority is sampled once per source slot, then shared by both banks.
        priority = torch.rand(length, generator=self.generator)
        base = dict(tokens=tokens, distribution=distribution,
                    labelled_mass=labelled_mass, hard_count=hard_count,
                    purity=purity, entropy=entropy,
                    events=torch.full((length,), event_id, dtype=torch.long),
                    images=torch.full((length,), image_id, dtype=torch.long),
                    regions=torch.arange(length), priority=priority)
        dominant = distribution.argmax(-1)
        for cls in range(4):
            class_mask = labelled & (dominant == cls)
            key = (event_id, cls)
            self._insert('all', key, base, class_mask)
            self._insert('hard', key, base, class_mask & (hard_count > 0))

    def finish(self, policy, image_names, event_names, identity, smoke):
        if policy not in self.POLICIES or not self.buckets[policy]:
            raise ValueError('No supports for requested reservoir policy')
        rows = [value for _, value in sorted(self.buckets[policy].items())]
        result = {name: torch.cat([row[name] for row in rows])
                  for name in rows[0] if name != 'priority'}
        result.update(format=BANK_FORMAT, policy=policy, split='train',
                      test_accessed=False, smoke=bool(smoke), identity=identity,
                      image_names=list(image_names), event_names=list(event_names),
                      image_event_ids=[self.image_events[i]
                                       for i in range(len(image_names))],
                      per_bucket=self.per_bucket)
        return result

    def counts(self, policy):
        return [dict(event_id=event_id, class_id=cls,
                     candidates=self.seen[policy][(event_id, cls)],
                     retained=len(value['tokens']))
                for (event_id, cls), value in sorted(self.buckets[policy].items())]


def validate_bank(bank, identity, smoke, policy, train_names=None):
    if (bank.get('format') != BANK_FORMAT or bank.get('policy') != policy
            or bank.get('split') != 'train' or bank.get('test_accessed') is not False):
        raise ValueError('Invalid V2 Train-only bank policy/format')
    if bool(bank.get('smoke')) != bool(smoke) or canonical(bank.get('identity')) != canonical(identity):
        raise ValueError('Bank token contract or smoke/formal mismatch')
    names, events = bank.get('image_names', []), bank.get('event_names', [])
    if not names or len(names) != len(set(names)) or not events or len(events) != len(set(events)):
        raise ValueError('Invalid bank source names')
    if train_names is not None and not set(names).issubset(set(train_names)):
        raise ValueError('Bank contains images outside Train')
    if smoke and len(names) != 2:
        raise ValueError('Smoke bank must use two Train images')
    n = len(bank.get('tokens', []))
    if bank['tokens'].ndim != 2 or not n or bank['distribution'].shape != (n, 4):
        raise ValueError('Invalid bank dimensions')
    if bank['tokens'].shape[1] != identity.get('dim', bank['tokens'].shape[1]):
        raise ValueError('Bank embedding dimension mismatch')
    for key in ('labelled_mass', 'hard_count', 'purity', 'entropy',
                'images', 'events', 'regions'):
        if bank[key].shape != (n,):
            raise ValueError('Invalid bank field: ' + key)
    if bank['hard_count'].dtype != torch.long or (bank['hard_count'] < 0).any():
        raise ValueError('hard_count must be nonnegative integer counts')
    floats = [bank[k] for k in
              ('tokens', 'distribution', 'labelled_mass', 'purity', 'entropy')]
    if not all(torch.isfinite(x).all() for x in floats):
        raise ValueError('Nonfinite bank values')
    if ((bank['distribution'] < 0).any() or (bank['labelled_mass'] <= 0).any()
            or (bank['purity'] < 0).any() or (bank['purity'] > 1).any()
            or (bank['entropy'] < 0).any()):
        raise ValueError('Invalid quality range')
    if policy == 'hard' and not (bank['hard_count'] > 0).all():
        raise ValueError('Hard bank contains zero-hard-occupancy support')
    if not torch.allclose(bank['distribution'].sum(-1), torch.ones(n), atol=1e-5):
        raise ValueError('Invalid support distributions')
    purity = bank['distribution'].max(-1).values
    entropy = -(bank['distribution'].clamp_min(
        torch.finfo(bank['distribution'].dtype).tiny).log()
        * bank['distribution']).sum(-1)
    if (not torch.allclose(bank['purity'], purity, atol=1e-6)
            or not torch.allclose(bank['entropy'], entropy, atol=1e-6)):
        raise ValueError('Stored purity/entropy disagree with labels')
    for key, ceiling in (('images', len(names)), ('events', len(events))):
        value = bank[key]
        if value.dtype != torch.long or (value < 0).any() or (value >= ceiling).any():
            raise ValueError('Invalid source IDs: ' + key)
    if bank['regions'].dtype != torch.long or (bank['regions'] < 0).any():
        raise ValueError('Invalid region IDs')
    mapping = bank.get('image_event_ids')
    if (mapping is None or len(mapping) != len(names)
            or any(not isinstance(x, int) or not 0 <= x < len(events) for x in mapping)):
        raise ValueError('Invalid image-event metadata')
    if not torch.equal(torch.tensor(mapping)[bank['images']], bank['events']):
        raise ValueError('Image/event IDs disagree')
    capacity = bank.get('per_bucket', 0)
    if not isinstance(capacity, int) or capacity < 1:
        raise ValueError('Invalid bank capacity')
    classes = bank['distribution'].argmax(-1)
    sizes = torch.bincount(bank['events'] * 4 + classes,
                           minlength=len(events) * 4)
    if int(sizes.max()) > capacity:
        raise ValueError('Bank exceeds event/class capacity')
    return bank


def read_banks(path, identity, smoke, digest, train_names=None):
    root = Path(path)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != BANK_FORMAT or manifest.get('complete') is not True:
        raise ValueError('Expected complete V2 dual quality bank')
    banks = {}
    for policy in ('all', 'hard'):
        filename = f'bank_{policy}.pt'
        if manifest.get(policy + '_sha256') != digest(root / filename):
            raise ValueError('Bank hash mismatch: ' + policy)
        bank = torch.load(root / filename, map_location='cpu', weights_only=False)
        validate_bank(bank, identity, smoke, policy, train_names)
        if canonical(manifest.get('identity')) != canonical(bank['identity']):
            raise ValueError('Manifest/payload identity mismatch')
        if manifest.get(policy + '_regions') != len(bank['tokens']):
            raise ValueError('Manifest bank size mismatch: ' + policy)
        banks[policy] = bank
    required = dict(smoke=bool(smoke), split='train', test_accessed=False,
                    dim=banks['all']['tokens'].shape[1],
                    image_count=len(banks['all']['image_names']),
                    per_bucket=banks['all']['per_bucket'])
    if any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError('Bank manifest metadata mismatch')
    for key in ('image_names', 'event_names', 'image_event_ids', 'identity'):
        if canonical(banks['all'][key]) != canonical(banks['hard'][key]):
            raise ValueError('Dual bank metadata mismatch: ' + key)
    for key in ('image_names', 'event_names', 'identity'):
        if canonical(manifest.get(key)) != canonical(banks['all'][key]):
            raise ValueError('Manifest/payload metadata mismatch: ' + key)
    return banks, manifest


def device_bank(bank, device):
    result = {key: value.to(device) if torch.is_tensor(value) else value
              for key, value in bank.items()}
    result['normalized_tokens'] = F.normalize(result['tokens'].float(), dim=-1)
    result['dominant_class'] = result['distribution'].argmax(-1)
    result['image_lookup'] = {name: i for i, name in enumerate(result['image_names'])}
    result['event_lookup'] = {name: i for i, name in enumerate(result['event_names'])}
    return result


def make_heads(dim, device):
    prototype = ResidualSupportHead(dim).to(device)
    return {name: copy.deepcopy(prototype) for name in MODES}


@torch.no_grad()
def retrieve(query, bank, event_id, image_id, topk=8,
             exclude_same_event=True, intervention='normal', balanced=False,
             generator=None):
    """Cosine retrieval; no query label or target is accepted."""
    if (intervention not in INTERVENTIONS or query.ndim != 2 or topk < 1
            or (balanced and (topk < 4 or topk % 4))):
        raise ValueError('Invalid retrieval request')
    tokens, labels = bank['tokens'], bank['distribution']
    if not len(tokens) or query.shape[-1] != tokens.shape[-1]:
        raise ValueError('Empty or incompatible bank')
    allowed = bank['images'] != int(image_id)
    if exclude_same_event:
        allowed = allowed & (bank['events'] != int(event_id))
    if intervention == 'random':
        scores = torch.rand((len(query), len(tokens)), device=query.device,
                            generator=generator)
    else:
        scores = F.normalize(query.detach().float(), dim=-1) @ bank['normalized_tokens'].T
    groups, quota = ((range(4), topk // 4) if balanced else ((None,), topk))
    selections, masks = [], []
    for cls in groups:
        permitted = allowed if cls is None else allowed & (bank['dominant_class'] == cls)
        ranked = scores.masked_fill(~permitted[None], -torch.inf)
        width = min(quota, len(tokens))
        values, indices = ranked.topk(width, dim=-1)
        valid = torch.isfinite(values)
        if balanced and width < quota:
            padding = torch.zeros((len(query), quota - width), dtype=torch.long,
                                  device=query.device)
            indices = torch.cat([indices, padding], 1)
            valid = torch.cat([valid, torch.zeros_like(padding, dtype=torch.bool)], 1)
        selections.append(indices); masks.append(valid)
    indices, valid = torch.cat(selections, 1), torch.cat(masks, 1)
    injected = labels
    if intervention == 'shuffle_labels':
        eligible = allowed.nonzero().flatten()
        permutation = torch.arange(len(tokens), device=query.device)
        if len(eligible):
            permutation[eligible] = eligible[torch.randperm(
                len(eligible), device=query.device, generator=generator)]
        injected = labels[permutation]
    if intervention == 'off':
        valid = torch.zeros_like(valid)
    metadata = {key: bank[key][indices] for key in
                ('hard_count', 'labelled_mass', 'purity', 'entropy', 'events', 'images')}
    metadata['dominant_class'] = bank['dominant_class'][indices]
    metadata['cosine'] = (F.normalize(query.detach().float(), dim=-1)[:, None]
                          * F.normalize(tokens[indices].float(), dim=-1)).sum(-1)
    return tokens[indices], injected[indices], valid, metadata


def shuffled_bank_labels(bank, event_id, image_id, exclude_same_event, generator):
    """One support-label permutation per query image, invariant to chunking."""
    allowed = bank['images'] != int(image_id)
    if exclude_same_event:
        allowed = allowed & (bank['events'] != int(event_id))
    eligible = allowed.nonzero().flatten()
    permutation = torch.arange(len(bank['tokens']), device=bank['tokens'].device)
    if len(eligible):
        permutation[eligible] = eligible[torch.randperm(
            len(eligible), device=permutation.device, generator=generator)]
    return dict(bank, distribution=bank['distribution'][permutation])


def _update_statistics(sums, valid, metadata):
    """Accumulate on device; caller performs one host synchronization."""
    sums[0] += valid.any(-1).sum()
    sums[1] += valid.sum()
    for cls in range(4):
        mask = valid & (metadata['dominant_class'] == cls)
        sums[5 + cls] += mask.sum()
        sums[9 + cls] += mask.any(-1).sum()
    for offset, key in enumerate(('hard_count', 'labelled_mass', 'purity', 'entropy')):
        sums[13 + offset] += metadata[key][valid].float().sum()


def classify_regions(head, tokens, mass, base_logits, banks, names, event_names, mode,
                     topk=8, query_chunk=256, exclude_same_event=True,
                     intervention='normal', generator=None):
    if mode not in MODES or query_chunk < 1 or tokens.ndim != 3:
        raise ValueError('Invalid classification request')
    if (mass.shape != tokens.shape[:2] or base_logits.shape != (*tokens.shape[:2], 4)
            or len(names) != len(tokens) or len(event_names) != len(names)):
        raise ValueError('Classification shapes/names disagree')
    bank = banks['all'] if mode == 'global_all' else banks['hard']
    images, events = bank['image_lookup'], bank['event_lookup']
    result, statistics = [], dict(queries=0, supported=0,
        neighbor_slots=0, valid_neighbors=0,
        valid_neighbors_by_class=[0] * 4,
        queries_with_class_by_class=[0] * 4, changed_regions=0,
        scalar_values=0, delta_abs_sum=0., base_abs_sum=0.)
    sums = torch.zeros(17, dtype=torch.float64, device=tokens.device)
    for index, (name, event_name) in enumerate(zip(names, event_names)):
        active = (mass[index] > 0).nonzero().flatten()
        event_id, image_id = events.get(event_name, -1), images.get(name, -1)
        query_bank, retrieval_intervention = bank, intervention
        if mode != 'nomemory' and intervention == 'shuffle_labels':
            query_bank = shuffled_bank_labels(
                bank, event_id, image_id, exclude_same_event, generator)
            retrieval_intervention = 'normal'
        outputs = []
        for begin in range(0, len(active), query_chunk):
            selected = active[begin:begin + query_chunk]
            query = tokens[index, selected]
            base = base_logits[index, selected]
            if mode == 'nomemory':
                logits = head(query, base, mode='nomemory')
            else:
                support, labels, valid, metadata = retrieve(
                    query, query_bank, event_id, image_id,
                    topk, exclude_same_event, retrieval_intervention,
                    balanced=mode == 'balanced_hard', generator=generator)
                logits = head(query, base, support, labels, valid, mode='memory')
                _update_statistics(sums, valid, metadata)
                statistics['neighbor_slots'] += valid.numel()
            sums[2] += (logits.argmax(-1) != base.argmax(-1)).sum()
            statistics['scalar_values'] += base.numel()
            sums[3] += (logits.detach().float() - base.detach().float()).abs().sum()
            sums[4] += base.detach().float().abs().sum()
            statistics['queries'] += len(query)
            outputs.append(logits)
        if not outputs:
            raise ValueError('Image has no occupied regions')
        result.append(base_logits[index].index_copy(0, active, torch.cat(outputs)))
    values = sums.cpu().tolist()
    statistics.update(
        supported=int(values[0]), valid_neighbors=int(values[1]),
        changed_regions=int(values[2]), delta_abs_sum=values[3],
        base_abs_sum=values[4],
        valid_neighbors_by_class=[int(x) for x in values[5:9]],
        queries_with_class_by_class=[int(x) for x in values[9:13]],
        **{'selected_' + key + '_sum': values[13 + index]
           for index, key in enumerate(('hard_count', 'labelled_mass', 'purity', 'entropy'))})
    return torch.stack(result), statistics


def add_statistics(total, row):
    for key, value in row.items():
        if isinstance(value, list):
            target = total.setdefault(key, [0] * len(value))
            for i, item in enumerate(value):
                target[i] += int(item)
        elif isinstance(value, float):
            total[key] = total.get(key, 0.) + value
        else:
            total[key] = total.get(key, 0) + int(value)


def finish_statistics(total):
    queries, neighbors = total.get('queries', 0), total.get('valid_neighbors', 0)
    counts = total.get('valid_neighbors_by_class', [0] * 4)
    covered = total.get('queries_with_class_by_class', [0] * 4)
    result = dict(**total,
        support_coverage=total.get('supported', 0) / max(1, queries),
        mean_valid_neighbors=neighbors / max(1, queries),
        class_basis='stored Train support dominant class before label intervention',
        by_class={name: dict(valid_neighbors=counts[i],
                             mean_valid_neighbors=counts[i] / max(1, queries),
                             query_coverage=covered[i] / max(1, queries))
                  for i, name in enumerate(CLASS_NAMES)})
    result['selected_quality_means'] = {
        key: total.get('selected_' + key + '_sum', 0.) / max(1, neighbors)
        for key in ('hard_count', 'labelled_mass', 'purity', 'entropy')}
    scalars = max(1, total.get('scalar_values', 0))
    result.update(
        correction_abs_mean=total.get('delta_abs_sum', 0.) / scalars,
        base_abs_mean=total.get('base_abs_sum', 0.) / scalars,
        correction_to_base_abs_ratio=(total.get('delta_abs_sum', 0.) /
                                      max(1e-12, total.get('base_abs_sum', 0.))),
        region_prediction_change_rate=total.get('changed_regions', 0) / max(1, queries))
    return result


def assert_global_matches_v1(query, bank, event_id, image_id, topk=8,
                             exclude_same_event=True):
    ours = retrieve(query, bank, event_id, image_id, topk,
                    exclude_same_event, 'normal', False)[:3]
    legacy = v1_global_retrieve(query, bank, event_id, image_id, topk,
                                exclude_same_event, 'normal')
    if any(not torch.equal(a, b) for a, b in zip(ours, legacy)):
        raise RuntimeError('Global cosine top-k drifted from V1 semantics')


def assert_no_query_target_interfaces():
    forbidden = {'gt', 'target', 'query_gt', 'query_label', 'query_labels'}
    for function in (retrieve, classify_regions, ResidualSupportHead.forward,
                     ResidualSupportHead.correction):
        if forbidden & set(inspect.signature(function).parameters):
            raise RuntimeError('Query targets must never enter retrieval/classification')
