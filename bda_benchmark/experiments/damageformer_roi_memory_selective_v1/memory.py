"""Selective Train-memory retrieval and explicit posterior fusion.

This module intentionally reuses the validated V2 *hard* bank.  It adds two
independent, falsifiable changes:

1. quality-aware adaptive retrieval: Train-support quality filters plus a
   per-class ceiling, with no class minimum and no low-quality backfill;
2. an explicit support-label posterior whose contribution is controlled by a
   reliability gate.  A matched query-only residual is present in every
   posterior head so that gains cannot be attributed merely to extra capacity.

No query label/GT enters retrieval, gating, or classification.
"""
import copy
import inspect
import math

import torch
from torch import nn
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_roi_memory_balanced_v2 import memory as parent
from bda_benchmark.experiments.damageformer_region_memory_v2.core import ResidualSupportHead


BANK_FORMAT = parent.BANK_FORMAT
HEAD_FORMAT = 'ROI_MEMORY_SELECTIVE_HEAD_V1'
CLASS_NAMES = parent.CLASS_NAMES
INTERVENTIONS = parent.INTERVENTIONS
RETRIEVAL_POLICIES = ('global_hard', 'fixed_balanced', 'adaptive_cap')
RETRIEVAL_MODES = ('nomemory',) + RETRIEVAL_POLICIES
POSTERIOR_MODES = (
    'query_only',
    'posterior_nogate',
    'posterior_query_gate',
    'posterior_quality_gate',
)


# The bank format and its validation remain owned by the parent experiment.
read_banks = parent.read_banks
validate_bank = parent.validate_bank
target_distribution_and_quality = parent.target_distribution_and_quality


def device_bank(bank, device):
    """Move a validated parent bank to ``device`` without changing semantics."""
    return parent.device_bank(bank, device)


@torch.no_grad()
def build_quality_view(bank, quantile=.25):
    """Attach class-conditional Train-only quality thresholds and eligibility.

    For each stored dominant class, a support is retained when all three hold:
    purity >= q(quantile), entropy <= q(1-quantile), and hard occupancy >=
    q(quantile).  Threshold comparisons are inclusive.  This changes neither
    the underlying bank nor its stored labels.
    """
    if not isinstance(quantile, (int, float)) or not 0 <= float(quantile) < .5:
        raise ValueError('quality quantile must be in [0, 0.5)')
    required = ('tokens', 'distribution', 'purity', 'entropy', 'hard_count')
    if any(key not in bank for key in required):
        raise ValueError('Quality view needs a validated V2 hard bank')
    if bank.get('policy') != 'hard' or bank.get('split') != 'train':
        raise ValueError('Quality thresholds must be estimated from the Train hard bank')
    classes = bank.get('dominant_class')
    if classes is None:
        classes = bank['distribution'].argmax(-1)
    eligible = torch.zeros(len(bank['tokens']), dtype=torch.bool,
                           device=bank['tokens'].device)
    thresholds = []
    q = float(quantile)
    for cls in range(4):
        selected = classes == cls
        count = int(selected.sum())
        if not count:
            thresholds.append(dict(class_id=cls, count=0, retained=0,
                                   purity_min=None, entropy_max=None,
                                   hard_count_min=None))
            continue
        purity_min = torch.quantile(bank['purity'][selected].float(), q)
        entropy_max = torch.quantile(bank['entropy'][selected].float(), 1. - q)
        hard_min = torch.quantile(bank['hard_count'][selected].float(), q)
        keep = (selected & (bank['purity'] >= purity_min)
                & (bank['entropy'] <= entropy_max)
                & (bank['hard_count'].float() >= hard_min))
        eligible |= keep
        thresholds.append(dict(
            class_id=cls, count=count, retained=int(keep.sum()),
            purity_min=float(purity_min), entropy_max=float(entropy_max),
            hard_count_min=float(hard_min)))
    result = dict(bank)
    result['dominant_class'] = classes
    result['quality_eligible'] = eligible
    result['quality_quantile'] = q
    result['quality_thresholds'] = thresholds
    result['quality_source'] = 'stored Train hard bank, class-conditional quantiles'
    return result


def _validate_retrieval(query, bank, policy, topk, preselect, per_class_cap,
                        min_cosine, intervention):
    if policy not in RETRIEVAL_POLICIES or intervention not in INTERVENTIONS:
        raise ValueError('Unknown retrieval policy/intervention')
    if query.ndim != 2 or not query.is_floating_point() or topk < 1:
        raise ValueError('Retrieval expects floating N-by-D queries and positive topk')
    if (bank['tokens'].ndim != 2 or not len(bank['tokens'])
            or query.shape[-1] != bank['tokens'].shape[-1]):
        raise ValueError('Empty or incompatible support bank')
    if bank['tokens'].device != query.device:
        raise ValueError('Queries and support bank must share a device')
    if policy == 'fixed_balanced' and (topk < 4 or topk % 4):
        raise ValueError('fixed_balanced requires topk divisible by four')
    if policy == 'adaptive_cap':
        if 'quality_eligible' not in bank:
            raise ValueError('adaptive_cap requires build_quality_view(bank, quantile)')
        if preselect < topk or per_class_cap < 1:
            raise ValueError('adaptive_cap requires preselect>=topk and positive cap')
        if not -1. <= min_cosine <= 1.:
            raise ValueError('min_cosine must be in [-1, 1]')
        if intervention == 'random' and min_cosine != -1.:
            raise ValueError(
                'random control requires min_cosine=-1 so its candidate pool '
                'is not conditioned on semantic similarity')


def _base_allowed(bank, event_id, image_id, exclude_same_event):
    allowed = bank['images'] != int(image_id)
    if exclude_same_event:
        allowed = allowed & (bank['events'] != int(event_id))
    return allowed


def _candidate_pool(bank, allowed, policy):
    if policy == 'adaptive_cap':
        return allowed & bank['quality_eligible']
    return allowed


def _shuffle_labels(bank, allowed, generator):
    """Return a label-only permutation; token rows/IDs remain unchanged."""
    eligible = allowed.nonzero().flatten()
    permutation = torch.arange(len(bank['tokens']), device=bank['tokens'].device)
    if len(eligible):
        order = torch.randperm(len(eligible), device=eligible.device,
                               generator=generator)
        permutation[eligible] = eligible[order]
    return bank['distribution'][permutation]


def shuffled_bank_labels(bank, event_id, image_id, exclude_same_event,
                         policy, generator):
    """Fix one support-label permutation per query image (chunk invariant)."""
    if policy not in RETRIEVAL_POLICIES:
        raise ValueError('Unknown retrieval policy')
    allowed = _candidate_pool(
        bank, _base_allowed(bank, event_id, image_id, exclude_same_event), policy)
    return dict(bank, distribution=_shuffle_labels(bank, allowed, generator))


def _pad_topk(indices, valid, topk):
    if indices.shape[1] < topk:
        width = topk - indices.shape[1]
        pad = torch.zeros((len(indices), width), dtype=torch.long,
                          device=indices.device)
        indices = torch.cat((indices, pad), 1)
        valid = torch.cat((valid, torch.zeros_like(pad, dtype=torch.bool)), 1)
    return indices, valid


@torch.no_grad()
def retrieve(query, bank, event_id, image_id, policy='adaptive_cap', topk=8,
             preselect=32, per_class_cap=4, min_cosine=-1.,
             exclude_same_event=True, intervention='normal', generator=None):
    """Retrieve supports without accepting query labels.

    ``adaptive_cap`` first ranks a global quality-filtered top-``preselect``
    candidate list, then walks that list in order while enforcing only a
    per-class *maximum*.  Missing classes are never forced and rejected rows
    are never replaced from below the preselection boundary.
    """
    _validate_retrieval(query, bank, policy, topk, preselect, per_class_cap,
                        min_cosine, intervention)
    tokens = bank['tokens']
    normalized = bank.get('normalized_tokens')
    if normalized is None:
        normalized = F.normalize(tokens.float(), dim=-1)
    cosine = F.normalize(query.detach().float(), dim=-1) @ normalized.T
    allowed = _base_allowed(bank, event_id, image_id, exclude_same_event)
    candidate = _candidate_pool(bank, allowed, policy)
    if intervention == 'random':
        ranking = torch.rand(cosine.shape, device=query.device, generator=generator)
    else:
        ranking = cosine

    n_query, n_bank = cosine.shape
    indices = torch.zeros((n_query, topk), dtype=torch.long, device=query.device)
    valid = torch.zeros((n_query, topk), dtype=torch.bool, device=query.device)

    if policy == 'global_hard':
        ranked = ranking.masked_fill(~candidate[None], -torch.inf)
        width = min(topk, n_bank)
        values, chosen = ranked.topk(width, dim=-1)
        chosen_valid = torch.isfinite(values)
        chosen, chosen_valid = _pad_topk(chosen, chosen_valid, topk)
        indices.copy_(chosen); valid.copy_(chosen_valid)
    elif policy == 'fixed_balanced':
        quota = topk // 4
        parts, masks = [], []
        classes = bank['dominant_class']
        for cls in range(4):
            permitted = candidate & (classes == cls)
            ranked = ranking.masked_fill(~permitted[None], -torch.inf)
            width = min(quota, n_bank)
            values, chosen = ranked.topk(width, dim=-1)
            chosen_valid = torch.isfinite(values)
            chosen, chosen_valid = _pad_topk(chosen, chosen_valid, quota)
            parts.append(chosen); masks.append(chosen_valid)
        indices.copy_(torch.cat(parts, 1)); valid.copy_(torch.cat(masks, 1))
    else:
        # Normal retrieval applies the similarity floor.  The random control
        # deliberately skips it (and is only valid with min_cosine=-1), so it
        # changes ranking without silently restoring semantic filtering.
        permitted = candidate[None]
        if intervention != 'random':
            permitted = permitted & (cosine >= float(min_cosine))
        ranked = ranking.masked_fill(~permitted, -torch.inf)
        width = min(preselect, n_bank)
        values, candidates = ranked.topk(width, dim=-1)
        finite = torch.isfinite(values)
        class_counts = torch.zeros((n_query, 4), dtype=torch.long,
                                   device=query.device)
        positions = torch.zeros(n_query, dtype=torch.long, device=query.device)
        classes = bank['dominant_class']
        rows_all = torch.arange(n_query, device=query.device)
        # A short bounded loop (default 32), vectorized over queries.
        for rank in range(width):
            proposed = candidates[:, rank]
            proposed_class = classes[proposed]
            current = class_counts.gather(1, proposed_class[:, None]).squeeze(1)
            take = finite[:, rank] & (positions < topk) & (current < per_class_cap)
            rows = rows_all[take]
            if len(rows):
                cols = positions[rows]
                picked = proposed[rows]
                picked_class = proposed_class[rows]
                indices[rows, cols] = picked
                valid[rows, cols] = True
                class_counts[rows, picked_class] += 1
                positions[rows] += 1

    injected = bank['distribution']
    if intervention == 'shuffle_labels':
        injected = _shuffle_labels(bank, candidate, generator)
    if intervention == 'off':
        valid.zero_()

    selected_cosine = cosine.gather(1, indices)
    metadata = {key: bank[key][indices] for key in
                ('hard_count', 'labelled_mass', 'purity', 'entropy',
                 'events', 'images', 'regions')}
    metadata.update(
        indices=indices,
        dominant_class=bank['dominant_class'][indices],
        cosine=selected_cosine,
        quality_eligible=(bank.get('quality_eligible',
                                   torch.ones(n_bank, dtype=torch.bool,
                                              device=query.device))[indices]),
    )
    return tokens[indices], injected[indices], valid, metadata


def make_retrieval_heads(dim, device):
    """Independent residual controls with identical initialization."""
    prototype = ResidualSupportHead(dim).to(device)
    return {name: copy.deepcopy(prototype) for name in RETRIEVAL_MODES}


class SelectivePosteriorHead(nn.Module):
    """Matched query residual plus optional reliability-gated label posterior.

    Fusion is performed in logit space:

      base + query_residual + strength * gate * (log(p_mem)-log(p_base)).

    This is a geometric posterior correction, not arithmetic probability
    mixing.  ``strength`` and the final query-residual layer start at zero, so
    every mode is exactly the frozen source at step zero.  Call
    :meth:`clamp_strength_` immediately after each optimizer step.
    """
    RETRIEVAL_FEATURES = 7

    def __init__(self, dim=128):
        super().__init__()
        if not isinstance(dim, int) or dim < 4:
            raise ValueError('Positive feature dimension >=4 required')
        self.dim = dim
        hidden = max(16, dim // 2)
        self.query_norm = nn.LayerNorm(dim)
        self.query_residual = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 4))
        self.gate = nn.Sequential(
            nn.Linear(dim + 2 + self.RETRIEVAL_FEATURES, hidden),
            nn.GELU(), nn.Linear(hidden, 1))
        self.strength = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.query_residual[-1].weight)
        nn.init.zeros_(self.query_residual[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    @torch.no_grad()
    def clamp_strength_(self):
        self.strength.clamp_(0., 1.)

    def _posterior(self, base_logits, distribution, valid, cosine,
                   temperature):
        if temperature <= 0:
            raise ValueError('Positive posterior temperature required')
        if (distribution.ndim != 3 or distribution.shape[-1] != 4
                or valid.shape != distribution.shape[:2]
                or cosine.shape != valid.shape or valid.dtype != torch.bool):
            raise ValueError('Invalid support posterior tensors')
        if any(value.device != base_logits.device
               for value in (distribution, valid, cosine)):
            raise ValueError('Posterior tensors must share a device')
        supported = valid.any(-1)
        safe_labels = torch.where(valid[..., None], distribution,
                                  torch.zeros_like(distribution))
        scores = torch.where(valid, cosine / float(temperature),
                             torch.full_like(cosine, -torch.inf))
        weights = torch.zeros_like(scores)
        if supported.any():
            weights[supported] = torch.softmax(scores[supported], dim=-1)
        posterior = (weights[..., None] * safe_labels).sum(1)
        base_probability = torch.softmax(base_logits.float(), dim=-1)
        posterior = torch.where(supported[:, None], posterior, base_probability)
        posterior = posterior.clamp_min(torch.finfo(posterior.dtype).tiny)
        posterior = posterior / posterior.sum(-1, keepdim=True)
        return posterior, base_probability, weights, supported

    def gate_features(self, query, base_probability, posterior, valid, cosine,
                      purity, support_entropy, supported, mode):
        """Construct the documented query/retrieval reliability vector."""
        if mode not in POSTERIOR_MODES[1:]:
            raise ValueError('Gate features require a posterior mode')
        if purity.shape != valid.shape or support_entropy.shape != valid.shape:
            raise ValueError('Support quality dimensions disagree')
        eps = torch.finfo(base_probability.dtype).tiny
        log_classes = math.log(4.)
        base_entropy = -(base_probability * base_probability.clamp_min(eps).log()).sum(-1)
        base_entropy = (base_entropy / log_classes).clamp(0., 1.)
        top2 = base_probability.topk(2, dim=-1).values
        base_margin = top2[:, 0] - top2[:, 1]

        safe_cosine = torch.where(valid, cosine, torch.full_like(cosine, -torch.inf))
        ordered = safe_cosine.sort(dim=-1, descending=True).values
        # Replacing padding infinities before subtraction avoids inf-inf even
        # when a query has no support or only one support.
        finite_ordered = torch.where(torch.isfinite(ordered), ordered,
                                     torch.zeros_like(ordered))
        top1 = torch.where(supported, finite_ordered[:, 0],
                           torch.zeros_like(finite_ordered[:, 0]))
        enough = valid.sum(-1) > 1
        gap = torch.where(enough,
                          finite_ordered[:, 0] - finite_ordered[:, 1],
                          torch.zeros_like(finite_ordered[:, 0]))
        post_confidence = posterior.max(-1).values
        post_entropy = -(posterior * posterior.clamp_min(eps).log()).sum(-1)
        post_reliability = (1. - post_entropy / log_classes).clamp(0., 1.)
        valid_fraction = valid.float().mean(-1)
        denominator = valid.sum(-1).clamp_min(1).float()
        safe_purity = torch.where(valid, purity, torch.zeros_like(purity))
        safe_entropy = torch.where(valid, support_entropy,
                                   torch.zeros_like(support_entropy))
        mean_purity = safe_purity.sum(-1) / denominator
        support_reliability = torch.where(
            valid, (1. - safe_entropy / log_classes).clamp(0., 1.),
            torch.zeros_like(safe_entropy))
        mean_support_reliability = support_reliability.sum(-1) / denominator
        retrieval = torch.stack((top1, gap, post_confidence, post_reliability,
                                 valid_fraction, mean_purity,
                                 mean_support_reliability), dim=-1)
        retrieval = torch.where(supported[:, None], retrieval,
                                torch.zeros_like(retrieval))
        if mode == 'posterior_query_gate':
            retrieval = torch.zeros_like(retrieval)
        query_part = F.normalize(self.query_norm(query).float(), dim=-1)
        return torch.cat((query_part, base_entropy[:, None],
                          base_margin[:, None], retrieval), dim=-1)

    def forward(self, query, base_logits, distribution=None, valid=None,
                cosine=None, purity=None, support_entropy=None,
                mode='posterior_quality_gate', temperature=.07,
                return_diagnostics=False):
        if (mode not in POSTERIOR_MODES or query.ndim != 2
                or query.shape[-1] != self.dim
                or base_logits.shape != (len(query), 4)
                or query.device != base_logits.device):
            raise ValueError('Invalid selective-posterior request')
        query_delta = self.query_residual(self.query_norm(query))
        if mode == 'query_only':
            logits = base_logits + query_delta
            diagnostics = dict(
                supported=torch.zeros(len(query), dtype=torch.bool, device=query.device),
                posterior_confidence=torch.zeros(len(query), device=query.device),
                posterior_entropy=torch.zeros(len(query), device=query.device),
                gate=torch.zeros(len(query), device=query.device),
                strength=self.strength.detach().expand(len(query)),
                posterior_changed=torch.zeros(len(query), dtype=torch.bool,
                                              device=query.device))
            return (logits, diagnostics) if return_diagnostics else logits
        if any(value is None for value in
               (distribution, valid, cosine, purity, support_entropy)):
            raise ValueError('Posterior modes require retrieved labels and quality')
        posterior, base_probability, _, supported = self._posterior(
            base_logits, distribution, valid, cosine, temperature)
        features = self.gate_features(
            query, base_probability, posterior, valid, cosine, purity,
            support_entropy, supported, mode)
        if mode == 'posterior_nogate':
            gate = supported.float()
        else:
            gate = torch.sigmoid(self.gate(features).squeeze(-1)) * supported.float()
        eps = torch.finfo(posterior.dtype).tiny
        # A one-hot support posterior otherwise produces an approximately
        # -87 log-ratio for absent classes in FP32.  Bound the correction so a
        # single support cannot overwhelm the frozen source logits; subtracting
        # the class mean removes a softmax-invariant common offset.
        log_ratio = (posterior.clamp_min(eps).log()
                     - base_probability.clamp_min(eps).log()).clamp(-5., 5.)
        log_ratio = log_ratio - log_ratio.mean(-1, keepdim=True)
        logits = base_logits + query_delta + self.strength * gate[:, None] * log_ratio
        entropy = -(posterior * posterior.clamp_min(eps).log()).sum(-1) / math.log(4.)
        diagnostics = dict(
            supported=supported,
            posterior_confidence=posterior.max(-1).values,
            posterior_entropy=entropy,
            gate=gate,
            strength=self.strength.detach().expand(len(query)),
            posterior_changed=posterior.argmax(-1) != base_probability.argmax(-1))
        return (logits, diagnostics) if return_diagnostics else logits


def make_posterior_heads(dim, device):
    """Independent posterior controls with identical tensors and state."""
    prototype = SelectivePosteriorHead(dim).to(device)
    return {name: copy.deepcopy(prototype) for name in POSTERIOR_MODES}


def _new_statistics():
    return dict(
        queries=0, supported=0, neighbor_slots=0, valid_neighbors=0,
        valid_neighbors_by_class=[0] * 4,
        queries_with_class_by_class=[0] * 4,
        changed_regions=0, scalar_values=0,
        delta_abs_sum=0., base_abs_sum=0.,
        selected_hard_count_sum=0., selected_labelled_mass_sum=0.,
        selected_purity_sum=0., selected_entropy_sum=0.,
        selected_cosine_sum=0.,
        posterior_confidence_sum=0., posterior_entropy_sum=0.,
        gate_sum=0., positive_gate_queries=0, strength_query_sum=0.,
        posterior_changed_regions=0)


def _accumulate_retrieval(sums, valid, metadata):
    sums[0] += valid.any(-1).sum()
    sums[1] += valid.sum()
    for cls in range(4):
        selected = valid & (metadata['dominant_class'] == cls)
        sums[5 + cls] += selected.sum()
        sums[9 + cls] += selected.any(-1).sum()
    for offset, key in enumerate(('hard_count', 'labelled_mass', 'purity', 'entropy')):
        safe = torch.where(valid, metadata[key].float(),
                           torch.zeros_like(metadata[key], dtype=torch.float))
        sums[13 + offset] += safe.sum()
    safe_cosine = torch.where(valid, metadata['cosine'].float(),
                              torch.zeros_like(metadata['cosine'], dtype=torch.float))
    sums[17] += safe_cosine.sum()


def _accumulate_output(sums, logits, base_logits):
    sums[2] += (logits.argmax(-1) != base_logits.argmax(-1)).sum()
    sums[3] += (logits.detach().float() - base_logits.detach().float()).abs().sum()
    sums[4] += base_logits.detach().float().abs().sum()


def _accumulate_posterior(sums, diagnostics):
    supported = diagnostics['supported']
    sums[18] += torch.where(supported, diagnostics['posterior_confidence'],
                            torch.zeros_like(diagnostics['posterior_confidence'])).sum()
    sums[19] += torch.where(supported, diagnostics['posterior_entropy'],
                            torch.zeros_like(diagnostics['posterior_entropy'])).sum()
    sums[20] += diagnostics['gate'].sum()
    sums[21] += (diagnostics['gate'] > 0).sum()
    sums[22] += diagnostics['strength'].sum()
    sums[23] += diagnostics['posterior_changed'].sum()


def _finish_call_statistics(statistics, sums):
    values = sums.detach().cpu().tolist()
    statistics.update(
        supported=int(values[0]), valid_neighbors=int(values[1]),
        changed_regions=int(values[2]), delta_abs_sum=values[3],
        base_abs_sum=values[4],
        valid_neighbors_by_class=[int(x) for x in values[5:9]],
        queries_with_class_by_class=[int(x) for x in values[9:13]],
        selected_hard_count_sum=values[13],
        selected_labelled_mass_sum=values[14],
        selected_purity_sum=values[15], selected_entropy_sum=values[16],
        selected_cosine_sum=values[17],
        posterior_confidence_sum=values[18],
        posterior_entropy_sum=values[19], gate_sum=values[20],
        positive_gate_queries=int(values[21]), strength_query_sum=values[22],
        posterior_changed_regions=int(values[23]))
    return statistics


def _validate_regions(tokens, mass, base_logits, names, event_names, query_chunk):
    if (tokens.ndim != 3 or mass.shape != tokens.shape[:2]
            or base_logits.shape != (*tokens.shape[:2], 4)
            or len(names) != len(tokens) or len(event_names) != len(names)
            or query_chunk < 1 or mass.device != tokens.device
            or base_logits.device != tokens.device):
        raise ValueError('Invalid region classification tensors/names')


def classify_residual_regions(head, tokens, mass, base_logits, bank, names,
                              event_names, mode, topk=8, query_chunk=256,
                              exclude_same_event=True, intervention='normal',
                              generator=None, preselect=32, per_class_cap=4,
                              min_cosine=-1.):
    """Run matched learned-slot/retrieval residual heads on occupied regions."""
    if mode not in RETRIEVAL_MODES or intervention not in INTERVENTIONS:
        raise ValueError('Unknown residual control')
    _validate_regions(tokens, mass, base_logits, names, event_names, query_chunk)
    if tokens.shape[-1] != head.dim or (mode != 'nomemory' and bank is None):
        raise ValueError('Residual head/bank mismatch')
    images = {} if bank is None else bank['image_lookup']
    events = {} if bank is None else bank['event_lookup']
    result, statistics = [], _new_statistics()
    sums = torch.zeros(24, dtype=torch.float64, device=tokens.device)
    for image_index, (name, event_name) in enumerate(zip(names, event_names)):
        active = (mass[image_index] > 0).nonzero().flatten()
        if not len(active):
            raise ValueError('Image has no occupied regions')
        event_id, image_id = events.get(event_name, -1), images.get(name, -1)
        query_bank, action = bank, intervention
        if mode != 'nomemory' and intervention == 'shuffle_labels':
            query_bank = shuffled_bank_labels(
                bank, event_id, image_id, exclude_same_event, mode, generator)
            action = 'normal'
        outputs = []
        for begin in range(0, len(active), query_chunk):
            selected = active[begin:begin + query_chunk]
            query = tokens[image_index, selected]
            base = base_logits[image_index, selected]
            if mode == 'nomemory':
                logits = head(query, base, mode='nomemory')
            else:
                support, labels, valid, metadata = retrieve(
                    query, query_bank, event_id, image_id, mode, topk,
                    preselect, per_class_cap, min_cosine,
                    exclude_same_event, action, generator)
                logits = head(query, base, support, labels, valid, mode='memory')
                _accumulate_retrieval(sums, valid, metadata)
                statistics['neighbor_slots'] += valid.numel()
            _accumulate_output(sums, logits, base)
            statistics['queries'] += len(query)
            statistics['scalar_values'] += base.numel()
            outputs.append(logits)
        result.append(base_logits[image_index].index_copy(0, active,
                                                          torch.cat(outputs)))
    return torch.stack(result), _finish_call_statistics(statistics, sums)


def classify_posterior_regions(head, tokens, mass, base_logits, bank, names,
                               event_names, mode,
                               retrieval_policy='adaptive_cap', topk=8,
                               query_chunk=256, exclude_same_event=True,
                               intervention='normal', generator=None,
                               preselect=32, per_class_cap=4, min_cosine=-1.,
                               temperature=.07):
    """Apply the explicit support posterior without changing routing/unpooling."""
    if mode not in POSTERIOR_MODES or intervention not in INTERVENTIONS:
        raise ValueError('Unknown posterior control')
    if retrieval_policy not in RETRIEVAL_POLICIES or temperature <= 0:
        raise ValueError('Unknown posterior retrieval policy/temperature')
    _validate_regions(tokens, mass, base_logits, names, event_names, query_chunk)
    if tokens.shape[-1] != head.dim or (mode != 'query_only' and bank is None):
        raise ValueError('Posterior head/bank mismatch')
    images = {} if bank is None else bank['image_lookup']
    events = {} if bank is None else bank['event_lookup']
    result, statistics = [], _new_statistics()
    sums = torch.zeros(24, dtype=torch.float64, device=tokens.device)
    for image_index, (name, event_name) in enumerate(zip(names, event_names)):
        active = (mass[image_index] > 0).nonzero().flatten()
        if not len(active):
            raise ValueError('Image has no occupied regions')
        event_id, image_id = events.get(event_name, -1), images.get(name, -1)
        query_bank, action = bank, intervention
        if mode != 'query_only' and intervention == 'shuffle_labels':
            query_bank = shuffled_bank_labels(
                bank, event_id, image_id, exclude_same_event,
                retrieval_policy, generator)
            action = 'normal'
        outputs = []
        for begin in range(0, len(active), query_chunk):
            selected = active[begin:begin + query_chunk]
            query = tokens[image_index, selected]
            base = base_logits[image_index, selected]
            if mode == 'query_only':
                logits, diagnostics = head(
                    query, base, mode=mode, return_diagnostics=True)
            else:
                _, labels, valid, metadata = retrieve(
                    query, query_bank, event_id, image_id, retrieval_policy,
                    topk, preselect, per_class_cap, min_cosine,
                    exclude_same_event, action, generator)
                logits, diagnostics = head(
                    query, base, labels, valid, metadata['cosine'],
                    metadata['purity'], metadata['entropy'], mode,
                    temperature, return_diagnostics=True)
                _accumulate_retrieval(sums, valid, metadata)
                statistics['neighbor_slots'] += valid.numel()
            _accumulate_output(sums, logits, base)
            _accumulate_posterior(sums, diagnostics)
            statistics['queries'] += len(query)
            statistics['scalar_values'] += base.numel()
            outputs.append(logits)
        result.append(base_logits[image_index].index_copy(0, active,
                                                          torch.cat(outputs)))
    return torch.stack(result), _finish_call_statistics(statistics, sums)


def add_statistics(total, row):
    """Aggregate raw counts/sums while preserving floating diagnostics."""
    for key, value in row.items():
        if isinstance(value, list):
            target = total.setdefault(key, [0] * len(value))
            for index, item in enumerate(value):
                target[index] += int(item)
        elif isinstance(value, float):
            total[key] = total.get(key, 0.) + value
        else:
            total[key] = total.get(key, 0) + int(value)


def finish_statistics(total):
    queries = total.get('queries', 0)
    neighbors = total.get('valid_neighbors', 0)
    scalars = total.get('scalar_values', 0)
    counts = total.get('valid_neighbors_by_class', [0] * 4)
    covered = total.get('queries_with_class_by_class', [0] * 4)
    result = dict(
        **total,
        support_coverage=total.get('supported', 0) / max(1, queries),
        mean_valid_neighbors=neighbors / max(1, queries),
        class_basis='stored Train support dominant class before label intervention',
        by_class={name: dict(
            valid_neighbors=counts[index],
            mean_valid_neighbors=counts[index] / max(1, queries),
            query_coverage=covered[index] / max(1, queries))
            for index, name in enumerate(CLASS_NAMES)},
        selected_quality_means={
            key: total.get('selected_' + key + '_sum', 0.) / max(1, neighbors)
            for key in ('hard_count', 'labelled_mass', 'purity', 'entropy', 'cosine')},
        correction_abs_mean=total.get('delta_abs_sum', 0.) / max(1, scalars),
        base_abs_mean=total.get('base_abs_sum', 0.) / max(1, scalars),
        correction_to_base_abs_ratio=(total.get('delta_abs_sum', 0.) /
                                      max(1e-12, total.get('base_abs_sum', 0.))),
        region_prediction_change_rate=(total.get('changed_regions', 0) /
                                       max(1, queries)),
        posterior_confidence_mean=(total.get('posterior_confidence_sum', 0.) /
                                   max(1, total.get('supported', 0))),
        posterior_entropy_mean=(total.get('posterior_entropy_sum', 0.) /
                                max(1, total.get('supported', 0))),
        gate_mean=total.get('gate_sum', 0.) / max(1, queries),
        positive_gate_fraction=(total.get('positive_gate_queries', 0) /
                                max(1, queries)),
        strength_mean=total.get('strength_query_sum', 0.) / max(1, queries),
        posterior_class_change_rate=(total.get('posterior_changed_regions', 0) /
                                     max(1, queries)))
    return result


def assert_no_query_target_interfaces():
    forbidden = {'gt', 'target', 'query_gt', 'query_label', 'query_labels'}
    functions = (retrieve, classify_residual_regions,
                 classify_posterior_regions, SelectivePosteriorHead.forward,
                 SelectivePosteriorHead.gate_features)
    for function in functions:
        if forbidden & set(inspect.signature(function).parameters):
            raise RuntimeError('Query targets must never enter retrieval/classification')


# Short aliases for orchestration code that treats stage names dynamically.
make_heads = make_retrieval_heads
classify_regions = classify_residual_regions

