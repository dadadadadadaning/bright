"""Final-Val retrieval diagnostics only; GT never selects a support.

Region GT compositions are label mixtures, not within-building damage maps.
These region-weighted summaries are NOT segmentation mIoU or its upper bound.
"""
import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_roi_followup_v1.memory import (
    retrieve, target_distribution,
)

CLASS_NAMES = ('Background', 'Intact', 'Damaged', 'Destroyed')
FIELDS = dict(queries=0, supported_queries=0, valid_neighbors=0,
              vote_agreement_queries=0, prior_eligible_queries=0,
              neighbor_class_mass_sum=0., eligible_prior_class_mass_sum=0.,
              mean_cosine_sum=0.)


@torch.no_grad()
def retrieval_quality(tokens, mass, route, gt, bank, names, event_names,
                      topk=8, query_chunk=256, exclude_same_event=True):
    """Feature-only retrieval FIRST, diagnostic GT comparison afterwards.

    Each labelled query contributes once to its majority-GT-class bucket.
    Neighbor votes are uniform means of valid supports' soft labels. The chance
    reference is the soft-label mean of the same eligible bank, with identical
    same-image/event exclusions. It is not an independently trained classifier.
    Raw counts/sums are additive across batches and JSON serializable.
    """
    if (tokens.ndim != 3 or mass.shape != tokens.shape[:2]
            or len(names) != len(tokens) or len(event_names) != len(tokens)
            or query_chunk < 1 or topk < 1):
        raise ValueError('Invalid retrieval diagnostic inputs')
    images = bank.get('image_lookup') or {name: i for i, name in enumerate(bank['image_names'])}
    events = bank.get('event_lookup') or {name: i for i, name in enumerate(bank['event_names'])}
    records = []
    priors = []
    for index, (name, event_name) in enumerate(zip(names, event_names)):
        image_id, event_id = images.get(name, -1), events.get(event_name, -1)
        allowed = bank['images'] != image_id
        if exclude_same_event:
            allowed = allowed & (bank['events'] != event_id)
        eligible = bank['distribution'][allowed]
        priors.append(eligible.mean(0) if len(eligible) else None)
        active = (mass[index] > 0).nonzero().flatten()
        for begin in range(0, len(active), query_chunk):
            selected = active[begin:begin + query_chunk]
            query = tokens[index, selected]
            support, labels, valid = retrieve(query, bank, event_id, image_id,
                topk=topk, exclude_same_event=exclude_same_event, intervention='normal')
            cosine = (F.normalize(query.float(), dim=-1)[:, None]
                      * F.normalize(support.float(), dim=-1)).sum(-1)
            records.append((index, selected, labels, valid, cosine))
    # Nothing above reads query GT. The exact same returned neighbors would be
    # used with entirely different GT labels; tests enforce that separation.
    query_distribution, labelled_mass = target_distribution(gt, route)
    if query_distribution.shape != (*tokens.shape[:2], 4):
        raise ValueError('Routing/GT region count differs from frozen tokens')
    result = dict(images=len(tokens), classes={name: dict(FIELDS) for name in CLASS_NAMES})
    for index, selected, labels, valid, cosine in records:
        labelled = labelled_mass[index, selected] > 0
        classes = query_distribution[index, selected].argmax(-1)
        counts = valid.sum(-1)
        supported = counts > 0
        vote = (labels * valid[..., None]).sum(1) / counts.clamp_min(1)[:, None]
        mean_cosine = (cosine * valid).sum(1) / counts.clamp_min(1)
        for cls, name in enumerate(CLASS_NAMES):
            mask = labelled & (classes == cls)
            with_support = mask & supported
            row = result['classes'][name]
            row['queries'] += int(mask.sum())
            row['supported_queries'] += int(with_support.sum())
            row['valid_neighbors'] += int(counts[mask].sum())
            row['vote_agreement_queries'] += int((with_support & (vote.argmax(-1) == cls)).sum())
            row['neighbor_class_mass_sum'] += float(vote[with_support, cls].double().sum())
            row['mean_cosine_sum'] += float(mean_cosine[with_support].double().sum())
            if priors[index] is not None:
                row['prior_eligible_queries'] += int(mask.sum())
                row['eligible_prior_class_mass_sum'] += int(mask.sum()) * float(priors[index][cls])
    return result


def add_quality(total, row):
    total['images'] = total.get('images', 0) + row['images']
    classes = total.setdefault('classes', {name: dict(FIELDS) for name in CLASS_NAMES})
    for name in CLASS_NAMES:
        for key, value in row['classes'][name].items():
            classes[name][key] += value


def finish_quality(total):
    def finish(row):
        q, supported, eligible = row['queries'], row['supported_queries'], row['prior_eligible_queries']
        return dict(row,
            support_coverage=supported/q if q else None,
            mean_valid_neighbors=row['valid_neighbors']/q if q else None,
            neighbor_query_class_mass=row['neighbor_class_mass_sum']/supported if supported else None,
            neighbor_vote_agreement=row['vote_agreement_queries']/supported if supported else None,
            eligible_bank_query_class_prior=row['eligible_prior_class_mass_sum']/eligible if eligible else None,
            mean_query_neighbor_cosine=row['mean_cosine_sum']/supported if supported else None)
    classes = total.get('classes', {name: dict(FIELDS) for name in CLASS_NAMES})
    combined = {key: sum(row[key] for row in classes.values()) for key in FIELDS}
    return dict(images=total.get('images', 0), all=finish(combined),
        by_class={name: finish(row) for name, row in classes.items()},
        weighting='one vote per labelled occupied region; class bucket is query majority GT',
        caution='Final-Val diagnostic only, not mIoU, not instance labels, not a training objective; GT never ranks supports')
