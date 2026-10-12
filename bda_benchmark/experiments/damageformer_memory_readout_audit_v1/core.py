"""Instrumentation only. Prediction stays on the original need_weights=False path.

Attention-weighted labels are explanatory summaries, NOT the learned reader's
posterior or causal attribution. No query GT enters any forward/intervention.
"""
import torch
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import valid_read_route
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import unpool

CLASSES = ('Background', 'Intact', 'Damaged', 'Destroyed')


@torch.no_grad()
def hard_counts(route):
    q, ids = route['q'], route['ids']
    if ids.ndim == 2:
        ids = ids[None].expand(len(q), -1, -1)
    if q.ndim != 3 or ids.shape != q.shape or not torch.isfinite(q).all():
        raise ValueError('Invalid routing')
    if (q < 0).any() or not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-5):
        raise ValueError('Routing must be normalized over candidates')
    winners = ids.gather(1, q.argmax(1, keepdim=True)).squeeze(1)
    counts = torch.zeros((len(q), route['k']), dtype=torch.long, device=q.device)
    counts.scatter_add_(1, winners, torch.ones_like(winners))
    return counts


@torch.no_grad()
def trace_reader(head, query, base, support, labels, valid):
    """Capture the real MHA inputs, then observe per-head weights separately.

    The first forward supplies ALL predictions. The second forward requests
    weights, checks numerical agreement, and cannot replace predictions.
    This avoids silently switching the production attention kernel for audit.
    """
    if head.training or any(p.requires_grad or p.grad is not None for p in head.parameters()):
        raise ValueError('Reader must be eval, frozen, and without gradients')
    if head.attention.dropout != 0:
        raise ValueError('This audit requires the trained dropout=0 reader')
    captured = {}
    def before(module, args, kwargs):
        captured['args'], captured['kwargs'] = args, dict(kwargs)
    def after(module, args, output):
        captured['context'] = output[0]
    pre = head.attention.register_forward_pre_hook(before, with_kwargs=True)
    post = head.attention.register_forward_hook(after)
    try:
        logits = head(query, base, support, labels, valid, mode='memory')
    finally:
        pre.remove(); post.remove()
    options = dict(captured['kwargs'], need_weights=True, average_attn_weights=False)
    context, weights = head.attention(*captured['args'], **options)
    replay = base + head.ffn(head.norm(query + context[:, 0]))
    torch.testing.assert_close(context, captured['context'], atol=2e-5, rtol=2e-4,
                               msg='Weight-observation context differs from actual reader')
    torch.testing.assert_close(replay, logits, atol=2e-5, rtol=2e-4,
                               msg='Weight-observation replay differs from actual reader')
    weights = weights[:, :, 0]
    if not torch.isfinite(logits).all() or not torch.isfinite(weights).all():
        raise RuntimeError('Nonfinite reader trace')
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]), atol=1e-5, rtol=1e-5)
    return dict(logits=logits, observation_logits=replay, weights=weights,
                context_error=float((context-captured['context']).abs().max()),
                logit_error=float((replay-logits).abs().max()),
                argmax_disagreements=int((replay.argmax(-1) != logits.argmax(-1)).sum()))


@torch.no_grad()
def split_corrections(base, normal, counts):
    if normal.shape != base.shape or base.shape[:-1] != counts.shape:
        raise ValueError('Slot shapes disagree')
    hard = counts > 0
    return dict(hard_only=torch.where(hard[..., None], normal, base),
                tail_only=torch.where(hard[..., None], base, normal))


@torch.no_grad()
def read_tail_fraction(route, mass, counts, output_size):
    valid_route = valid_read_route(route, mass)
    tail = (counts == 0) & (mass > 0)
    slot = torch.stack((~tail, tail), dim=-1).float()
    fraction = unpool(slot, valid_route['q'], valid_route['ids'],
                      valid_route['shape'], output_size)[:, 1]
    if not torch.isfinite(fraction).all() or (fraction < -1e-6).any() or (fraction > 1+1e-6).any():
        raise RuntimeError('Invalid soft-tail read fraction')
    return fraction.clamp(0, 1)


@torch.no_grad()
def reader_columns(base, trace, labels, valid, cosine, target, labelled_mass, temperature=.07):
    """GT used only AFTER original predictions to construct diagnostic columns."""
    logits, weights = trace['logits'], trace['weights']
    attention = weights.mean(1)
    external = attention[:, :-1]
    external_mass = external.sum(-1)
    truth = target.argmax(-1)
    safe_labels = torch.where(valid[..., None], labels, torch.zeros_like(labels))
    own_label_mass = safe_labels.gather(2, truth[:, None, None].expand(-1, labels.shape[1], 1)).squeeze(-1)
    denom = external_mass.clamp_min(1e-12)
    true_mass = (external * own_label_mass).sum(-1) / denom
    same_class = ((safe_labels.argmax(-1) == truth[:, None]) & valid).float()
    any_support = valid.any(-1)
    scores = (torch.where(valid, cosine, torch.zeros_like(cosine)) / temperature).masked_fill(~valid, -1e9)
    cw = scores.softmax(-1) * valid
    cw = cw / cw.sum(-1, keepdim=True).clamp_min(1e-12)
    delta = logits-base
    centred = delta-delta.mean(-1, keepdim=True)
    return dict(
        labelled=labelled_mass > 0, truth=truth, supported=any_support,
        attention_label_defined=any_support & (external_mass > 1e-12),
        query_attention=attention[:, -1], external_attention=external_mass,
        external_label_true_mass=true_mass,
        cosine_label_true_mass=(cw*own_label_mass).sum(-1),
        same_class_external_attention=(external*same_class).sum(-1),
        delta_abs=delta.abs().mean(-1), centered_delta_abs=centred.abs().mean(-1),
        prob_l1=(logits.softmax(-1)-base.softmax(-1)).abs().sum(-1),
        changed_region=(logits.argmax(-1) != base.argmax(-1)).float(),
        source_region_correct=(base.argmax(-1) == truth).float(),
        reader_region_correct=(logits.argmax(-1) == truth).float(),
        attention_by_head=weights[:, :, -1],
    )


class ReaderStats:
    """Slot means and pooling-mass means separately; neither is pixel accuracy."""
    GENERAL = ('query_attention', 'external_attention', 'delta_abs', 'centered_delta_abs', 'prob_l1', 'changed_region')

    def __init__(self):
        self.data = {}

    @torch.no_grad()
    def update(self, case, base, trace, labels, valid, cosine, target, labelled_mass, mass, counts, temperature=.07):
        cols = reader_columns(base, trace, labels, valid, cosine, target, labelled_mass, temperature)
        masks = dict(all=torch.ones_like(counts, dtype=torch.bool), hard=counts>0, tail=counts==0)
        for cls, name in enumerate(CLASSES):
            mask = cols['labelled'] & (cols['truth'] == cls)
            masks[name] = mask
            masks['hard__'+name] = mask & (counts>0)
            masks['tail__'+name] = mask & (counts==0)
        for group, mask in masks.items():
            row = self.data.setdefault(case, {}).setdefault(group, dict(queries=0, labelled_queries=0,
                supported_queries=0, attention_label_queries=0, mass=0., labelled_mass=0., sums={}, head_sum=None))
            n = int(mask.sum()); row['queries'] += n
            row['mass'] += float(mass[mask].double().sum())
            lm = mask & cols['labelled']; sm = lm & cols['supported']; am = lm & cols['attention_label_defined']
            row['labelled_queries'] += int(lm.sum()); row['supported_queries'] += int(sm.sum())
            row['attention_label_queries'] += int(am.sum())
            row['labelled_mass'] += float(labelled_mass[mask].double().sum())
            for key in self.GENERAL:
                row['sums'][key] = row['sums'].get(key, 0.) + float(cols[key][mask].double().sum())
                mk = key+'_mass_weighted'
                row['sums'][mk] = row['sums'].get(mk, 0.) + float((cols[key][mask].double()*mass[mask]).sum())
            for key in ('source_region_correct','reader_region_correct'):
                row['sums'][key] = row['sums'].get(key,0.) + float(cols[key][lm].double().sum())
            for key, selected in (('external_label_true_mass',am),('cosine_label_true_mass',sm),('same_class_external_attention',lm)):
                row['sums'][key] = row['sums'].get(key,0.) + float(cols[key][selected].double().sum())
            hs = cols['attention_by_head'][mask].double().sum(0).cpu()
            row['head_sum'] = hs if row['head_sum'] is None else row['head_sum']+hs

    def finish(self):
        def avg(x,n): return x/n if n else None
        result = {}
        for case, groups in self.data.items():
            result[case] = {}
            for name,row in groups.items():
                n, sums = row['queries'], row['sums']
                out = {k:row[k] for k in ('queries','labelled_queries','supported_queries','attention_label_queries','mass','labelled_mass')}
                for key in self.GENERAL:
                    suffix = 'fraction' if key == 'changed_region' else 'mean'
                    out[key+'_'+suffix] = avg(sums[key],n)
                    out[key+'_mass_weighted'] = avg(sums[key+'_mass_weighted'],row['mass'])
                for key in ('source','reader'):
                    out[key+'_region_accuracy'] = avg(sums[key+'_region_correct'],row['labelled_queries'])
                for key, denominator in (('external_label_true_mass',row['attention_label_queries']),
                    ('cosine_label_true_mass',row['supported_queries']),('same_class_external_attention',row['labelled_queries'])):
                    out[key+('_mean' if key == 'same_class_external_attention' else '')] = avg(sums[key],denominator)
                out['attention_by_head'] = [avg(float(v),n) for v in row['head_sum']]
                result[case][name] = out
        return result
