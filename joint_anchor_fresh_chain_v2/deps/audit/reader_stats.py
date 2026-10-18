"""Read-only reconstruction of the actual trained Reader attention.

No labels enter attention scores. Attention-weighted labels describe exposure,
not an explanation of causality or a replacement for the classifier output.
"""
import torch


@torch.no_grad()
def reader_diagnostics(head, result, support, *, chunk_size=32):
    query = result['slot_encoding'][0]
    valid_query = support['slot_valid'][0]
    ids = support['support_ids']
    names = ('background_label_mass', 'intact_label_mass',
             'damaged_label_mass', 'destroyed_label_mass',
             'background_role_attention', 'tiny_fraction_attention',
             'effective_slots', 'has_support')
    # The trained Reader returns a zero context before gathering an empty table.
    # There is no dummy candidate to read when M=0; -1 padding must not index 0.
    if support['support_valid'].shape[0] == 0:
        if bool((ids != -1).any()):
            raise ValueError('Nonpadding support ID with empty support table')
        expected = result['memory_context'][0]
        if not torch.allclose(query * 0., expected, atol=2e-5, rtol=2e-5):
            raise ValueError('Reader attention reconstruction differs from actual context')
        rows = {name: query.new_full((len(query),), float('nan')) for name in names}
        rows['has_support'] = query.new_zeros(len(query))
        return rows
    table = head._encode_slots(support['support_f'], support['support_t'],
                               support['support_fraction'], support['support_valid'])
    keys = head.key_projection(table)
    clean_labels = torch.where(support['support_valid'][..., None], support['support_labels'], 0.)
    values = head.value_projection(table + head.label_embedding(clean_labels))
    rows = {k: [] for k in names}
    contexts = []
    width, heads = head.hidden, head.heads
    dim = width // heads
    for start in range(0, len(query), chunk_size):
        end = min(start + chunk_size, len(query))
        chosen = ids[start:end]
        safe = chosen.clamp_min(0)
        mask = ((chosen[..., None] >= 0) & support['support_valid'][safe]).flatten(1)
        k = keys[safe].reshape(end-start, -1, heads, dim)
        v = values[safe].reshape(end-start, -1, heads, dim)
        q = head.query_projection(query[start:end]).reshape(end-start, 2, heads, dim)
        scores = torch.einsum('rqhd,rkhd->rqhk', q, k) * dim**-0.5
        scores = scores.masked_fill(~mask[:, None, None, :], -torch.finfo(scores.dtype).max)
        a = scores.softmax(-1) * mask[:, None, None, :]
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-30)
        context = torch.einsum('rqhk,rkhd->rqhd', a, v).reshape(end-start, 2, width)
        context = head.context_projection(context)
        context = torch.where(valid_query[start:end, :, None], context, 0.)
        contexts.append(context)
        # Building-role query, average across heads; all selected support roles remain.
        att = a[:, 1].mean(1)
        valid = valid_query[start:end, 1] & mask.any(1)
        label = clean_labels[safe].flatten(1, 2)
        weighted = (att[..., None] * label).sum(1)
        fraction = support['support_fraction'][safe].flatten(1)
        role_bg = torch.arange(mask.shape[1], device=query.device) % 2 == 0
        metrics = dict(zip(('background_label_mass', 'intact_label_mass',
                             'damaged_label_mass', 'destroyed_label_mass'), weighted.unbind(1)))
        metrics.update(background_role_attention=att[:, role_bg].sum(1),
                       tiny_fraction_attention=(att * (fraction < .01)).sum(1),
                       effective_slots=1. / att.square().sum(1).clamp_min(1e-30))
        for name, value in metrics.items():
            rows[name].append(torch.where(valid, value, torch.full_like(value, float('nan'))))
        rows['has_support'].append(valid.float())
    replay = torch.cat(contexts)
    expected = result['memory_context'][0]
    if not torch.allclose(replay, expected, atol=2e-5, rtol=2e-5):
        raise ValueError('Reader attention reconstruction differs from actual context')
    return {k: torch.cat(v) for k, v in rows.items()}


def add_posthoc_label_scores(rows, labels, valid):
    """GT is consulted only after the trained attention has been computed."""
    predicted = torch.stack([rows[c+'_label_mass'] for c in
                             ('background', 'intact', 'damaged', 'destroyed')], -1)
    ok = valid[:, 1] & torch.isfinite(predicted).all(-1)
    tv = .5 * (predicted - labels[:, 1]).abs().sum(-1)
    rows['posthoc_label_TV'] = torch.where(ok, tv, torch.full_like(tv, float('nan')))
    return rows
